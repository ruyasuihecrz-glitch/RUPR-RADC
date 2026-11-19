"""Frozen Reader pooling and legal CLIP summaries for CCN calibration."""
import hashlib
import time

import numpy as np
import torch

from .continuous_scores import append_prefix, extract_score, label_tokens
from .evidence_interaction import record_spans
from .prompt_reader import reader_messages


@torch.inference_mode()
def reader_features(backend, row, config, prompt, check=False):
    messages = reader_messages(row, prompt, row['_root'])
    inputs, _, _ = backend.prepare_batch([messages])
    prefix, labels = label_tokens(backend.processor.tokenizer, config)
    inputs = append_prefix(inputs, prefix)
    if inputs['input_ids'].shape[1] > config['max_input_tokens']:
        raise ValueError('Full Reader input exceeds budget')
    spans = record_spans(row, messages, backend.processor.tokenizer, inputs, prompt)
    captured = []

    def capture(module, args, value):
        if value.shape[:2] != inputs['input_ids'].shape:
            raise ValueError('Reader nodes and actual token sequence differ')
        nodes = torch.stack([value[0, s['token_indices']].float().mean(0) for s in spans])
        captured.append((nodes.cpu().numpy(), value[0, -1].float().cpu().numpy().copy()))

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    before = time.monotonic()
    handle = backend.model.model.language_model.norm.register_forward_hook(capture)
    try:
        output = backend.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Expected exactly one pooled Reader forward')
    value = extract_score(output.logits[0, -1], labels)
    value.update(input_tokens=int(inputs['attention_mask'].sum()),
                 prefixed_input_ids_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                 image_grid_thw=inputs['image_grid_thw'].tolist(), spans=spans,
                 forward_seconds=time.monotonic()-before, reader_trainable_parameters=0)
    if check:
        generated = backend.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                          use_cache=True, return_dict_in_generate=True, output_logits=True)
        other = extract_score(generated.logits[0][0], labels)
        error = abs(value['score']-other['score'])
        if error > .02:
            raise ValueError(f'Reader forward/generation mismatch: {error}')
        value['independent_generation_check'] = dict(absolute_error=error, passed=True)
    nodes, answer = captured[0]
    dimension = backend.model.config.text_config.hidden_size
    if nodes.shape != (2+len(row['sources']), dimension) or answer.shape != (dimension,):
        raise ValueError('Unexpected Reader feature dimensions')
    if not np.isfinite(nodes).all() or not np.isfinite(answer).all():
        raise ValueError('Nonfinite frozen Reader features')
    return nodes, answer, value


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def clip_summary(image, text, reference_images, evidence_texts, raw_score):
    image, text = unit(image), unit(text)
    refs = unit(reference_images) if len(reference_images) else np.empty((0, 768), np.float32)
    evidence = unit(evidence_texts) if len(evidence_texts) else np.empty((0, 768), np.float32)
    ri, rt = np.zeros(768, np.float32), np.zeros(768, np.float32)
    image_similarity = 0.
    text_max = text_mean = 0.
    selected = []
    if len(refs):
        similarities = refs @ image
        selected = sorted(range(len(refs)), key=lambda k: (-float(similarities[k]), k))[:4]
        ri = unit(refs[selected].mean(0))
        image_similarity = float(similarities[selected[0]])
    if len(evidence):
        similarities = evidence @ text
        best = int(np.argmax(similarities))
        rt = evidence[best]
        text_max, text_mean = float(similarities[best]), float(similarities.mean())
    return dict(image=image, text=text, reference_image=ri, reference_text=rt,
                clip_present=np.array([True, True, bool(len(refs)), bool(len(evidence))]),
                signals=np.array([raw_score, float(image @ text), image_similarity,
                                  text_max, text_mean, len(evidence)/12.], np.float32)), selected
