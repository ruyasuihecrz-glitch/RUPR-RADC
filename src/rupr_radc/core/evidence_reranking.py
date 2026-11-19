import json
import math
import time
import importlib.util

def reader_text(item, sources, prompt):
    # Explicit allowlist; metadata/path/ranking scores never enter the prompt.
    return prompt + '\n\n' + json.dumps({'news': item['news'], 'retrieval_records': sources}, ensure_ascii=False)

class FrozenReranker:
    def __init__(self, config):
        import torch
        spec = importlib.util.spec_from_file_location('official_qwen_reranker', config['reranker_helper'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.Qwen3VLReranker(
            config['reranker_path'], max_length=config['max_input_tokens'],
            min_pixels=config['min_pixels'], max_pixels=config['max_pixels'],
            torch_dtype=torch.bfloat16, attn_implementation=config['attention'], local_files_only=True)
        self.limit = config['max_input_tokens']
        for param in self.native.model.parameters():
            param.requires_grad_(False)

    def pair(self, news, image, text, instruction):
        return self.native.format_mm_instruction(news, image, None, text, None, None,
                                                instruction=instruction)

    def tokenize(self, pair, with_image):
        from qwen_vl_utils import process_vision_info
        proc = self.native.processor
        rendered = proc.apply_chat_template([pair], tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            [pair], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            raise ValueError('Unexpected video in image/text diagnostic')
        inputs = proc(text=rendered, images=images, padding=True, truncation=False,
                      do_resize=False, return_tensors='pt', **video_kwargs)
        has_image = 'pixel_values' in inputs and 'image_grid_thw' in inputs
        if has_image != with_image or (with_image and len(inputs['image_grid_thw']) != 1):
            raise ValueError('Actual query image tensor disagrees with treatment')
        if inputs['input_ids'].shape[1] > self.limit:
            raise ValueError('Reranker input exceeds budget; refusing truncation')
        return inputs.to(self.native.device)

    def score(self, pair, with_image):
        import torch
        inputs = self.tokenize(pair, with_image)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            value = self.native.compute_scores(inputs)[0]
        torch.cuda.synchronize()
        if not math.isfinite(value):
            raise ValueError('Nonfinite reranking score')
        return {'score': value, 'seconds': time.monotonic()-start,
                'input_tokens': int(inputs['attention_mask'].sum()),
                'image_grid_thw': inputs['image_grid_thw'].tolist() if with_image else None}
