"""Frozen OpenAI CLIP, with the official AITR image preparation recipe."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import digest


class FrozenAITRClip:
    """Read existing local weights; return raw 768-dimensional CLIP features."""

    def __init__(self, checkpoint, expected_sha256, device='cpu'):
        import open_clip

        path = Path(checkpoint)
        if not path.is_file() or digest(path) != expected_sha256:
            raise ValueError('CLIP checkpoint missing or checksum mismatch')
        # Keep the original OpenAI loader: constructing a generic CLIP and
        # passing a local checkpoint can silently select a different GELU.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', precision='fp32',
            device=device, cache_dir=str(path.parent))
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.model.eval().requires_grad_(False)
        self.device = device

    def image_tensor(self, path):
        with Image.open(path) as original:
            image = original.convert('RGB')
        width, height = image.size
        if width > height:
            size = (400, int(height * (400 / width)))
        else:
            size = (int(width * (400 / height)), 400)
        # Matches DatasetIteratorSource and ImageIteratorSource upstream.
        return self.preprocess(image.resize(size))

    @staticmethod
    def checked_features(value, count):
        result = value.float().cpu().numpy()
        if result.shape != (count, 768) or not np.isfinite(result).all():
            raise ValueError('Invalid CLIP feature shape or non-finite value')
        return result

    @torch.inference_mode()
    def encode_images(self, paths):
        images = torch.stack([self.image_tensor(path) for path in paths])
        return self.checked_features(self.model.encode_image(images.to(self.device)), len(paths))

    @torch.inference_mode()
    def encode_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        return self.checked_features(self.model.encode_text(tokens), len(texts))
