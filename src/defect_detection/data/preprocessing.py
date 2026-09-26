"""Deterministic image/mask loading and evaluation-time preprocessing.

Images and masks are wrapped as torchvision ``tv_tensors`` so that every
geometric transform is applied identically to both (masks use nearest-neighbour
interpolation), while photometric transforms touch the image only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from defect_detection.utils.config import PreprocessingConfig


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_mask(path: str | Path | None, size: tuple[int, int]) -> np.ndarray:
    """Load a binary mask as ``uint8`` {0, 1} of shape (H, W).

    ``size`` is the (width, height) of the paired image. Normal samples
    (``path is None``) get an all-zero mask. A size mismatch raises, since it
    would silently misalign ground truth.
    """
    width, height = size
    if path is None:
        return np.zeros((height, width), dtype=np.uint8)
    with Image.open(path) as m:
        if m.size != size:
            raise ValueError(f"Mask {path} has size {m.size}, image has {size}.")
        return (np.asarray(m.convert("L")) > 0).astype(np.uint8)


def geometric_resize_ops(cfg: PreprocessingConfig) -> list[v2.Transform]:
    ops: list[v2.Transform] = [v2.Resize((cfg.image_size, cfg.image_size), antialias=True)]
    if cfg.crop_size is not None:
        ops.append(v2.CenterCrop(cfg.crop_size))
    return ops


class PairTransform:
    """Apply a v2 pipeline jointly to an (image, mask) pair.

    Returns a float image tensor (3, H, W) and a float mask tensor (1, H, W) in {0, 1}.
    """

    def __init__(self, pipeline: v2.Transform) -> None:
        self.pipeline = pipeline

    def __call__(self, image: Image.Image, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        img_t = tv_tensors.Image(F.pil_to_tensor(image))
        mask_t = tv_tensors.Mask(torch.from_numpy(mask).unsqueeze(0))
        img_out, mask_out = self.pipeline(img_t, mask_t)
        return img_out.as_subclass(torch.Tensor), mask_out.as_subclass(torch.Tensor).float()


def to_float_ops() -> list[v2.Transform]:
    return [v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True)]


def normalise_ops(cfg: PreprocessingConfig) -> list[v2.Transform]:
    return [v2.Normalize(mean=list(cfg.mean), std=list(cfg.std))]


def build_eval_transform(cfg: PreprocessingConfig) -> PairTransform:
    """Deterministic transform for validation, test and inference."""
    return PairTransform(v2.Compose(geometric_resize_ops(cfg) + to_float_ops() + normalise_ops(cfg)))


def denormalise(image: torch.Tensor, cfg: PreprocessingConfig) -> torch.Tensor:
    """Invert normalisation for display; returns values clipped to [0, 1]."""
    mean = torch.tensor(cfg.mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(cfg.std, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)
