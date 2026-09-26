"""Configurable, realistic training augmentations.

Randomness comes from the global torch RNG, so runs are reproducible once
:func:`defect_detection.utils.reproducibility.seed_everything` has been called
and DataLoader workers use ``seed_worker``.
"""

from __future__ import annotations

from torchvision.transforms import v2

from defect_detection.data.preprocessing import (
    PairTransform,
    geometric_resize_ops,
    normalise_ops,
    to_float_ops,
)
from defect_detection.utils.config import AugmentationConfig, PreprocessingConfig


def geometric_augment_ops(cfg: AugmentationConfig) -> list[v2.Transform]:
    """Transforms applied identically to image and mask."""
    ops: list[v2.Transform] = []
    if cfg.hflip_p > 0:
        ops.append(v2.RandomHorizontalFlip(p=cfg.hflip_p))
    if cfg.vflip_p > 0:
        ops.append(v2.RandomVerticalFlip(p=cfg.vflip_p))
    if cfg.rotation_degrees > 0:
        ops.append(v2.RandomRotation(degrees=cfg.rotation_degrees))
    return ops


def photometric_augment_ops(cfg: AugmentationConfig) -> list[v2.Transform]:
    """Image-only transforms; expect a float image in [0, 1]."""
    ops: list[v2.Transform] = []
    if cfg.brightness > 0 or cfg.contrast > 0:
        ops.append(v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))
    if cfg.blur_sigma is not None:
        ops.append(v2.GaussianBlur(kernel_size=5, sigma=cfg.blur_sigma))
    if cfg.noise_std > 0:
        ops.append(v2.GaussianNoise(mean=0.0, sigma=cfg.noise_std, clip=True))
    return ops


def build_train_transform(pre: PreprocessingConfig, aug: AugmentationConfig) -> PairTransform:
    ops = (
        geometric_resize_ops(pre)
        + geometric_augment_ops(aug)
        + to_float_ops()
        + photometric_augment_ops(aug)
        + normalise_ops(pre)
    )
    return PairTransform(v2.Compose(ops))
