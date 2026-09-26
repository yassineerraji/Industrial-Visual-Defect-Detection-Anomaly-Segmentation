"""Threshold selection from normal validation data only (never the test set)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdConfig:
    image_method: str = "mean_std"  # "mean_std" (mean + k*std) or "quantile"
    image_k: float = 3.0
    image_quantile: float = 0.99
    pixel_quantile: float = 0.999  # quantile of normal validation pixel scores


@dataclass(frozen=True)
class Thresholds:
    image: float
    pixel: float
    source: str  # which data and rule produced them

    def to_dict(self) -> dict:
        return asdict(self)


def image_threshold(normal_scores: np.ndarray, cfg: ThresholdConfig) -> float:
    if normal_scores.size == 0:
        raise ValueError("Need normal validation scores to select a threshold")
    if cfg.image_method == "mean_std":
        return float(normal_scores.mean() + cfg.image_k * normal_scores.std())
    if cfg.image_method == "quantile":
        return float(np.quantile(normal_scores, cfg.image_quantile))
    raise ValueError(f"Unknown image threshold method '{cfg.image_method}'")


def pixel_threshold(normal_maps: np.ndarray, cfg: ThresholdConfig) -> float:
    """High quantile of per-pixel scores on normal images: bounds the pixel false-positive rate."""
    return float(np.quantile(normal_maps.ravel(), cfg.pixel_quantile))


def select_thresholds(normal_scores: np.ndarray, normal_maps: np.ndarray, cfg: ThresholdConfig) -> Thresholds:
    if cfg.image_method == "mean_std":
        rule = f"mean+{cfg.image_k}*std"
    else:
        rule = f"quantile {cfg.image_quantile}"
    return Thresholds(
        image=image_threshold(normal_scores, cfg),
        pixel=pixel_threshold(normal_maps, cfg),
        source=f"validation normals (n={normal_scores.size}); image: {rule}; pixel: quantile {cfg.pixel_quantile}",
    )
