"""Model-agnostic anomaly-map post-processing and image-level scoring.

Any model exposing ``anomaly_map(images) -> (B, 1, H, W)`` can be scored here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import functional as F


@dataclass(frozen=True)
class ScoringConfig:
    smoothing_sigma: float = 4.0  # Gaussian sigma in pixels at model resolution; 0 disables
    image_score: str = "max"  # "max" or "topk_mean"
    topk_fraction: float = 0.001  # fraction of pixels averaged for "topk_mean"


@dataclass
class MapOutputs:
    maps: np.ndarray  # (N, H, W) float32, smoothed
    masks: np.ndarray  # (N, H, W) bool
    labels: np.ndarray  # (N,) int
    scores: np.ndarray  # (N,) float
    paths: list[str]


def smooth_maps(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return maps
    kernel = 2 * int(4 * sigma + 0.5) + 1
    return F.gaussian_blur(maps, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def image_scores(maps: torch.Tensor, cfg: ScoringConfig) -> torch.Tensor:
    """Reduce (B, 1, H, W) maps to one score per image."""
    flat = maps.flatten(start_dim=1)
    if cfg.image_score == "max":
        return flat.max(dim=1).values
    if cfg.image_score == "topk_mean":
        k = max(1, int(round(flat.shape[1] * cfg.topk_fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unknown image_score '{cfg.image_score}'")


@torch.no_grad()
def compute_map_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: ScoringConfig) -> MapOutputs:
    model.eval()
    maps, masks, labels, scores, paths = [], [], [], [], []
    for batch in loader:
        raw = model.anomaly_map(batch["image"].to(device))
        smoothed = smooth_maps(raw, cfg.smoothing_sigma)
        scores.append(image_scores(smoothed, cfg).cpu().numpy())
        maps.append(smoothed[:, 0].cpu().numpy().astype(np.float32))
        masks.append(batch["mask"][:, 0].numpy() > 0.5)
        labels.append(np.asarray(batch["label"]))
        paths.extend(batch["path"])
    return MapOutputs(
        maps=np.concatenate(maps),
        masks=np.concatenate(masks),
        labels=np.concatenate(labels),
        scores=np.concatenate(scores),
        paths=paths,
    )
