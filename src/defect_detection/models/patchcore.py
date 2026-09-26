"""PatchCore (Roth et al., CVPR 2022): pretrained-feature anomaly detection (Regime A).

Simplifications vs. the paper, kept deliberately small:
- features are locally averaged (3x3) and concatenated, without per-layer
  adaptive pooling to a fixed dimension;
- image score is the maximum patch distance (no nearest-neighbour reweighting).
Inputs must use the backbone's ImageNet normalisation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor

from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def greedy_coreset(
    features: torch.Tensor, num_samples: int, projection_dim: int | None = 128, seed: int = 0
) -> torch.Tensor:
    """Greedy k-center selection: iteratively add the point farthest from the current set.

    Distances are computed in a random Gaussian projection (Johnson-Lindenstrauss)
    of the features when ``projection_dim`` is smaller than the feature dimension.
    Returns indices into ``features``.
    """
    n = features.shape[0]
    if num_samples >= n:
        return torch.arange(n, device=features.device)
    generator = torch.Generator().manual_seed(seed)
    if projection_dim is not None and projection_dim < features.shape[1]:
        proj = torch.randn(features.shape[1], projection_dim, generator=generator) / projection_dim**0.5
        reduced = features @ proj.to(features.device, features.dtype)
    else:
        reduced = features

    start = int(torch.randint(n, (1,), generator=generator))
    selected = [start]
    min_dist = torch.linalg.vector_norm(reduced - reduced[start], dim=1)
    for _ in range(num_samples - 1):
        idx = int(torch.argmax(min_dist))
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.linalg.vector_norm(reduced - reduced[idx], dim=1))
    return torch.tensor(selected, device=features.device)


@torch.no_grad()
def nearest_distances(queries: torch.Tensor, bank: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Euclidean distance from each query row to its nearest bank row."""
    out = [torch.cdist(q, bank).min(dim=1).values for q in queries.split(chunk_size)]
    return torch.cat(out)


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] | list[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        coreset_ratio: float = 0.01,
        projection_dim: int | None = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        self.layers = list(layers)
        self.coreset_ratio = coreset_ratio
        self.projection_dim = projection_dim
        self.seed = seed
        net = get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.feature_extractor = create_feature_extractor(net, return_nodes={l: l for l in self.layers})
        self.feature_extractor.eval().requires_grad_(False)
        self.register_buffer("memory_bank", torch.empty(0))

    def train(self, mode: bool = True) -> "PatchCore":
        super().train(mode)
        self.feature_extractor.eval()  # frozen backbone: BatchNorm always in inference mode
        return self

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Locally aggregated multi-layer patch features, shape (B, C, h, w) at the first layer's resolution."""
        feats = self.feature_extractor(x)
        maps = [F.avg_pool2d(feats[l], kernel_size=3, stride=1, padding=1) for l in self.layers]
        size = maps[0].shape[-2:]
        maps = [m if m.shape[-2:] == size else F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps]
        return torch.cat(maps, dim=1)

    @torch.no_grad()
    def fit(self, loader: DataLoader, device: torch.device) -> dict[str, Any]:
        """Build the memory bank from normal images; returns bank statistics."""
        self.to(device).eval()
        patches = []
        for batch in loader:
            emb = self.embed(batch["image"].to(device))
            patches.append(emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1]))
        features = torch.cat(patches)
        num_samples = max(1, int(round(features.shape[0] * self.coreset_ratio)))
        logger.info("PatchCore: %d patches x %d dims -> coreset of %d", *features.shape, num_samples)
        idx = greedy_coreset(features, num_samples, self.projection_dim, self.seed)
        self.memory_bank = features[idx].contiguous()
        return {"num_patches": features.shape[0], "feature_dim": features.shape[1], "memory_bank_size": num_samples}

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour patch distances upsampled to input size, shape (B, 1, H, W)."""
        if self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore memory bank is empty; call fit() first")
        emb = self.embed(x)
        b, c, h, w = emb.shape
        dist = nearest_distances(emb.permute(0, 2, 3, 1).reshape(-1, c), self.memory_bank)
        return F.interpolate(dist.view(b, 1, h, w), size=x.shape[-2:], mode="bilinear", align_corners=False)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # The memory bank size is only known after fitting: resize before loading.
        if "memory_bank" in state_dict:
            self.memory_bank = torch.empty_like(state_dict["memory_bank"], device=self.memory_bank.device)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
