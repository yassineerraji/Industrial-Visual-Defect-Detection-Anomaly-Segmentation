"""Training losses. Each takes ``(model, batch, device)`` and returns a scalar tensor."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

LossFn = Callable[[nn.Module, dict[str, Any], torch.device], torch.Tensor]


def reconstruction_mse(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    x = batch["image"].to(device)
    return F.mse_loss(model(x), x)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed over the whole batch so defect-free images do not dominate."""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (probs.sum() + target.sum() + eps)


def bce_dice(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """BCE (per-pixel calibration) + Dice (robust to the heavy class imbalance of small defects)."""
    logits = model(batch["image"].to(device))
    target = batch["mask"].to(device)
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


LOSSES: dict[str, LossFn] = {"mse": reconstruction_mse, "bce_dice": bce_dice}


def get_loss(name: str) -> LossFn:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name]
