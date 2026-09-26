"""Generic epoch loop for gradient-trained models (autoencoder, U-Net)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from defect_detection.training.losses import LossFn
from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    loss: str = "mse"
    grad_accum_steps: int = 1
    num_workers: int = 0
    device: str = "auto"
    select_best: bool = True  # keep lowest-val-loss weights; False keeps the final epoch


def build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: LossFn, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        n = batch["image"].shape[0]
        total += loss_fn(model, batch, device).item() * n
        count += n
    return total / max(count, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: LossFn,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train for ``cfg.epochs``; if ``cfg.select_best``, restore the lowest-validation-loss weights.

    Returns the model and the per-epoch history.
    """
    model.to(device)
    optimizer = build_optimizer(model.parameters(), cfg)
    best_loss, best_state, history = float("inf"), None, []

    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            loss = loss_fn(model, batch, device)
            (loss / cfg.grad_accum_steps).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            n = batch["image"].shape[0]
            running += loss.item() * n
            seen += n

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": running / seen,
            "val_loss": val_loss,
            "epoch_time_s": time.perf_counter() - start,
        }
        history.append(record)
        if cfg.select_best and val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            record["best"] = True
        logger.info(
            "epoch %d/%d train=%.5f val=%.5f (%.1fs)%s",
            epoch, cfg.epochs, record["train_loss"], val_loss, record["epoch_time_s"], " *" if record.get("best") else "",
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
