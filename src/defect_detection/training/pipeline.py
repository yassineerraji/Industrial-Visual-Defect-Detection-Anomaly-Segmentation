"""End-to-end training for normal-only (Regime A) gradient models.

Train on the ``train`` split, select the checkpoint and thresholds on the
``validation`` split. The test split is never touched here.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from defect_detection.data.augmentations import build_train_transform
from defect_detection.data.dataset import NORMAL, MVTecAD2Dataset
from defect_detection.data.loaders import build_loader
from defect_detection.data.preprocessing import build_eval_transform
from defect_detection.evaluation.scoring import ScoringConfig, compute_map_outputs
from defect_detection.evaluation.thresholds import ThresholdConfig, select_thresholds
from defect_detection.models import build_model
from defect_detection.models.patchcore import PatchCore
from defect_detection.runs import RUNS_DIR, new_run_dir, save_json, save_run
from defect_detection.training.losses import get_loss
from defect_detection.training.trainer import TrainConfig, fit
from defect_detection.utils.config import DataConfig, parse_data_config
from defect_detection.utils.device import get_device
from defect_detection.utils.logging import get_logger
from defect_detection.utils.reproducibility import seed_everything

logger = get_logger(__name__)


def _require_normal_only(dataset: MVTecAD2Dataset, split: str) -> None:
    n_anomalous = sum(s.label != NORMAL for s in dataset.samples)
    if n_anomalous:
        raise ValueError(f"'{split}' has {n_anomalous} anomalous samples; Regime A requires normal-only data")


def train_normal_only(cfg: dict[str, Any], data_root: str | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    data = parse_data_config(cfg["data"], data_root=data_root)
    cfg["data"]["dataset"]["root"] = str(data.dataset.root)  # record the root actually used
    train_cfg = TrainConfig(**cfg["training"])
    scoring_cfg = ScoringConfig(**cfg.get("scoring", {}))
    threshold_cfg = ThresholdConfig(**cfg.get("threshold", {}))

    seed_everything(data.seed)
    device = get_device(train_cfg.device)

    train_ds = MVTecAD2Dataset.from_config(
        data.dataset, "train", build_train_transform(data.preprocessing, data.augmentation)
    )
    val_ds = MVTecAD2Dataset.from_config(data.dataset, "validation", build_eval_transform(data.preprocessing))
    _require_normal_only(train_ds, "train")
    _require_normal_only(val_ds, "validation")
    logger.info("train=%d validation=%d images at %s", len(train_ds), len(val_ds), data.preprocessing.output_size)

    train_loader = build_loader(train_ds, train_cfg.batch_size, True, data.seed, train_cfg.num_workers)
    val_loader = build_loader(val_ds, train_cfg.batch_size, False, data.seed, train_cfg.num_workers)

    model = build_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model=%s params=%d", cfg["model"]["name"], n_params)

    start = time.perf_counter()
    if isinstance(model, PatchCore):
        fit_stats, history = model.fit(train_loader, device), []
    else:
        model, history = fit(model, train_loader, val_loader, get_loss(train_cfg.loss), train_cfg, device)
        best = min(history, key=lambda r: r["val_loss"])
        fit_stats = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"]}
    training_time = time.perf_counter() - start

    val_out = compute_map_outputs(model, val_loader, device, scoring_cfg)
    thresholds = select_thresholds(val_out.scores, val_out.maps, threshold_cfg)
    logger.info("thresholds: image=%.6f pixel=%.6f (%s)", thresholds.image, thresholds.pixel, thresholds.source)

    run_id, run_dir = new_run_dir(cfg["experiment_name"], runs_dir)
    save_run(run_dir, cfg, model, thresholds)
    save_json(run_dir / "history.json", history)
    record = experiment_record(
        run_id, cfg, data, train_cfg, device, model, training_time,
        supervision="normal_only", gradient_trained=not isinstance(model, PatchCore),
    )
    record["fit"] = fit_stats
    record["metrics"] = {
        "val_normal_score_mean": float(np.mean(val_out.scores)),
        "val_normal_score_std": float(np.std(val_out.scores)),
    }
    save_json(run_dir / "run_info.json", record)
    logger.info("Run saved to %s", run_dir)
    return run_dir


def experiment_record(
    run_id: str,
    cfg: dict[str, Any],
    data: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    model: torch.nn.Module,
    training_time: float,
    supervision: str,
    gradient_trained: bool,
) -> dict[str, Any]:
    """Protocol fields recorded for every experiment."""
    return {
        "experiment_id": run_id,
        "model": cfg["model"]["name"],
        "supervision": supervision,
        "dataset": "MVTec AD 2",
        "category": data.dataset.category,
        "image_size": list(data.preprocessing.output_size),
        "batch_size": train_cfg.batch_size,
        "epochs": train_cfg.epochs if gradient_trained else None,
        "learning_rate": train_cfg.learning_rate if gradient_trained else None,
        "optimizer": train_cfg.optimizer if gradient_trained else None,
        "loss": train_cfg.loss if gradient_trained else None,
        "augmentation": cfg["data"].get("augmentation", {}),
        "seed": data.seed,
        "device": str(device),
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "training_time_s": training_time,
    }
