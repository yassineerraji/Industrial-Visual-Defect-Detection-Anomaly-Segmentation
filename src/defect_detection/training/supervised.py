"""Supervised segmentation (Regime C) with grouped cross-validation.

MVTec AD 2 has pixel labels only in the public test split, so a supervised model
cannot be trained and evaluated on disjoint official splits. Protocol:

- partition the labelled split into ``n_folds`` folds grouped by physical part;
- for fold k, train on the official normal ``train`` images plus all labelled
  images outside fold k, for a fixed number of epochs (final weights, no
  checkpoint selection);
- thresholds are predefined (not tuned), so no held-out data influences the model;
- evaluation pools each fold model's predictions on its own held-out fold.

Optionally (``cross_validation.final_model``) one more model is trained on all
labelled data for deployment; its expected performance is the CV estimate,
since no labelled data remains to evaluate it independently.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from defect_detection.data.augmentations import build_train_transform
from defect_detection.data.dataset import ANOMALOUS, NORMAL, MVTecAD2Dataset, index_split
from defect_detection.data.folds import fold_key, grouped_folds, part_id
from defect_detection.data.loaders import build_loader
from defect_detection.data.preprocessing import build_eval_transform
from defect_detection.evaluation.thresholds import Thresholds
from defect_detection.models import build_model
from defect_detection.runs import RUNS_DIR, new_run_dir, save_json, save_run_config
from defect_detection.training.losses import get_loss
from defect_detection.training.pipeline import experiment_record
from defect_detection.training.trainer import TrainConfig, fit
from defect_detection.utils.config import parse_data_config
from defect_detection.utils.device import get_device
from defect_detection.utils.logging import get_logger
from defect_detection.utils.reproducibility import seed_everything

logger = get_logger(__name__)


def _free_accelerator_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_supervised_cv(cfg: dict[str, Any], data_root: str | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    data = parse_data_config(cfg["data"], data_root=data_root)
    cfg["data"]["dataset"]["root"] = str(data.dataset.root)
    train_cfg = TrainConfig(**cfg["training"])
    cv = cfg["cross_validation"]
    n_folds, cv_split = int(cv["n_folds"]), cv.get("split", "test")
    thr = cfg["threshold"]
    thresholds = Thresholds(
        image=float(thr["image"]),
        pixel=float(thr["pixel"]),
        source=f"predefined (not tuned): image {thr['image']} on max map value, pixel {thr['pixel']} on sigmoid",
    )

    seed_everything(data.seed)
    device = get_device(train_cfg.device)
    train_tf = build_train_transform(data.preprocessing, data.augmentation)
    eval_tf = build_eval_transform(data.preprocessing)

    normals = index_split(data.dataset, "train")
    if any(s.label != NORMAL for s in normals):
        raise ValueError("Official train split is expected to be normal-only")
    val_loader = build_loader(  # monitoring only: never used for selection
        MVTecAD2Dataset.from_config(data.dataset, "validation", eval_tf),
        train_cfg.batch_size, False, data.seed, train_cfg.num_workers,
    )
    labelled = index_split(data.dataset, cv_split)
    folds = grouped_folds(labelled, n_folds, data.seed)

    run_id, run_dir = new_run_dir(cfg["experiment_name"], runs_dir)
    save_run_config(run_dir, cfg, thresholds)
    save_json(run_dir / "folds.json", {fold_key(data.dataset, s): f for s, f in zip(labelled, folds)})

    def train_one(samples: list, seed: int) -> tuple[torch.nn.Module, list, float]:
        seed_everything(seed)
        loader = build_loader(MVTecAD2Dataset(samples, train_tf), train_cfg.batch_size, True, seed, train_cfg.num_workers)
        t0 = time.perf_counter()
        model, history = fit(build_model(cfg["model"]), loader, val_loader, get_loss(train_cfg.loss), train_cfg, device)
        return model, history, time.perf_counter() - t0

    fold_summary, histories = [], {}
    start = time.perf_counter()
    for k in range(n_folds):
        held_out = [s for s, f in zip(labelled, folds) if f == k]
        train_samples = normals + [s for s, f in zip(labelled, folds) if f != k]
        counts = Counter(s.label for s in train_samples)
        summary = {
            "fold": k,
            "train_normal": counts[NORMAL],
            "train_anomalous": counts[ANOMALOUS],
            "heldout_normal": sum(s.label == NORMAL for s in held_out),
            "heldout_anomalous": sum(s.label == ANOMALOUS for s in held_out),
            "heldout_parts": sorted({part_id(s) for s in held_out}),
        }
        logger.info("fold %d/%d: %s", k + 1, n_folds, summary)

        model, history, summary["training_time_s"] = train_one(train_samples, data.seed + k)
        fold_dir = run_dir / f"fold_{k}"
        fold_dir.mkdir()
        torch.save(model.state_dict(), fold_dir / "model.pt")
        histories[f"fold_{k}"] = history
        fold_summary.append(summary)
        del model
        _free_accelerator_memory(device)

    final_model = bool(cv.get("final_model", False))
    if final_model:
        logger.info("final model: all %d labelled + %d normal train images", len(labelled), len(normals))
        model, history, final_time = train_one(normals + labelled, data.seed + n_folds)
        torch.save(model.state_dict(), run_dir / "model.pt")
        histories["final"] = history

    record = experiment_record(
        run_id, cfg, data, train_cfg, device, build_model(cfg["model"]), time.perf_counter() - start,
        supervision="pixel_labels", gradient_trained=True,
    )
    record["protocol"] = {
        "type": "grouped_cross_validation",
        "n_folds": n_folds,
        "labelled_split": cv_split,
        "group": "physical part id (all acquisition conditions of a part share a fold)",
        "extra_training_data": "official normal train split",
        "checkpoint": "final epoch (no selection)",
        "thresholds": thresholds.source,
        "final_model": "trained on all labelled data; performance estimated by CV only" if final_model else None,
    }
    record["folds"] = fold_summary
    if final_model:
        record["final_model_training_time_s"] = final_time
    save_json(run_dir / "history.json", histories)
    save_json(run_dir / "run_info.json", record)
    logger.info("Run saved to %s", run_dir)
    return run_dir
