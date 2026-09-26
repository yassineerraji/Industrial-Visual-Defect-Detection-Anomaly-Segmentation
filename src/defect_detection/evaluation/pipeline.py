"""Evaluate a saved run on a labelled split using the run's stored config and thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from defect_detection.data.dataset import MVTecAD2Dataset, Sample, index_split
from defect_detection.data.folds import fold_key
from defect_detection.data.loaders import build_loader
from defect_detection.data.preprocessing import PairTransform, build_eval_transform, denormalise
from defect_detection.evaluation.metrics import image_metrics, per_condition_image_auroc, pixel_metrics
from defect_detection.evaluation.scoring import MapOutputs, ScoringConfig, compute_map_outputs
from defect_detection.evaluation.visualisation import plot_predictions, select_examples
from defect_detection.runs import load_json, load_model, load_run_config, save_json
from defect_detection.utils.config import DataConfig, parse_data_config
from defect_detection.utils.device import get_device
from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _concat_outputs(parts: list[MapOutputs]) -> MapOutputs:
    return MapOutputs(
        maps=np.concatenate([p.maps for p in parts]),
        masks=np.concatenate([p.masks for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        scores=np.concatenate([p.scores for p in parts]),
        paths=[path for p in parts for path in p.paths],
    )


def _predict_cross_validation(
    run_dir: Path, cfg: dict, data: DataConfig, samples: list[Sample], transform: PairTransform, device: torch.device
) -> tuple[MapOutputs, list[Sample]]:
    """Out-of-fold predictions: each sample is scored by the model that held out its fold."""
    fold_of = load_json(run_dir / "folds.json")
    keys = [fold_key(data.dataset, s) for s in samples]
    missing = [k for k in keys if k not in fold_of]
    if missing:
        raise ValueError(f"{len(missing)} samples are not in this run's folds (e.g. {missing[0]})")
    scoring_cfg = ScoringConfig(**cfg.get("scoring", {}))
    parts, ordered = [], []
    for k in sorted(set(fold_of.values())):
        fold_samples = [s for s, key in zip(samples, keys) if fold_of[key] == k]
        if not fold_samples:  # a subset of the split may not touch every fold
            continue
        model = load_model(run_dir / f"fold_{k}", cfg["model"], device)
        loader = build_loader(MVTecAD2Dataset(fold_samples, transform), cfg["training"]["batch_size"], False, data.seed)
        parts.append(compute_map_outputs(model, loader, device, scoring_cfg))
        ordered += fold_samples
    return _concat_outputs(parts), ordered


def is_cross_validation_run(run_dir: Path) -> bool:
    return (run_dir / "folds.json").is_file()


def predict_split(
    run_dir: Path, cfg: dict, data: DataConfig, samples: list[Sample], transform: PairTransform, device: torch.device
) -> tuple[MapOutputs, list[Sample]]:
    """Score ``samples`` with the run's model (or out-of-fold models); returns outputs and their sample order.

    For cross-validation runs, the final all-data model (if any) is never used for evaluation.
    """
    if is_cross_validation_run(run_dir):
        return _predict_cross_validation(run_dir, cfg, data, samples, transform, device)
    model = load_model(run_dir, cfg["model"], device)
    loader = build_loader(MVTecAD2Dataset(samples, transform), cfg["training"]["batch_size"], False, data.seed)
    return compute_map_outputs(model, loader, device, ScoringConfig(**cfg.get("scoring", {}))), samples


def evaluate_run(
    run_dir: Path,
    split: str = "test",
    data_root: str | None = None,
    device_name: str = "auto",
    num_figures: int = 8,
) -> dict[str, Any]:
    device = get_device(device_name)
    cfg, thresholds = load_run_config(run_dir)
    data = parse_data_config(cfg["data"], data_root=data_root)
    if not data.dataset.root.is_dir():
        raise FileNotFoundError(
            f"Dataset root {data.dataset.root} (recorded when the run was trained) does not exist here; "
            "pass --data-root to point at the local copy."
        )
    scoring_cfg = ScoringConfig(**cfg.get("scoring", {}))
    transform = build_eval_transform(data.preprocessing)
    out, samples = predict_split(run_dir, cfg, data, index_split(data.dataset, split), transform, device)
    is_cv = is_cross_validation_run(run_dir)
    dataset = MVTecAD2Dataset(samples, transform)

    results = {
        "run": run_dir.name,
        "split": split,
        "protocol": {
            "evaluation": "pooled out-of-fold predictions (grouped CV)" if is_cv else "single model on full split",
            "num_normal": int((out.labels == 0).sum()),
            "num_anomalous": int((out.labels == 1).sum()),
            "resolution": list(data.preprocessing.output_size),
            "scoring": vars(scoring_cfg),
            "thresholds": thresholds.to_dict(),
            "pixel_metrics_resolution": "model resolution (masks resized with nearest neighbour)",
        },
        "image": image_metrics(out.labels, out.scores, thresholds.image),
        "pixel": pixel_metrics(out.masks, out.maps, thresholds.pixel),
        "per_condition": per_condition_image_auroc(out.labels, out.scores, out.paths),
    }
    save_json(run_dir / f"metrics_{split}.json", results)
    logger.info("image: %s", {k: round(v, 4) for k, v in results["image"].items()})
    logger.info("pixel: %s", {k: round(v, 4) for k, v in results["pixel"].items()})

    if num_figures > 0:
        picks = select_examples(out.labels, num_figures)
        images = [denormalise(dataset[i]["image"], data.preprocessing).permute(1, 2, 0).numpy() for i in picks]
        fig = plot_predictions(
            images,
            out.masks[picks],
            out.maps[picks],
            out.scores[picks],
            [out.paths[i] for i in picks],
            thresholds.image,
            thresholds.pixel,
            # Display only: shared scale across rows; some models (PatchCore) have a large non-zero baseline.
            vmin=float(np.quantile(out.maps, 0.01)),
            vmax=float(np.quantile(out.maps, 0.999)),
        )
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f"predictions_{split}.png", dpi=110)
        logger.info("Figure saved to %s", fig_dir / f"predictions_{split}.png")
    return results
