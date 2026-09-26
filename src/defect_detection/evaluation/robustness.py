"""Robustness to controlled, deterministic image perturbations.

Thresholds stay fixed at their validation-selected values, as in deployment,
so threshold-dependent metrics (F1, Dice) show operational degradation while
AUROC shows ranking degradation. Results are stored separately from the
clean test metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from defect_detection.data.dataset import index_split
from defect_detection.data.preprocessing import build_eval_transform
from defect_detection.evaluation.metrics import image_metrics, pixel_metrics
from defect_detection.evaluation.pipeline import is_cross_validation_run, predict_split
from defect_detection.runs import load_run_config
from defect_detection.utils.config import parse_data_config
from defect_detection.utils.device import get_device
from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = ("mild", "moderate", "strong")


def _perturbation_fn(name: str, value: float, noise_seed: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "brightness":
        return lambda x: F.adjust_brightness(x, value)
    if name == "contrast":
        return lambda x: F.adjust_contrast(x, value)
    if name == "blur":
        kernel = 2 * int(4 * value + 0.5) + 1
        return lambda x: F.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[value, value])
    if name == "gaussian_noise":
        def add_noise(x: torch.Tensor) -> torch.Tensor:
            # Fixed seed per image: identical, reproducible noise regardless of evaluation order.
            gen = torch.Generator().manual_seed(noise_seed)
            return (x + value * torch.randn(x.shape, generator=gen)).clamp(0.0, 1.0)
        return add_noise
    raise ValueError(f"Unknown perturbation '{name}'")


def make_perturbation(name: str, value: float, noise_seed: int = 0) -> v2.Transform:
    """Image-only transform (masks pass through untouched)."""
    fn = _perturbation_fn(name, value, noise_seed)
    return v2.Lambda(lambda img: tv_tensors.wrap(fn(img.as_subclass(torch.Tensor)), like=img), tv_tensors.Image)


def _summarise(out, thresholds) -> dict[str, float]:
    img = image_metrics(out.labels, out.scores, thresholds.image)
    pix = pixel_metrics(out.masks, out.maps, thresholds.pixel)
    return {
        "image_auroc": img["image_auroc"],
        "image_ap": img["image_ap"],
        "f1": img["f1"],
        "false_positive_rate": img["false_positive_rate"],
        "pixel_auroc": pix["pixel_auroc"],
        "dice": pix["dice"],
    }


def evaluate_robustness(
    run_dir: Path, spec: dict[str, Any], data_root: str | None = None, device_name: str = "auto"
) -> dict[str, Any]:
    device = get_device(device_name)
    cfg, thresholds = load_run_config(run_dir)
    data = parse_data_config(cfg["data"], data_root=data_root)
    split = spec.get("split", "test")
    samples = index_split(data.dataset, split)
    noise_seed = int(spec.get("noise_seed", 0))

    conditions = [("clean", "none", None)]
    for name, levels in spec["perturbations"].items():
        for severity in SEVERITY_ORDER:
            if severity in levels:
                conditions.append((name, severity, float(levels[severity])))

    rows = []
    for name, severity, value in conditions:
        perturbation = None if value is None else make_perturbation(name, value, noise_seed)
        transform = build_eval_transform(data.preprocessing, perturbation)
        out, _ = predict_split(run_dir, cfg, data, samples, transform, device)
        row = {"perturbation": name, "severity": severity, "value": value, **_summarise(out, thresholds)}
        rows.append(row)
        logger.info("%-15s %-9s image_auroc=%.4f pixel_auroc=%.4f f1=%.4f",
                    name, severity, row["image_auroc"], row["pixel_auroc"], row["f1"])

    return {
        "run": run_dir.name,
        "split": split,
        "protocol": {
            "evaluation": "pooled out-of-fold predictions (grouped CV)" if is_cross_validation_run(run_dir)
            else "single model on full split",
            "thresholds": thresholds.to_dict(),
            "perturbation_space": "float [0, 1] image at model resolution, before normalisation",
            "spec": spec,
        },
        "results": rows,
    }
