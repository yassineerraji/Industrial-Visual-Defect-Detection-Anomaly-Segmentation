"""Export a trained run as a self-contained release bundle for deployment.

releases/<name>/
    model.pt, config.yaml, thresholds.json    loadable by Predictor.from_run
    folds.json, fold_<k>/model.pt             cross-validation runs only (out-of-fold scoring)
    release.json                              provenance, metrics summary, display and drift references
    run_info.json, metrics_test.json, benchmark_<device>.json, robustness_test.json (when present)

Reference statistics use only non-test data: heatmap display range from normal
validation images, input-brightness reference from the normal train images.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from defect_detection.data.dataset import index_split
from defect_detection.data.preprocessing import load_image
from defect_detection.inference.predictor import Predictor
from defect_detection.runs import load_json, save_json
from defect_detection.utils.config import parse_data_config

REQUIRED = ("model.pt", "config.yaml", "thresholds.json", "run_info.json")
OPTIONAL = ("metrics_test.json", "benchmark_cpu.json", "benchmark_mps.json", "robustness_test.json")
LICENCE = (
    "Trained on MVTec AD 2 (MVTec Software GmbH), licensed CC BY-NC-SA 4.0. "
    "Model weights are distributed under the same terms, for non-commercial use."
)


def image_brightness(image: np.ndarray) -> float:
    """Mean grey level in [0, 1] of an (H, W, 3) uint8 image."""
    return float(image.astype(np.float32).mean() / 255.0)


def display_range(normal_maps: list[np.ndarray], pixel_threshold: float) -> dict[str, float]:
    """Colour scale where 99% of normal pixels are transparent and the pixel threshold is mid-scale."""
    vmin = float(np.quantile(np.concatenate([m.ravel() for m in normal_maps]), 0.99))
    vmin = min(vmin, pixel_threshold * 0.999)
    return {"vmin": vmin, "vmax": 2 * pixel_threshold - vmin}


def _metric_summary(bundle: Path) -> dict[str, Any]:
    path = bundle / "metrics_test.json"
    if not path.is_file():
        return {}
    metrics = load_json(path)
    return {
        "evaluation": metrics["protocol"].get("evaluation"),
        "image_auroc": metrics["image"]["image_auroc"],
        "pixel_auroc": metrics["pixel"]["pixel_auroc"],
        "dice": metrics["pixel"]["dice"],
        "f1": metrics["image"]["f1"],
    }


def export_release(
    run_dir: Path, out_dir: Path, name: str, data_root: str | None = None, device: str = "auto"
) -> Path:
    missing = [f for f in REQUIRED if not (run_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is not deployable, missing {missing}")
    bundle = out_dir / name
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    for f in REQUIRED + OPTIONAL:
        if (run_dir / f).is_file():
            shutil.copy2(run_dir / f, bundle / f)
    # Cross-validation runs: ship fold models so images from the labelled split can be scored out-of-fold
    # (the final all-data model has seen every labelled image).
    is_cv = (run_dir / "folds.json").is_file()
    if is_cv:
        shutil.copy2(run_dir / "folds.json", bundle / "folds.json")
        for fold_dir in sorted(run_dir.glob("fold_*")):
            (bundle / fold_dir.name).mkdir()
            shutil.copy2(fold_dir / "model.pt", bundle / fold_dir.name / "model.pt")

    predictor = Predictor.from_run(bundle, device=device)
    data = parse_data_config(yaml.safe_load((bundle / "config.yaml").read_text())["data"], data_root=data_root)
    # Maps are upsampled from model resolution, so a strided view keeps all information at 1/16 the memory.
    val_maps = [predictor.predict(s.image_path).anomaly_map[::4, ::4] for s in index_split(data.dataset, "validation")]
    train_brightness = [image_brightness(np.asarray(load_image(s.image_path))) for s in index_split(data.dataset, "train")]

    info = load_json(run_dir / "run_info.json")
    save_json(
        bundle / "release.json",
        {
            "name": name,
            "model": info["model"],
            "supervision": info["supervision"],
            "category": info["category"],
            "source_run": run_dir.name,
            "git_commit": info.get("git_commit"),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "thresholds": load_json(bundle / "thresholds.json"),
            "display": display_range(val_maps, predictor.thresholds.pixel),
            "reference": {
                "source": f"train split normals (n={len(train_brightness)})",
                "brightness_mean": float(np.mean(train_brightness)),
                "brightness_std": float(np.std(train_brightness)),
            },
            "metrics": _metric_summary(bundle),
            "cross_validation": is_cv,
            "licence": LICENCE,
        },
    )
    return bundle
