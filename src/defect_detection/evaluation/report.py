"""Collect results from run directories into a machine-readable file and a Markdown table.

Only values present in run artifacts are reported; anything missing is shown as "TBD".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from defect_detection.runs import load_json

SUPERVISION_LABELS = {"normal_only": "Normal only", "pixel_labels": "Pixel labels", "image_labels": "Image labels"}
MODEL_LABELS = {"autoencoder": "Autoencoder", "patchcore": "PatchCore", "unet": "U-Net"}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _get(path: Path, *keys: str) -> Any:
    if not path.is_file():
        return None
    value: Any = load_json(path)
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def collect_row(run_dir: Path, split: str = "test") -> dict[str, Any]:
    info = load_json(run_dir / "run_info.json")
    metrics = run_dir / f"metrics_{split}.json"
    bench = run_dir / "benchmark.json"
    return {
        "run": run_dir.name,
        "model": info["model"],
        "supervision": info["supervision"],
        "protocol": _get(metrics, "protocol", "evaluation"),
        "category": info["category"],
        "image_size": info["image_size"],
        "image_auroc": _get(metrics, "image", "image_auroc"),
        "image_ap": _get(metrics, "image", "image_ap"),
        "f1": _get(metrics, "image", "f1"),
        "pixel_auroc": _get(metrics, "pixel", "pixel_auroc"),
        "dice": _get(metrics, "pixel", "dice"),
        "iou": _get(metrics, "pixel", "iou"),
        "latency_ms": _get(bench, "latency_ms_mean"),
        "checkpoint_mb": _get(bench, "checkpoint_mb"),
        "peak_rss_mb": _get(bench, "peak_process_rss_mb"),
        "benchmark_device": _get(bench, "device"),
        "training_time_s": info.get("training_time_s"),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "TBD"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Supervision | Evaluation | Image AUROC | Image AP | F1 | Pixel AUROC | Dice | IoU "
        "| Latency (ms) | Checkpoint (MB) | Peak RSS (MB) |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {MODEL_LABELS.get(r['model'], r['model'])} | {SUPERVISION_LABELS.get(r['supervision'], r['supervision'])} "
        f"| {_fmt(r['protocol'])} | {_fmt(r['image_auroc'])} | {_fmt(r['image_ap'])} | {_fmt(r['f1'])} "
        f"| {_fmt(r['pixel_auroc'])} | {_fmt(r['dice'])} | {_fmt(r['iou'])} | {_fmt(r['latency_ms'], 1)} "
        f"| {_fmt(r['checkpoint_mb'], 1)} | {_fmt(r['peak_rss_mb'], 0)} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def update_readme(readme: Path, table: str) -> bool:
    """Replace the text between the RESULTS markers; returns False if the markers are absent."""
    text = readme.read_text()
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(f"{README_START}\n{table}{README_END}", text))
    return True
