"""Operational benchmark: single-image latency, throughput, model size, memory.

Latency is measured through :class:`Predictor` (preprocessing + model +
post-processing, excluding disk I/O) at batch size 1 on already-loaded images.
"""

from __future__ import annotations

import platform
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from defect_detection.data.dataset import index_split
from defect_detection.data.preprocessing import load_image
from defect_detection.inference.predictor import Predictor
from defect_detection.utils.config import parse_data_config


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KiB on Linux


def _accelerator_memory_mb(device: torch.device) -> float | None:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return None


def benchmark_run(
    run_dir: Path, split: str = "validation", num_images: int = 10, repeats: int = 3, warmup: int = 3,
    device: str = "auto", data_root: str | None = None,
) -> dict[str, Any]:
    predictor = Predictor.from_run(run_dir, device=device)
    data = parse_data_config(predictor.cfg["data"], data_root=data_root)
    samples = index_split(data.dataset, split)[:num_images]
    images = [load_image(s.image_path) for s in samples]

    for image in images[:warmup]:
        predictor.predict(image)
    latencies = [predictor.predict(image).latency_ms for _ in range(repeats) for image in images]

    lat = np.asarray(latencies)
    checkpoint = run_dir / "model.pt"
    return {
        "run": run_dir.name,
        "model": predictor.cfg["model"]["name"],
        "hardware": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "device": str(predictor.device),
        "torch_version": torch.__version__,
        "input_resolution": list(images[0].size[::-1]),  # (H, W) of the raw image
        "model_resolution": list(data.preprocessing.output_size),
        "batch_size": 1,
        "num_measurements": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_img_per_s": float(1000.0 / lat.mean()),
        "num_parameters": sum(p.numel() for p in predictor.model.parameters()),
        "checkpoint_mb": checkpoint.stat().st_size / 1e6,
        "peak_process_rss_mb": _peak_rss_mb(),
        "accelerator_memory_mb": _accelerator_memory_mb(predictor.device),
        "notes": "latency includes preprocessing, model and map upsampling to input size; excludes image decoding",
    }
