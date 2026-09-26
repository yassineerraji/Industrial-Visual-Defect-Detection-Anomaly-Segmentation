"""Framework-independent inference: ``Predictor.from_run(run_dir).predict(image)``.

Used by the Streamlit demo, benchmarks and (later) deployment endpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from defect_detection.data.preprocessing import build_eval_transform, load_image
from defect_detection.evaluation.scoring import ScoringConfig, image_scores, smooth_maps
from defect_detection.evaluation.thresholds import Thresholds
from defect_detection.runs import load_model, load_run_config
from defect_detection.utils.config import parse_data_config
from defect_detection.utils.device import get_device


@dataclass
class PredictionResult:
    is_anomalous: bool
    anomaly_score: float  # unnormalised; compare with ``threshold``, not a probability
    threshold: float
    anomaly_map: np.ndarray  # (H, W) float32 at the input image's resolution
    pixel_threshold: float
    latency_ms: float  # preprocessing + model + post-processing, excluding file I/O

    def to_dict(self) -> dict:
        return {
            "is_anomalous": self.is_anomalous,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "anomaly_map": self.anomaly_map,
            "pixel_threshold": self.pixel_threshold,
            "latency_ms": self.latency_ms,
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class Predictor:
    def __init__(
        self, model: torch.nn.Module, cfg: dict, thresholds: Thresholds, device: torch.device, name: str = ""
    ) -> None:
        self.model = model.eval()
        self.cfg = cfg
        self.thresholds = thresholds
        self.device = device
        self.name = name
        data = parse_data_config(cfg["data"])
        self.transform = build_eval_transform(data.preprocessing)
        self.scoring = ScoringConfig(**cfg.get("scoring", {}))

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str = "auto") -> "Predictor":
        run_dir = Path(run_dir)
        if not (run_dir / "model.pt").is_file():
            raise ValueError(
                f"{run_dir} has no deployable model.pt (cross-validation runs need "
                "cross_validation.final_model: true)."
            )
        dev = get_device(device)
        cfg, thresholds = load_run_config(run_dir)
        return cls(load_model(run_dir, cfg["model"], dev), cfg, thresholds, dev, name=run_dir.name)

    @torch.no_grad()
    def predict(self, image: Image.Image | np.ndarray | str | Path) -> PredictionResult:
        if isinstance(image, (str, Path)):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        width, height = image.size

        start = time.perf_counter()
        tensor, _ = self.transform(image, np.zeros((height, width), dtype=np.uint8))
        raw = self.model.anomaly_map(tensor.unsqueeze(0).to(self.device))
        smoothed = smooth_maps(raw, self.scoring.smoothing_sigma)
        score = float(image_scores(smoothed, self.scoring)[0])
        full_map = F.interpolate(smoothed, size=(height, width), mode="bilinear", align_corners=False)
        anomaly_map = full_map[0, 0].cpu().numpy().astype(np.float32)
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictionResult(
            is_anomalous=score > self.thresholds.image,
            anomaly_score=score,
            threshold=self.thresholds.image,
            anomaly_map=anomaly_map,
            pixel_threshold=self.thresholds.pixel,
            latency_ms=latency_ms,
        )
