"""Run directory layout: everything needed to reload and evaluate a trained model.

artifacts/runs/<run_id>/
    config.yaml        fully resolved experiment config (incl. data section)
    model.pt           model state_dict
    thresholds.json    thresholds selected on validation data
    history.json       per-epoch training history
    run_info.json      experiment record (protocol fields, timings)
    metrics_<split>.json, figures/  written by evaluation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from defect_detection.evaluation.thresholds import Thresholds
from defect_detection.models import build_model

RUNS_DIR = Path("artifacts/runs")


def new_run_dir(name: str, base: Path = RUNS_DIR) -> tuple[str, Path]:
    run_id = f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def resolve_run_dir(run: str | Path, base: Path = RUNS_DIR) -> Path:
    """Accept a run id (looked up under ``base``) or a path to a run directory."""
    path = Path(run)
    if (path / "config.yaml").is_file():
        return path
    if (base / run / "config.yaml").is_file():
        return base / run
    raise FileNotFoundError(f"No run found at '{run}' or '{base / run}'")


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_run_config(run_dir: Path, cfg: dict, thresholds: Thresholds) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(run_dir / "thresholds.json", thresholds.to_dict())


def save_run(run_dir: Path, cfg: dict, model: nn.Module, thresholds: Thresholds) -> None:
    save_run_config(run_dir, cfg, thresholds)
    torch.save(model.state_dict(), run_dir / "model.pt")


def load_run_config(run_dir: Path) -> tuple[dict, Thresholds]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return cfg, Thresholds(**load_json(run_dir / "thresholds.json"))


def load_model(model_dir: Path, model_cfg: dict, device: torch.device) -> nn.Module:
    """Rebuild a model from config and load ``model_dir/model.pt``."""
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval()


def load_run(run_dir: Path, device: torch.device) -> tuple[dict, nn.Module, Thresholds]:
    cfg, thresholds = load_run_config(run_dir)
    return cfg, load_model(run_dir, cfg["model"], device), thresholds
