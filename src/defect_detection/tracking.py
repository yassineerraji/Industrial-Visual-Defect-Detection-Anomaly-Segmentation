"""Mirror a run directory into MLflow.

Run directories remain the source of truth; MLflow is a view over them. The
tracking backend comes from ``MLFLOW_TRACKING_URI`` (local SQLite by default,
an Azure ML workspace URI later), so the same code serves both.
Re-logging a run resumes the same MLflow run (id stored in ``mlflow.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from defect_detection.runs import load_json, load_run_config, save_json
from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)

_PARAM_SECTIONS = ("model", "training", "scoring", "threshold", "cross_validation")
_INFO_PARAMS = ("supervision", "category", "image_size", "seed", "device", "hardware", "torch_version", "num_parameters")


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def _numeric(d: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _log_history(history: list[dict] | dict[str, list[dict]]) -> None:
    runs = history.items() if isinstance(history, dict) else [("", history)]
    for prefix, records in runs:
        tag = f"{prefix}/" if prefix else ""
        for rec in records:
            mlflow.log_metrics(
                {f"{tag}train_loss": rec["train_loss"], f"{tag}val_loss": rec["val_loss"]}, step=rec["epoch"]
            )


def _log_evaluation(metrics_file: Path) -> None:
    results = load_json(metrics_file)
    split = results["split"]
    mlflow.log_metrics({f"{split}/{k}": v for k, v in _numeric(results["image"]).items()})
    mlflow.log_metrics({f"{split}/{k}": v for k, v in _numeric(results["pixel"]).items()})
    for cond, vals in results.get("per_condition", {}).items():
        mlflow.log_metric(f"{split}/condition/{cond}/image_auroc", vals["image_auroc"])


def log_run(run_dir: Path, experiment: str | None = None) -> str:
    """Log (or update) ``run_dir`` in MLflow; returns the MLflow run id."""
    info = load_json(run_dir / "run_info.json")
    cfg, _ = load_run_config(run_dir)
    cfg_params: dict[str, Any] = {}
    for section in _PARAM_SECTIONS:
        if section in cfg:
            cfg_params.update(flatten(cfg[section], f"{section}."))
    cfg_params.update(flatten(cfg["data"].get("preprocessing", {}), "preprocessing."))
    cfg_params.update(flatten(cfg["data"].get("augmentation", {}), "augmentation."))

    mlflow.set_experiment(experiment or f"mvtec_ad2_{info['category']}")
    state_file = run_dir / "mlflow.json"
    existing = load_json(state_file)["run_id"] if state_file.is_file() else None

    with mlflow.start_run(run_id=existing, run_name=None if existing else info["experiment_id"]) as run:
        if existing is None:
            mlflow.log_params({k: str(v) for k, v in cfg_params.items()})
            mlflow.log_params({k: str(v) for k, v in info.items() if k in _INFO_PARAMS})
            mlflow.set_tags({"model": info["model"], "supervision": info["supervision"], "run_dir": str(run_dir)})
            mlflow.log_metric("training_time_s", info["training_time_s"])
            _log_history(load_json(run_dir / "history.json"))
            thresholds = load_json(run_dir / "thresholds.json")
            mlflow.log_metrics({"threshold_image": thresholds["image"], "threshold_pixel": thresholds["pixel"]})
            for name in ("config.yaml", "run_info.json", "thresholds.json"):
                mlflow.log_artifact(str(run_dir / name))
        for metrics_file in sorted(run_dir.glob("metrics_*.json")):
            _log_evaluation(metrics_file)
            mlflow.log_artifact(str(metrics_file))
        for robustness_file in sorted(run_dir.glob("robustness_*.json")):
            for row in load_json(robustness_file)["results"]:
                prefix = f"robustness/{row['perturbation']}/{row['severity']}"
                mlflow.log_metrics({f"{prefix}/{k}": v for k, v in _numeric(row).items() if k != "value"})
            mlflow.log_artifact(str(robustness_file))
        if (run_dir / "benchmark.json").is_file():
            bench = load_json(run_dir / "benchmark.json")
            mlflow.log_metrics({f"benchmark/{k}": v for k, v in _numeric(bench).items()})
            mlflow.log_artifact(str(run_dir / "benchmark.json"))
        if (run_dir / "figures").is_dir():
            mlflow.log_artifacts(str(run_dir / "figures"), artifact_path="figures")
        run_id = run.info.run_id

    save_json(state_file, {"run_id": run_id, "tracking_uri": mlflow.get_tracking_uri()})
    logger.info("Logged %s to MLflow run %s", run_dir.name, run_id)
    return run_id
