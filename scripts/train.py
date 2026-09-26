"""Train a model from an experiment config and save a self-contained run directory.

Example:
    python scripts/train.py --config configs/autoencoder.yaml
"""

from __future__ import annotations

import argparse

from defect_detection.tracking import log_run
from defect_detection.training.pipeline import train_normal_only
from defect_detection.training.supervised import train_supervised_cv
from defect_detection.utils.config import resolve_experiment_config
from defect_detection.utils.logging import configure_logging

NORMAL_ONLY_MODELS = {"autoencoder", "patchcore"}
SUPERVISED_MODELS = {"unet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data.dataset.root")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs (logged in the run config)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip logging the run to MLflow")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = resolve_experiment_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    name = cfg["model"]["name"]
    if name in NORMAL_ONLY_MODELS:
        run_dir = train_normal_only(cfg, data_root=args.data_root)
    elif name in SUPERVISED_MODELS:
        run_dir = train_supervised_cv(cfg, data_root=args.data_root)
    else:
        raise SystemExit(f"No training pipeline for model '{name}'")
    if not args.no_mlflow:
        log_run(run_dir)


if __name__ == "__main__":
    main()
