"""Evaluate a saved run on a labelled split (default: public test set).

Example:
    python scripts/evaluate.py --run autoencoder_sheet_metal_20260926-160000
"""

from __future__ import annotations

import argparse

from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.runs import resolve_run_dir
from defect_detection.tracking import log_run
from defect_detection.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run id under artifacts/runs, or a run directory path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-figures", type=int, default=8)
    parser.add_argument("--no-mlflow", action="store_true", help="skip logging results to MLflow")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    evaluate_run(run_dir, args.split, args.data_root, args.device, args.num_figures)
    if not args.no_mlflow:
        log_run(run_dir)


if __name__ == "__main__":
    main()
