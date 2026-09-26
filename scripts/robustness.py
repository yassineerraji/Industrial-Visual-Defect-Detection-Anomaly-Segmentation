"""Evaluate a saved run under controlled perturbations at several severities.

Results go to <run>/robustness_<split>.json, separate from clean test metrics.

Example:
    python scripts/robustness.py --run autoencoder_sheet_metal_20260926-160000
"""

from __future__ import annotations

import argparse

from defect_detection.evaluation.robustness import evaluate_robustness
from defect_detection.evaluation.visualisation import plot_robustness
from defect_detection.runs import resolve_run_dir, save_json
from defect_detection.tracking import log_run
from defect_detection.utils.config import load_yaml
from defect_detection.utils.logging import configure_logging, get_logger

logger = get_logger("robustness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    spec = load_yaml(args.config)
    results = evaluate_robustness(run_dir, spec, args.data_root, args.device)
    save_json(run_dir / f"robustness_{results['split']}.json", results)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_robustness(results["results"]).savefig(fig_dir / f"robustness_{results['split']}.png", dpi=110)
    logger.info("Saved robustness results for %s", run_dir.name)
    if not args.no_mlflow:
        log_run(run_dir)


if __name__ == "__main__":
    main()
