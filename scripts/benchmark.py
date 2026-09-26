"""Measure inference latency, throughput, model size and memory for a saved run.

Run one benchmark per process so peak-memory figures are not shared between models.

Example:
    python scripts/benchmark.py --run autoencoder_sheet_metal_20260926-160000
"""

from __future__ import annotations

import argparse

from defect_detection.evaluation.benchmark import benchmark_run
from defect_detection.runs import resolve_run_dir, save_json
from defect_detection.tracking import log_run
from defect_detection.utils.logging import configure_logging, get_logger

logger = get_logger("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="validation", help="images used for timing (labels are not used)")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    result = benchmark_run(run_dir, args.split, args.num_images, args.repeats, device=args.device)
    save_json(run_dir / "benchmark.json", result)
    logger.info(
        "%s on %s: %.1f ms mean (p95 %.1f), %.1f img/s, %.1f MB checkpoint, %.0f MB peak RSS",
        result["model"], result["device"], result["latency_ms_mean"], result["latency_ms_p95"],
        result["throughput_img_per_s"], result["checkpoint_mb"], result["peak_process_rss_mb"],
    )
    if not args.no_mlflow:
        log_run(run_dir)


if __name__ == "__main__":
    main()
