"""Validate the on-disk MVTec AD 2 layout for one category and write a summary.

Example:
    python scripts/prepare_data.py --config configs/dataset.yaml --category sheet_metal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defect_detection.data.validation import available_categories, validate_dataset
from defect_detection.utils.config import load_yaml, parse_data_config
from defect_detection.utils.logging import configure_logging, get_logger

logger = get_logger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--data-root", default=None, help="override dataset.root")
    parser.add_argument("--category", default=None, help="override dataset.category")
    parser.add_argument("--output-dir", default="artifacts/data")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    raw = load_yaml(args.config)
    if args.category:
        raw["dataset"]["category"] = args.category
    cfg = parse_data_config(raw, data_root=args.data_root).dataset

    if not cfg.category_dir.is_dir():
        found = available_categories(cfg.root)
        logger.error("Category directory not found: %s", cfg.category_dir)
        logger.error("Categories present under %s: %s", cfg.root, found or "none")
        return 1

    summary = validate_dataset(cfg)
    for split in summary["splits"]:
        logger.info(
            "%-10s normal=%-4d anomalous=%-4d sizes=%s",
            split["split"], split["num_normal"], split["num_anomalous"], split["image_sizes"],
        )
        if split["mean_defect_area_fraction"] is not None:
            logger.info("%-10s mean defect area fraction=%.4f", split["split"], split["mean_defect_area_fraction"])
        for w in split["warnings"][:10]:
            logger.warning("%s: %s", split["split"], w)
        for e in split["errors"][:10]:
            logger.error("%s: %s", split["split"], e)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.category}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s (valid=%s)", out_path, summary["valid"])
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
