"""Build results/results.json, the full results table and the compact README table from selected runs.

Example:
    python scripts/make_results_table.py --runs <ae_run> <patchcore_run> <unet_run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from defect_detection.evaluation.report import collect_row, render_compact, render_markdown, update_readme
from defect_detection.runs import resolve_run_dir, save_json
from defect_detection.utils.logging import configure_logging, get_logger

logger = get_logger("make_results_table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--readme", default="README.md")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    rows = [collect_row(resolve_run_dir(r), args.split) for r in args.runs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "results.json", rows)
    table = render_markdown(rows)
    (out_dir / "results_table.md").write_text(table)
    if update_readme(Path(args.readme), render_compact(rows)):
        logger.info("README results table updated")
    else:
        logger.warning("README has no RESULTS markers; table written to %s only", out_dir / "results_table.md")
    print(table)


if __name__ == "__main__":
    main()
