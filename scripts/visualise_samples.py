"""Save a grid of preprocessed samples with ground-truth mask overlays.

Uses the evaluation transform, so the figure shows exactly what models see.

Example:
    python scripts/visualise_samples.py --config configs/dataset.yaml --split test --num 8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defect_detection.data.dataset import ANOMALOUS, MVTecAD2Dataset, index_split  # noqa: E402
from defect_detection.data.preprocessing import build_eval_transform, denormalise  # noqa: E402
from defect_detection.utils.config import load_yaml, parse_data_config  # noqa: E402
from defect_detection.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("visualise_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=8, help="number of samples to show")
    parser.add_argument("--output-dir", default="artifacts/figures")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    raw = load_yaml(args.config)
    if args.category:
        raw["dataset"]["category"] = args.category
    cfg = parse_data_config(raw, data_root=args.data_root)

    samples = index_split(cfg.dataset, args.split)
    # Prefer anomalous samples so masks are visible; fixed seed for a stable figure.
    anomalous = [s for s in samples if s.label == ANOMALOUS]
    pool = anomalous if anomalous else samples
    chosen = random.Random(cfg.seed).sample(pool, k=min(args.num, len(pool)))
    dataset = MVTecAD2Dataset(chosen, build_eval_transform(cfg.preprocessing))

    height, width = cfg.preprocessing.output_size
    panel_w = 5.0
    fig, axes = plt.subplots(
        len(dataset), 3, figsize=(3 * panel_w, len(dataset) * (panel_w * height / width + 0.4)), squeeze=False
    )
    for row, item in zip(axes, dataset):
        image = denormalise(item["image"], cfg.preprocessing).permute(1, 2, 0).numpy()
        mask = item["mask"][0].numpy()
        row[0].imshow(image)
        row[0].set_title(Path(item["path"]).name, fontsize=8)
        row[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        row[1].set_title("ground truth", fontsize=8)
        row[2].imshow(image)
        row[2].imshow(mask, cmap="Reds", alpha=0.45 * (mask > 0))
        row[2].set_title("overlay", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.dataset.category}_{args.split}_samples.png"
    fig.savefig(out_path, dpi=120)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
