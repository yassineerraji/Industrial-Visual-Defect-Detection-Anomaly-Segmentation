"""Dataset structure validation and summary statistics.

Every number produced here is measured from the files on disk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from defect_detection.data.dataset import expected_mask_path, list_images
from defect_detection.utils.config import DatasetConfig


@dataclass
class SplitReport:
    split: str
    directory: str
    num_normal: int = 0
    num_anomalous: int = 0
    image_sizes: dict[str, int] = field(default_factory=dict)  # "WxH" -> count
    mask_value_sets: dict[str, int] = field(default_factory=dict)  # "0,255" -> count
    mean_defect_area_fraction: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _mask_stats(mask_path: Path, image_size: tuple[int, int], report: SplitReport) -> float | None:
    with Image.open(mask_path) as m:
        if m.size != image_size:
            report.errors.append(f"size mismatch: {mask_path.name} {m.size} vs image {image_size}")
            return None
        arr = np.asarray(m.convert("L"))
    values = ",".join(str(v) for v in np.unique(arr)[:8])
    report.mask_value_sets[values] = report.mask_value_sets.get(values, 0) + 1
    fraction = float((arr > 0).mean())
    if fraction == 0.0:
        report.warnings.append(f"empty mask for anomalous image: {mask_path.name}")
    return fraction


def validate_split(cfg: DatasetConfig, split: str) -> SplitReport:
    split_dir = cfg.split_dir(split)
    report = SplitReport(split=split, directory=str(split_dir))
    if not split_dir.is_dir():
        report.errors.append(f"missing directory {split_dir}")
        return report

    exts = cfg.layout.image_extensions
    good = list_images(split_dir / cfg.layout.good_dir, exts)
    bad = list_images(split_dir / cfg.layout.bad_dir, exts)
    report.num_normal, report.num_anomalous = len(good), len(bad)
    if not good and not bad:
        report.errors.append(f"no images under {split_dir}/{{{cfg.layout.good_dir},{cfg.layout.bad_dir}}}")

    sizes: Counter[str] = Counter()
    for p in good:
        with Image.open(p) as img:
            sizes[f"{img.size[0]}x{img.size[1]}"] += 1

    fractions: list[float] = []
    expected_masks: set[Path] = set()
    for p in bad:
        with Image.open(p) as img:
            size = img.size
        sizes[f"{size[0]}x{size[1]}"] += 1
        mask = expected_mask_path(cfg, split, p)
        expected_masks.add(mask)
        if not mask.is_file():
            report.errors.append(f"missing mask for {p.name} (expected {mask})")
            continue
        fraction = _mask_stats(mask, size, report)
        if fraction is not None:
            fractions.append(fraction)

    mask_dir = split_dir / cfg.layout.mask_dir
    orphans = [m for m in list_images(mask_dir, exts) if m not in expected_masks]
    if orphans:
        report.warnings.append(f"{len(orphans)} masks without a matching image, e.g. {orphans[0].name}")

    report.image_sizes = dict(sizes)
    if fractions:
        report.mean_defect_area_fraction = float(np.mean(fractions))
    return report


def validate_dataset(cfg: DatasetConfig) -> dict:
    reports = [validate_split(cfg, split) for split in cfg.layout.splits]
    train = next((r for r in reports if r.split == "train"), None)
    if train is not None and train.num_anomalous > 0:
        train.warnings.append("train split contains anomalous images; Regime A assumes normal-only training")
    return {
        "root": str(cfg.root),
        "category": cfg.category,
        "splits": [asdict(r) for r in reports],
        "valid": all(not r.errors for r in reports),
    }


def available_categories(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
