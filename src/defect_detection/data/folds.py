"""Grouped cross-validation folds over a labelled split.

MVTec AD 2 photographs each physical part under several acquisition conditions
('000_regular', '000_overexposed', '000_shift_1', ...). All images of one part
must fall in the same fold, otherwise a model is evaluated on parts it saw in
training.
"""

from __future__ import annotations

import random

from defect_detection.data.dataset import ANOMALOUS, Sample
from defect_detection.utils.config import DatasetConfig


def fold_key(cfg: DatasetConfig, sample: Sample) -> str:
    """Machine-independent key: path relative to the category directory, e.g. 'test_public/bad/000_regular.png'."""
    return sample.image_path.relative_to(cfg.category_dir).as_posix()


def part_id(sample: Sample) -> str:
    """Physical-part key: normal and anomalous parts are numbered independently."""
    return f"{sample.label}_{sample.image_path.stem.split('_')[0]}"


def grouped_folds(samples: list[Sample], n_folds: int, seed: int) -> list[int]:
    """Assign a fold index to each sample, grouping by part and stratifying by label.

    Anomalous and normal parts are shuffled separately (seeded) and dealt
    round-robin, so anomalous parts are spread as evenly as possible.
    """
    rng = random.Random(seed)
    fold_of_part: dict[str, int] = {}
    for label in sorted({s.label for s in samples}, reverse=True):  # anomalous first
        parts = sorted({part_id(s) for s in samples if s.label == label})
        rng.shuffle(parts)
        for i, part in enumerate(parts):
            fold_of_part[part] = i % n_folds
    n_anomalous_parts = len({part_id(s) for s in samples if s.label == ANOMALOUS})
    if n_anomalous_parts < n_folds:
        raise ValueError(f"Only {n_anomalous_parts} anomalous parts for {n_folds} folds")
    return [fold_of_part[part_id(s)] for s in samples]
