"""Synthetic MVTec AD 2-style dataset fixture.

Anomalous images contain a bright rectangle on a dark background, and the mask
marks exactly that rectangle, so image/mask alignment can be checked numerically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defect_detection.utils.config import DatasetConfig, DatasetLayout

WIDTH, HEIGHT = 96, 64  # non-square on purpose
BACKGROUND, DEFECT = 40, 230
DEFECT_BOX = (slice(10, 30), slice(50, 80))  # rows, cols: off-centre, asymmetric


def _save(path: Path, array: np.ndarray, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def make_normal_image() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)


def make_defect_pair() -> tuple[np.ndarray, np.ndarray]:
    image = make_normal_image()
    image[DEFECT_BOX] = DEFECT
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[DEFECT_BOX] = 255
    return image, mask


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DatasetConfig:
    cat = tmp_path / "mvtec" / "widget"
    for split, n_good, n_bad in (("train", 4, 0), ("validation", 2, 0), ("test_public", 2, 3)):
        for i in range(n_good):
            _save(cat / split / "good" / f"{i:03d}_regular.png", make_normal_image(), "RGB")
        for i in range(n_bad):
            image, mask = make_defect_pair()
            _save(cat / split / "bad" / f"{i:03d}_regular.png", image, "RGB")
            _save(cat / split / "ground_truth" / "bad" / f"{i:03d}_regular_mask.png", mask, "L")
    return DatasetConfig(
        root=tmp_path / "mvtec",
        category="widget",
        layout=DatasetLayout(splits={"train": "train", "validation": "validation", "test": "test_public"}),
    )
