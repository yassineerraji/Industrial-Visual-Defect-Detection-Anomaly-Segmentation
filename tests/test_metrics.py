import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from defect_detection.evaluation.metrics import (
    auroc,
    condition_of,
    dice_iou,
    image_metrics,
    per_condition_image_auroc,
    pixel_metrics,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auroc_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 500)
    scores = rng.integers(0, 20, 500).astype(float) + labels * 3  # many ties
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_extremes_and_single_class():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(labels, np.ones(4)) == 0.5
    assert math.isnan(auroc(np.zeros(4), np.arange(4)))


def test_image_metrics_threshold_is_strict():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    m = image_metrics(labels, scores, threshold=0.5)  # 0.5 is NOT anomalous
    assert (m["precision"], m["recall"]) == (1.0, 0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == 0.0


def test_dice_iou_known_values():
    pred = np.array([1, 1, 0, 0], dtype=bool)
    target = np.array([1, 0, 1, 0], dtype=bool)
    assert dice_iou(pred, target) == pytest.approx((0.5, 1 / 3))
    assert dice_iou(np.zeros(3, bool), np.zeros(3, bool)) == (1.0, 1.0)


def test_pixel_metrics_perfect_map():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 1:3, 1:3] = True
    maps = masks.astype(np.float32)
    m = pixel_metrics(masks, maps, threshold=0.5)
    assert m["pixel_auroc"] == 1.0 and m["dice"] == 1.0 and m["iou"] == 1.0
    assert m["mean_dice_anomalous"] == 1.0


def test_condition_parsing_and_grouping():
    assert condition_of("/x/003_shift_1.png") == "shift_1"
    assert condition_of("000_regular.png") == "regular"
    res = per_condition_image_auroc(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.9, 0.1]),
        ["000_regular.png", "001_regular.png", "000_dark.png", "001_dark.png"],
    )
    assert res["regular"]["image_auroc"] == 1.0 and res["dark"]["image_auroc"] == 0.0
