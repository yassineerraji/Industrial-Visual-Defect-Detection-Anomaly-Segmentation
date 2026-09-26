import numpy as np
import pytest
import torch

from defect_detection.evaluation.scoring import ScoringConfig, image_scores, smooth_maps
from defect_detection.evaluation.thresholds import ThresholdConfig, image_threshold, select_thresholds


def test_image_scores_max_and_topk():
    maps = torch.zeros(2, 1, 10, 10)
    maps[0, 0, 0, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    assert image_scores(maps, ScoringConfig(image_score="max")).tolist() == [5.0, 0.0]
    topk = image_scores(maps, ScoringConfig(image_score="topk_mean", topk_fraction=0.02))  # k=2
    assert topk.tolist() == [4.5, 0.0]


def test_smoothing_preserves_shape_and_mass_centre():
    maps = torch.zeros(1, 1, 33, 33)
    maps[0, 0, 16, 16] = 1.0
    out = smooth_maps(maps, sigma=2.0)
    assert out.shape == maps.shape
    assert out[0, 0].argmax().item() == 16 * 33 + 16
    assert torch.equal(smooth_maps(maps, 0.0), maps)


def test_image_threshold_rules():
    scores = np.array([1.0, 2.0, 3.0])
    assert image_threshold(scores, ThresholdConfig(image_method="mean_std", image_k=1.0)) == pytest.approx(
        2.0 + np.std(scores)
    )
    assert image_threshold(scores, ThresholdConfig(image_method="quantile", image_quantile=0.5)) == 2.0
    with pytest.raises(ValueError):
        image_threshold(np.array([]), ThresholdConfig())


def test_pixel_threshold_bounds_normal_false_positive_rate():
    rng = np.random.default_rng(0)
    maps = rng.random((4, 50, 50))
    t = select_thresholds(maps.max(axis=(1, 2)), maps, ThresholdConfig(pixel_quantile=0.99))
    assert (maps > t.pixel).mean() == pytest.approx(0.01, abs=0.002)
    assert "validation" in t.source
