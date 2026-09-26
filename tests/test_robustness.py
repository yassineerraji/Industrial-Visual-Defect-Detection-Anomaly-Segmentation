from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection.data.preprocessing import build_eval_transform, denormalise
from defect_detection.evaluation.robustness import evaluate_robustness, make_perturbation
from defect_detection.training.pipeline import train_normal_only
from defect_detection.utils.config import PreprocessingConfig, load_yaml
from tests.conftest import make_defect_pair
from tests.test_autoencoder import _tiny_cfg

PRE = PreprocessingConfig(image_size=[32, 64], mean=(0, 0, 0), std=(1, 1, 1))
REPO_SPEC = Path(__file__).resolve().parents[1] / "configs" / "robustness.yaml"


def _apply(perturbation):
    image, mask = make_defect_pair()
    return build_eval_transform(PRE, perturbation)(Image.fromarray(image), (mask > 0).astype(np.uint8))


@pytest.mark.parametrize(
    "name,value", [("brightness", 0.5), ("contrast", 0.5), ("gaussian_noise", 0.1), ("blur", 1.0)]
)
def test_perturbation_changes_image_but_not_mask(name, value):
    clean_img, clean_mask = _apply(None)
    img, mask = _apply(make_perturbation(name, value))
    assert not torch.equal(img, clean_img)
    assert torch.equal(mask, clean_mask)
    assert img.min() >= 0 and img.max() <= 1


def test_brightness_scales_intensity():
    clean, _ = _apply(None)
    dark, _ = _apply(make_perturbation("brightness", 0.5))
    assert torch.allclose(dark, clean * 0.5, atol=1e-6)


def test_noise_is_deterministic():
    a, _ = _apply(make_perturbation("gaussian_noise", 0.05, noise_seed=3))
    b, _ = _apply(make_perturbation("gaussian_noise", 0.05, noise_seed=3))
    assert torch.equal(a, b)


def test_denormalised_view_consistent():
    img, _ = _apply(None)
    assert torch.equal(denormalise(img, PRE), img.clamp(0, 1))


def test_evaluate_robustness_covers_all_conditions(dataset_cfg, tmp_path):
    run_dir = train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "runs")
    spec = load_yaml(REPO_SPEC)
    results = evaluate_robustness(run_dir, spec, device_name="cpu")
    rows = results["results"]
    assert len(rows) == 1 + 4 * 3
    assert rows[0]["perturbation"] == "clean"
    assert {r["severity"] for r in rows[1:]} == {"mild", "moderate", "strong"}
    assert all(0 <= r["image_auroc"] <= 1 for r in rows)
