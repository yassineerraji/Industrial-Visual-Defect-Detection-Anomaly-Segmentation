import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection.data.augmentations import build_train_transform
from defect_detection.data.preprocessing import build_eval_transform, denormalise, load_mask
from defect_detection.utils.config import AugmentationConfig, PreprocessingConfig
from defect_detection.utils.reproducibility import seed_everything
from tests.conftest import make_defect_pair

PRE = PreprocessingConfig(image_size=64)


def _pair():
    image, mask = make_defect_pair()
    return Image.fromarray(image), (mask > 0).astype(np.uint8)


def _assert_aligned(image_t: torch.Tensor, mask_t: torch.Tensor) -> None:
    """Bright pixels must coincide with the mask (tolerating interpolated edges)."""
    brightness = denormalise(image_t, PRE).mean(0)
    mask = mask_t[0].bool()
    assert mask.any()
    assert brightness[mask].mean() > 0.75
    assert brightness[~mask].mean() < 0.3


@pytest.mark.parametrize("crop,expected", [(None, 64), (48, 48)])
def test_eval_output_shapes(crop, expected):
    pre = PreprocessingConfig(image_size=64, crop_size=crop)
    image_t, mask_t = build_eval_transform(pre)(*_pair())
    assert image_t.shape == (3, expected, expected)
    assert mask_t.shape == (1, expected, expected)
    assert mask_t.dtype == torch.float32


def test_eval_transform_is_deterministic():
    t = build_eval_transform(PRE)
    a, b = t(*_pair()), t(*_pair())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_eval_mask_alignment():
    _assert_aligned(*build_eval_transform(PRE)(*_pair()))


def test_geometric_augmentation_keeps_alignment():
    aug = AugmentationConfig(hflip_p=1.0, vflip_p=1.0, rotation_degrees=30.0)
    seed_everything(0)
    image_t, mask_t = build_train_transform(PRE, aug)(*_pair())
    _assert_aligned(image_t, mask_t)
    # The defect box is off-centre, so a flip must actually have moved it.
    eval_mask = build_eval_transform(PRE)(*_pair())[1]
    assert not torch.equal(mask_t, eval_mask)


def test_photometric_augmentation_leaves_mask_binary_and_unchanged():
    aug = AugmentationConfig(brightness=0.3, contrast=0.3, blur_sigma=(0.5, 1.0), noise_std=0.05)
    seed_everything(0)
    _, mask_t = build_train_transform(PRE, aug)(*_pair())
    eval_mask = build_eval_transform(PRE)(*_pair())[1]
    assert torch.equal(mask_t, eval_mask)


def test_train_transform_reproducible_with_seed():
    aug = AugmentationConfig(hflip_p=0.5, rotation_degrees=10.0, brightness=0.2, noise_std=0.02)
    t = build_train_transform(PRE, aug)
    seed_everything(123)
    a = t(*_pair())
    seed_everything(123)
    b = t(*_pair())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_mask_size_mismatch_raises(tmp_path):
    path = tmp_path / "m.png"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="size"):
        load_mask(path, (20, 10))
