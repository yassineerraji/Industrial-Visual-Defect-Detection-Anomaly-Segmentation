from pathlib import Path

import pytest

from defect_detection.utils.config import PreprocessingConfig, load_data_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml"


def test_repo_dataset_config_parses():
    cfg = load_data_config(REPO_CONFIG)
    assert cfg.dataset.layout.splits["test"] == "test_public"
    assert cfg.preprocessing.image_size == 256
    assert cfg.dataset.layout.image_extensions == (".png",)
    assert cfg.augmentation.blur_sigma is None


def test_data_root_override(tmp_path):
    cfg = load_data_config(REPO_CONFIG, data_root=tmp_path)
    assert cfg.dataset.root == tmp_path
    assert cfg.dataset.split_dir("train") == tmp_path / cfg.dataset.category / "train"


def test_unknown_split_raises():
    cfg = load_data_config(REPO_CONFIG)
    with pytest.raises(KeyError):
        cfg.dataset.split_dir("test_private")


def test_crop_larger_than_resize_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(image_size=128, crop_size=256)
