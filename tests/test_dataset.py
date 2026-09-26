import pytest
import torch

from defect_detection.data.dataset import ANOMALOUS, NORMAL, MVTecAD2Dataset, index_split
from defect_detection.data.preprocessing import build_eval_transform
from defect_detection.data.validation import validate_dataset
from defect_detection.utils.config import PreprocessingConfig


def test_index_counts_and_labels(dataset_cfg):
    assert len(index_split(dataset_cfg, "train")) == 4
    test = index_split(dataset_cfg, "test")
    assert [s.label for s in test] == [NORMAL] * 2 + [ANOMALOUS] * 3
    assert all(s.mask_path is None for s in test if s.label == NORMAL)
    assert all(s.mask_path.is_file() for s in test if s.label == ANOMALOUS)


def test_index_is_deterministic(dataset_cfg):
    assert index_split(dataset_cfg, "test") == index_split(dataset_cfg, "test")


def test_missing_mask_raises(dataset_cfg):
    mask = dataset_cfg.split_dir("test") / "ground_truth" / "bad" / "001_regular_mask.png"
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="masks missing"):
        index_split(dataset_cfg, "test")


def test_missing_split_raises(dataset_cfg, tmp_path):
    import shutil

    shutil.rmtree(dataset_cfg.split_dir("validation"))
    with pytest.raises(FileNotFoundError):
        index_split(dataset_cfg, "validation")


def test_getitem_schema_and_shapes(dataset_cfg):
    ds = MVTecAD2Dataset.from_config(dataset_cfg, "test", build_eval_transform(PreprocessingConfig(image_size=32)))
    item = ds[len(ds) - 1]
    assert set(item) == {"image", "mask", "label", "path"}
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert item["mask"].shape == (1, 32, 32)
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}
    assert item["label"] == ANOMALOUS
    assert ds[0]["mask"].sum() == 0  # normal sample -> empty mask


def test_validation_report(dataset_cfg):
    summary = validate_dataset(dataset_cfg)
    assert summary["valid"]
    test = next(s for s in summary["splits"] if s["split"] == "test")
    assert (test["num_normal"], test["num_anomalous"]) == (2, 3)
    assert test["image_sizes"] == {"96x64": 5}
    assert test["mask_value_sets"] == {"0,255": 3}


def test_validation_flags_missing_mask(dataset_cfg):
    (dataset_cfg.split_dir("test") / "ground_truth" / "bad" / "000_regular_mask.png").unlink()
    summary = validate_dataset(dataset_cfg)
    assert not summary["valid"]
