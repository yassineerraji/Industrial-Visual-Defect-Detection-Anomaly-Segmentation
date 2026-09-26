import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defect_detection.data.dataset import ANOMALOUS, NORMAL, Sample, index_split
from defect_detection.data.folds import grouped_folds, part_id
from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.inference.predictor import Predictor
from defect_detection.training.supervised import train_supervised_cv
from tests.conftest import _save, make_defect_pair, make_normal_image

CONDITIONS = ("regular", "overexposed", "shift_1")


def _samples(n_bad: int, n_good: int) -> list[Sample]:
    out = []
    for label, n in ((ANOMALOUS, n_bad), (NORMAL, n_good)):
        for i in range(n):
            for c in CONDITIONS:
                out.append(Sample(Path(f"{i:03d}_{c}.png"), None, label, "test"))
    return out


def test_part_id_distinguishes_label_and_ignores_condition():
    a, b, c = _samples(1, 1)[0], _samples(1, 1)[1], _samples(1, 1)[3]
    assert part_id(a) == part_id(b)  # same part, different condition
    assert part_id(a) != part_id(c)  # anomalous 000 vs normal 000


def test_folds_never_split_a_part_and_are_stratified():
    samples = _samples(15, 4)
    folds = grouped_folds(samples, n_folds=5, seed=42)
    fold_of_part: dict[str, set[int]] = {}
    for s, f in zip(samples, folds):
        fold_of_part.setdefault(part_id(s), set()).add(f)
    assert all(len(v) == 1 for v in fold_of_part.values())
    anomalous_parts_per_fold = [
        len({part_id(s) for s, f in zip(samples, folds) if f == k and s.label == ANOMALOUS}) for k in range(5)
    ]
    assert anomalous_parts_per_fold == [3, 3, 3, 3, 3]


def test_folds_are_seeded():
    samples = _samples(15, 4)
    assert grouped_folds(samples, 5, 1) == grouped_folds(samples, 5, 1)
    assert grouped_folds(samples, 5, 1) != grouped_folds(samples, 5, 2)


def test_too_few_anomalous_parts_rejected():
    with pytest.raises(ValueError):
        grouped_folds(_samples(2, 2), n_folds=5, seed=0)


@pytest.fixture
def cv_dataset_cfg(dataset_cfg):
    """Extend the synthetic test split to 4 anomalous parts x 2 conditions."""
    split = dataset_cfg.split_dir("test")
    for i in range(4):
        for cond in ("regular", "dark"):
            image, mask = make_defect_pair()
            if cond == "dark":
                image = (image * 0.6).astype(np.uint8)
            _save(split / "bad" / f"{i:03d}_{cond}.png", image, "RGB")
            _save(split / "ground_truth" / "bad" / f"{i:03d}_{cond}_mask.png", mask, "L")
    for i in range(2):
        _save(split / "good" / f"{i:03d}_dark.png", (make_normal_image() * 0.6).astype(np.uint8), "RGB")
    return dataset_cfg


def test_unet_cv_train_then_evaluate(cv_dataset_cfg, tmp_path):
    cfg = {
        "experiment_name": "unet_test",
        "data": {
            "dataset": {
                "root": str(cv_dataset_cfg.root),
                "category": cv_dataset_cfg.category,
                "layout": {"splits": dict(cv_dataset_cfg.layout.splits)},
            },
            "preprocessing": {"image_size": [64, 128]},
            "seed": 0,
        },
        "model": {"name": "unet", "pretrained": False, "decoder_channels": 8},
        "cross_validation": {"n_folds": 2, "split": "test", "final_model": True},
        "training": {"epochs": 1, "batch_size": 2, "loss": "bce_dice", "device": "cpu", "select_best": False},
        "scoring": {"smoothing_sigma": 0.0},
        "threshold": {"image": 0.5, "pixel": 0.5},
    }
    run_dir = train_supervised_cv(cfg, runs_dir=tmp_path / "runs")
    assert (run_dir / "fold_0" / "model.pt").is_file() and (run_dir / "fold_1" / "model.pt").is_file()
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["supervision"] == "pixel_labels"
    for fold in info["folds"]:
        # the official normal train images are always included in training
        assert fold["train_normal"] >= 4 and fold["heldout_anomalous"] > 0

    results = evaluate_run(run_dir, split="test", device_name="cpu", num_figures=2)
    n_test = len(index_split(cv_dataset_cfg, "test"))
    assert results["protocol"]["num_normal"] + results["protocol"]["num_anomalous"] == n_test
    assert "out-of-fold" in results["protocol"]["evaluation"]

    with pytest.raises(ValueError, match="not in this run's folds"):
        evaluate_run(run_dir, split="validation", device_name="cpu", num_figures=0)

    # Deployable all-data model is saved at the run root and loadable for inference.
    assert (run_dir / "model.pt").is_file() and "final" in json.loads((run_dir / "history.json").read_text())
    image = Image.open(next(cv_dataset_cfg.split_dir("test").glob("bad/*.png")))
    assert Predictor.from_run(run_dir, device="cpu").predict(image).anomaly_map.shape == image.size[::-1]

    # Runs are portable: evaluation works after the dataset moves (e.g. trained on Colab, evaluated locally).
    moved_root = tmp_path / "elsewhere"
    shutil.move(str(cv_dataset_cfg.root), moved_root)
    moved = evaluate_run(run_dir, split="test", data_root=str(moved_root), device_name="cpu", num_figures=0)
    assert moved["image"] == results["image"]
