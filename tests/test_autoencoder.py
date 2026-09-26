import json

import pytest
import torch
import yaml

from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.models.autoencoder import ConvAutoencoder
from defect_detection.runs import load_run
from defect_detection.training.pipeline import train_normal_only


@pytest.mark.parametrize("shape", [(2, 3, 32, 128), (1, 3, 64, 64)])
def test_forward_and_anomaly_map_shapes(shape):
    model = ConvAutoencoder(base_channels=8, num_downsamples=3, latent_channels=4).eval()
    x = torch.rand(shape)
    out = model(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    amap = model.anomaly_map(x)
    assert amap.shape == (shape[0], 1, *shape[2:])
    assert (amap >= 0).all()


def test_indivisible_input_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ConvAutoencoder(num_downsamples=3)(torch.rand(1, 3, 30, 64))


def _tiny_cfg(dataset_cfg) -> dict:
    return {
        "experiment_name": "ae_test",
        "data": {
            "dataset": {
                "root": str(dataset_cfg.root),
                "category": dataset_cfg.category,
                "layout": {"splits": dict(dataset_cfg.layout.splits)},
            },
            "preprocessing": {"image_size": [32, 64], "mean": [0, 0, 0], "std": [1, 1, 1]},
            "seed": 0,
        },
        "model": {"name": "autoencoder", "base_channels": 4, "num_downsamples": 2, "latent_channels": 2},
        "training": {"epochs": 2, "batch_size": 2, "device": "cpu"},
        "scoring": {"smoothing_sigma": 1.0},
    }


def test_train_then_evaluate_end_to_end(dataset_cfg, tmp_path):
    run_dir = train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "runs")
    for name in ("config.yaml", "model.pt", "thresholds.json", "history.json", "run_info.json"):
        assert (run_dir / name).is_file()
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["image_size"] == [32, 64] and info["seed"] == 0 and info["epochs"] == 2

    cfg, model, thresholds = load_run(run_dir, torch.device("cpu"))
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == cfg
    assert thresholds.image > 0 and thresholds.pixel > 0

    results = evaluate_run(run_dir, split="test", device_name="cpu", num_figures=3)
    assert results["protocol"]["num_normal"] == 2 and results["protocol"]["num_anomalous"] == 3
    assert set(results["image"]) >= {"image_auroc", "image_ap", "precision", "recall", "f1"}
    assert set(results["pixel"]) >= {"pixel_auroc", "dice", "iou"}
    assert (run_dir / "metrics_test.json").is_file()
    assert (run_dir / "figures" / "predictions_test.png").is_file()


def test_training_is_reproducible(dataset_cfg, tmp_path):
    a = train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "a")
    b = train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "b")
    ha = json.loads((a / "history.json").read_text())
    hb = json.loads((b / "history.json").read_text())
    assert [r["train_loss"] for r in ha] == [r["train_loss"] for r in hb]
