import json

import mlflow

from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.tracking import flatten, log_run
from defect_detection.training.pipeline import train_normal_only
from tests.test_autoencoder import _tiny_cfg


def test_flatten_nested():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_log_run_then_update_same_run(dataset_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    run_dir = train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "runs")

    run_id = log_run(run_dir, experiment="test_exp")
    evaluate_run(run_dir, split="test", device_name="cpu", num_figures=0)
    assert log_run(run_dir, experiment="test_exp") == run_id  # resumed, not duplicated

    run = mlflow.get_run(run_id)
    assert run.data.params["model.name"] == "autoencoder"
    assert run.data.tags["supervision"] == "normal_only"
    assert "test/image_auroc" in run.data.metrics and "train_loss" in run.data.metrics
    history = mlflow.MlflowClient().get_metric_history(run_id, "train_loss")
    assert len(history) == 2  # one point per epoch
    assert json.loads((run_dir / "mlflow.json").read_text())["run_id"] == run_id
