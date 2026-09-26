import numpy as np
import pytest
from PIL import Image

from defect_detection.data.dataset import MVTecAD2Dataset, index_split
from defect_detection.data.loaders import build_loader
from defect_detection.evaluation.benchmark import benchmark_run
from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.evaluation.scoring import compute_map_outputs
from defect_detection.inference.predictor import Predictor
from defect_detection.runs import load_json
from defect_detection.training.pipeline import train_normal_only
from tests.conftest import HEIGHT, WIDTH, make_defect_pair
from tests.test_autoencoder import _tiny_cfg


@pytest.fixture
def run_dir(dataset_cfg, tmp_path):
    return train_normal_only(_tiny_cfg(dataset_cfg), runs_dir=tmp_path / "runs")


def test_predict_output_schema(run_dir):
    predictor = Predictor.from_run(run_dir, device="cpu")
    result = predictor.predict(Image.fromarray(make_defect_pair()[0]))
    d = result.to_dict()
    assert set(d) >= {"is_anomalous", "anomaly_score", "threshold", "anomaly_map", "latency_ms"}
    assert isinstance(d["is_anomalous"], bool) and isinstance(d["anomaly_score"], float)
    assert d["anomaly_map"].shape == (HEIGHT, WIDTH) and d["anomaly_map"].dtype == np.float32
    assert d["latency_ms"] > 0
    assert d["is_anomalous"] == (d["anomaly_score"] > d["threshold"])


def test_predict_accepts_path_and_array(run_dir, tmp_path):
    image = make_defect_pair()[0]
    path = tmp_path / "x.png"
    Image.fromarray(image).save(path)
    predictor = Predictor.from_run(run_dir, device="cpu")
    a, b = predictor.predict(path), predictor.predict(image)
    assert a.anomaly_score == pytest.approx(b.anomaly_score)


def test_predictor_scores_match_evaluation_pipeline(run_dir, dataset_cfg):
    """Single-image inference must reproduce batch evaluation scores."""
    evaluate_run(run_dir, split="test", device_name="cpu", num_figures=0)
    predictor = Predictor.from_run(run_dir, device="cpu")
    sample = sorted(dataset_cfg.split_dir("test").glob("bad/*.png"))[0]
    score = predictor.predict(sample).anomaly_score
    ds = MVTecAD2Dataset([s for s in index_split(dataset_cfg, "test") if s.image_path == sample], predictor.transform)
    out = compute_map_outputs(predictor.model, build_loader(ds, 1, False, 0), predictor.device, predictor.scoring)
    assert score == pytest.approx(float(out.scores[0]), rel=1e-5)
    assert load_json(run_dir / "thresholds.json")["image"] == predictor.thresholds.image


def test_benchmark_reports_operational_metrics(run_dir):
    result = benchmark_run(run_dir, split="validation", num_images=2, repeats=2, warmup=1, device="cpu")
    assert result["num_measurements"] == 4 and result["batch_size"] == 1
    assert result["latency_ms_mean"] > 0 and result["throughput_img_per_s"] > 0
    assert result["model_resolution"] == [32, 64] and result["input_resolution"] == [HEIGHT, WIDTH]
    assert result["checkpoint_mb"] > 0 and result["peak_process_rss_mb"] > 0
