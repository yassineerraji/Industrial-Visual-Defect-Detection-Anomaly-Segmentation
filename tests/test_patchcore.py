import json

import pytest
import torch

from defect_detection.evaluation.pipeline import evaluate_run
from defect_detection.models.patchcore import PatchCore, greedy_coreset, nearest_distances
from defect_detection.training.pipeline import train_normal_only


def _model(ratio: float = 1.0) -> PatchCore:
    return PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=ratio).eval()


def _loader(images: torch.Tensor):
    return [{"image": images}]


def test_coreset_covers_separated_clusters():
    torch.manual_seed(0)
    centres = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    points = torch.cat([c + torch.randn(50, 2) for c in centres])
    idx = greedy_coreset(points, 3, projection_dim=None, seed=0)
    assert len(set(idx.tolist())) == 3
    assert sorted((i // 50) for i in idx.tolist()) == [0, 1, 2]
    assert torch.equal(greedy_coreset(points, 1000), torch.arange(150))


def test_nearest_distances_matches_brute_force():
    torch.manual_seed(0)
    q, bank = torch.randn(37, 5), torch.randn(11, 5)
    expected = torch.cdist(q, bank).min(dim=1).values
    assert torch.allclose(nearest_distances(q, bank, chunk_size=8), expected)


def test_embedding_and_map_shapes():
    model = _model()
    x = torch.randn(2, 3, 64, 128)
    assert model.embed(x).shape == (2, 128 + 256, 8, 16)  # layer2 (stride 8) + layer3
    model.fit(_loader(x), torch.device("cpu"))
    assert model.anomaly_map(x).shape == (2, 1, 64, 128)


def test_training_images_score_near_zero_with_full_bank():
    # cdist uses the matrix-product expansion, so self-distances are ~sqrt(eps)*|x|, not exactly 0.
    torch.manual_seed(0)
    model = _model(ratio=1.0)
    x = torch.randn(1, 3, 64, 128)
    model.fit(_loader(x), torch.device("cpu"))
    unseen = model.anomaly_map(torch.randn(1, 3, 64, 128))
    assert model.anomaly_map(x).max() < 0.1 * unseen.mean()


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError, match="empty"):
        _model().anomaly_map(torch.randn(1, 3, 64, 64))


def test_state_dict_roundtrip_restores_memory_bank():
    torch.manual_seed(0)
    model = _model(ratio=0.5)
    x = torch.randn(2, 3, 64, 64)
    model.fit(_loader(x), torch.device("cpu"))
    restored = _model()
    restored.load_state_dict(model.state_dict())
    assert restored.memory_bank.shape == model.memory_bank.shape
    assert torch.equal(restored.anomaly_map(x), model.anomaly_map(x))


def test_backbone_stays_frozen_in_train_mode():
    model = _model().train()
    assert not model.feature_extractor.training
    assert not any(p.requires_grad for p in model.parameters())


def test_patchcore_train_then_evaluate_end_to_end(dataset_cfg, tmp_path):
    cfg = {
        "experiment_name": "pc_test",
        "data": {
            "dataset": {
                "root": str(dataset_cfg.root),
                "category": dataset_cfg.category,
                "layout": {"splits": dict(dataset_cfg.layout.splits)},
            },
            "preprocessing": {"image_size": [64, 128]},
            "seed": 0,
        },
        "model": {"name": "patchcore", "pretrained": False, "coreset_ratio": 0.25},
        "training": {"batch_size": 2, "device": "cpu"},
    }
    run_dir = train_normal_only(cfg, runs_dir=tmp_path / "runs")
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["epochs"] is None and info["fit"]["memory_bank_size"] == round(4 * 8 * 16 * 0.25)
    results = evaluate_run(run_dir, split="test", device_name="cpu", num_figures=2)
    assert 0.0 <= results["image"]["image_auroc"] <= 1.0
