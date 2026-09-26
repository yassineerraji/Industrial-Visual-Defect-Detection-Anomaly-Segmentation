"""Visual inspection demo: upload an image, pick a model, get PASS/FAIL with a defect heatmap.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from defect_detection.evaluation.visualisation import overlay_heatmap
from defect_detection.inference.predictor import PredictionResult, Predictor
from defect_detection.runs import RUNS_DIR, load_json

MODEL_LABELS = {"autoencoder": "Autoencoder", "patchcore": "PatchCore", "unet": "U-Net"}
SUPERVISION_LABELS = {"normal_only": "normal images only", "pixel_labels": "pixel-level labels"}

st.set_page_config(page_title="Visual Defect Inspection", page_icon="🔍", layout="wide")


def deployable_runs(base: Path = RUNS_DIR) -> dict[str, Path]:
    """Single-model runs with weights and thresholds (CV runs are evaluation-only)."""
    runs = {}
    for run_dir in sorted(base.glob("*"), reverse=True):
        if (run_dir / "model.pt").is_file() and (run_dir / "run_info.json").is_file():
            info = load_json(run_dir / "run_info.json")
            runs[f"{MODEL_LABELS.get(info['model'], info['model'])} · {run_dir.name}"] = run_dir
    return runs


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(run_dir: str) -> Predictor:
    return Predictor.from_run(run_dir)


def test_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "metrics_test.json"
    return load_json(path) if path.is_file() else None


def render_result(label: str, run_dir: Path, image: np.ndarray, result: PredictionResult) -> None:
    info = load_json(run_dir / "run_info.json")
    status = "FAIL" if result.is_anomalous else "PASS"
    colour = "#c62828" if result.is_anomalous else "#2e7d32"
    st.markdown(f"#### {label.split(' · ')[0]}")
    st.markdown(
        f"<div style='padding:0.6rem 1rem;border-radius:6px;background:{colour};color:white;"
        f"font-size:1.4rem;font-weight:600;display:inline-block'>{status}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly score", f"{result.anomaly_score:.4g}")
    c2.metric("Threshold", f"{result.threshold:.4g}")
    c3.metric("Latency", f"{result.latency_ms:.0f} ms")
    st.caption(
        f"Trained on {SUPERVISION_LABELS.get(info['supervision'], info['supervision'])}; "
        f"model resolution {info['image_size'][0]}×{info['image_size'][1]}; device {get_predictor(str(run_dir)).device}. "
        "The score is an unnormalised anomaly score (not a probability); FAIL means score > threshold, "
        "with the threshold chosen on normal validation images."
    )
    overlay = overlay_heatmap(image, result.anomaly_map, vmax=2 * result.pixel_threshold)
    st.image(overlay, caption="Defect heatmap overlay (colour scale 0 → 2× pixel threshold)", width="stretch")


def main() -> None:
    st.title("Industrial Visual Defect Inspection")
    st.caption("MVTec AD 2 · anomaly detection and localisation")

    runs = deployable_runs()
    if not runs:
        st.warning("No trained runs found in artifacts/runs. Train a model first: "
                   "`python scripts/train.py --config configs/patchcore.yaml`")
        return

    with st.sidebar:
        st.header("Model")
        selected = st.multiselect("Select model(s)", list(runs), default=list(runs)[:1])
        for label in selected:
            metrics = test_metrics(runs[label])
            if metrics:
                st.caption(
                    f"**{label.split(' · ')[0]}** test image AUROC {metrics['image']['image_auroc']:.3f}, "
                    f"pixel AUROC {metrics['pixel']['pixel_auroc']:.3f}"
                )
        st.header("Image")
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

    if upload is None:
        st.info("Upload an image in the sidebar to run an inspection.")
        return
    image = np.asarray(Image.open(upload).convert("RGB"))
    st.image(image, caption=f"Input ({image.shape[1]}×{image.shape[0]})", width="stretch")

    if not selected:
        st.info("Select at least one model.")
        return
    if st.button("Run inspection", type="primary"):
        for label in selected:
            result = get_predictor(str(runs[label])).predict(image)
            render_result(label, runs[label], image, result)
            st.divider()


main()
