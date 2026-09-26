"""Industrial visual inspection demo.

Runs from a built Space folder (see scripts/build_space.py):
    python scripts/build_space.py && cd build/hf_space && streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

from defect_detection.evaluation.visualisation import overlay_heatmap
from defect_detection.inference.predictor import PredictionResult, Predictor

BASE = Path(__file__).resolve().parent
DISPLAY_WIDTH = 1600
MODEL_LABELS = {"autoencoder": "Autoencoder", "patchcore": "PatchCore", "unet": "U-Net"}
SUPERVISION = {
    "normal_only": "Regime A · trained on normal images only",
    "pixel_labels": "Regime C · trained with pixel-level defect masks",
}
DRIFT_Z = 3.0

st.set_page_config(page_title="Visual Defect Inspection", page_icon="🔍", layout="wide")


# ---------- loading ----------------------------------------------------------------------------

def read_json(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.is_file() else None


@st.cache_data
def load_demo() -> tuple[dict, dict, list]:
    demo = read_json(BASE / "demo.json")
    if demo is None:
        return {}, {}, []
    releases = {}
    for name in demo["releases"]:
        folder = BASE / "releases" / name
        releases[name] = {
            "dir": str(folder),
            "meta": read_json(folder / "release.json"),
            "metrics": read_json(folder / "metrics_test.json"),
            "robustness": read_json(folder / "robustness_test.json"),
            "folds": read_json(folder / "folds.json"),
        }
    return demo, releases, read_json(BASE / "samples.json") or []


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(folder: str, model_dir: str | None = None) -> Predictor:
    return Predictor.from_run(folder, model_dir=model_dir)


def heldout_model_dir(release: dict, sample: dict | None) -> str | None:
    """For cross-validated models, the fold model that never saw this labelled sample."""
    folds = release["folds"]
    if sample is None or folds is None or sample["source"] not in folds:
        return None
    return str(Path(release["dir"]) / f"fold_{folds[sample['source']]}")


@st.cache_data
def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@st.cache_data
def thumbnail(path: str, width: int = 480) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img.resize((width, max(1, round(img.height * width / img.width)))))


# ---------- rendering helpers -----------------------------------------------------------------

def label_of(name: str, releases: dict) -> str:
    return MODEL_LABELS.get(releases[name]["meta"]["model"], name)


def resize_for_display(image: np.ndarray, amap: np.ndarray, mask: np.ndarray | None):
    h, w = image.shape[:2]
    size = (DISPLAY_WIDTH, round(h * DISPLAY_WIDTH / w))
    img = np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))
    amap = np.asarray(Image.fromarray(amap.astype(np.float32), mode="F").resize(size, Image.BILINEAR))
    if mask is not None:
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST)) > 0
    return img, amap, mask


def draw_outline(image: np.ndarray, mask: np.ndarray, colour=(57, 255, 20)) -> np.ndarray:
    m = Image.fromarray((mask * 255).astype(np.uint8))
    edge = (np.asarray(m.filter(ImageFilter.MaxFilter(5))) > 0) & ~(np.asarray(m.filter(ImageFilter.MinFilter(3))) > 0)
    out = image.copy()
    out[edge] = colour
    return out


def verdict_badge(result: PredictionResult, truth: int | None) -> str:
    status, colour = ("FAIL", "#c62828") if result.is_anomalous else ("PASS", "#2e7d32")
    badge = (f"<span style='padding:0.25rem 0.9rem;border-radius:4px;background:{colour};color:white;"
             f"font-weight:700;font-size:1.15rem'>{status}</span>")
    if truth is None:
        return badge
    if bool(truth) == result.is_anomalous:
        note = "✓ correct"
    else:
        note = "✗ missed defect" if truth else "✗ false alarm"
    return f"{badge}&nbsp;&nbsp;<span style='font-size:1.05rem'>{note}</span>"


def record(model: str, result: PredictionResult, image: np.ndarray, meta: dict, source: str) -> None:
    ref = meta["reference"]
    brightness = float(image.astype(np.float32).mean() / 255.0)
    st.session_state.history.append({
        "time": time.strftime("%H:%M:%S"),
        "model": model,
        "input": source,
        "verdict": "FAIL" if result.is_anomalous else "PASS",
        "score / threshold": result.anomaly_score / result.threshold,
        "latency_ms": result.latency_ms,
        "brightness": brightness,
        "brightness_z": (brightness - ref["brightness_mean"]) / max(ref["brightness_std"], 1e-6),
    })


# ---------- tabs ---------------------------------------------------------------------------------

def inspect_tab(releases: dict, samples: list) -> None:
    st.markdown("Pick a sample part (or upload your own image), choose models, and compare their verdicts.")
    st.caption("Samples come from the MVTec AD 2 public test set, chosen by a fixed rule (not by model "
               "performance). Green outlines show the ground-truth defect.")

    cols = st.columns(4)
    for i, s in enumerate(samples):
        with cols[i % 4]:
            st.image(thumbnail(str(BASE / s["file"])), width="stretch")
            kind = "Defective" if s["label"] else "Normal"
            if st.button(f"{kind} · {s['condition']}", key=f"sample_{i}", width="stretch"):
                st.session_state.source = ("sample", i)

    upload = st.file_uploader("…or upload an image", type=["png", "jpg", "jpeg", "bmp"])
    if upload is not None and st.session_state.get("upload_id") != upload.file_id:
        st.session_state.upload_id = upload.file_id
        st.session_state.source = ("upload", upload.file_id)

    chosen = st.multiselect("Models", list(releases), default=list(releases),
                            format_func=lambda n: label_of(n, releases))
    source = st.session_state.get("source")
    if source is None or not chosen:
        st.info("Select a sample or upload an image to run an inspection.")
        return

    if source[0] == "sample":
        s = samples[source[1]]
        image, truth, name = load_rgb(str(BASE / s["file"])), s["label"], Path(s["source"]).name
        mask = load_rgb(str(BASE / s["mask"]))[..., 0] if s["mask"] else np.zeros(image.shape[:2], np.uint8)
        st.markdown(f"**Input:** `{name}` · ground truth **{'DEFECTIVE' if truth else 'NORMAL'}** · "
                    f"{image.shape[1]}×{image.shape[0]} px")
    else:
        if upload is None or upload.file_id != source[1]:
            st.info("Upload removed; select a sample or upload an image.")
            return
        image, truth, mask, name = np.asarray(Image.open(upload).convert("RGB")), None, None, upload.name
        st.markdown(f"**Input:** `{name}` · ground truth unknown · {image.shape[1]}×{image.shape[0]} px")

    for model in chosen:
        meta = releases[model]["meta"]
        # Streamlit reruns the script on every interaction: predict (and log) each input/model pair once.
        key = (source, model)
        sample = samples[source[1]] if source[0] == "sample" else None
        model_dir = heldout_model_dir(releases[model], sample)
        if key not in st.session_state.predictions:
            result = get_predictor(releases[model]["dir"], model_dir).predict(image)
            st.session_state.predictions[key] = result
            record(label_of(model, releases), result, image, meta, name)
        result = st.session_state.predictions[key]

        st.markdown(f"##### {label_of(model, releases)}")
        note = SUPERVISION.get(meta["supervision"], meta["supervision"])
        if model_dir is not None:
            note += (f" · scored by cross-validation model {Path(model_dir).name}, which never saw this part "
                     "(the deployed model was trained on all labelled images, including this one)")
        st.caption(note)
        left, right = st.columns([1, 3])
        with left:
            st.markdown(verdict_badge(result, truth), unsafe_allow_html=True)
            st.metric("Anomaly score", f"{result.anomaly_score:.4g}")
            st.metric("Threshold", f"{result.threshold:.4g}")
            st.metric("Latency", f"{result.latency_ms:.0f} ms")
        with right:
            img, amap, m = resize_for_display(image, result.anomaly_map, mask)
            shown = overlay_heatmap(img, amap, meta["display"]["vmin"], meta["display"]["vmax"])
            if m is not None and m.any():
                shown = draw_outline(shown, m)
            st.image(shown, width="stretch")
    st.caption("Anomaly scores are unnormalised model outputs, not probabilities. FAIL means score > threshold; "
               "thresholds were fixed on validation data (never on test data). Heatmap: transparent below the "
               "99th percentile of normal pixels, pixel threshold at mid-scale.")


def results_tab(releases: dict) -> None:
    results = read_json(BASE / "results.json")
    st.markdown("All numbers below are generated from experiment artifacts; nothing is typed in by hand.")
    if results:
        table = pd.DataFrame([{
            "Model": MODEL_LABELS.get(r["model"], r["model"]),
            "Labels needed": {"normal_only": "none", "pixel_labels": "pixel masks"}.get(r["supervision"]),
            "Image AUROC": r["image_auroc"], "Image AP": r["image_ap"], "F1": r["f1"],
            "False-positive rate": r.get("false_positive_rate"), "Pixel AUROC": r["pixel_auroc"], "Dice": r["dice"],
            "CPU latency (ms)": r.get("latency_cpu_ms"), "Checkpoint (MB)": r.get("checkpoint_mb"),
        } for r in results])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[2:8]})
        st.caption("Test set: MVTec AD 2 sheet_metal test_public (24 normal, 90 defective images of 19 physical "
                   "parts). The U-Net is evaluated with 5-fold cross-validation grouped by part, a different "
                   "protocol from the normal-only models. With 79% defective images, a model flagging everything "
                   "gets F1 = 0.88, so always read F1 together with the false-positive rate.")
    else:
        st.info("Results table not bundled with this build.")

    st.subheader("Robustness to image degradation")
    metric = st.radio("Metric", ["image_auroc", "pixel_auroc", "false_positive_rate"], horizontal=True,
                      format_func=lambda m: m.replace("_", " "))
    levels = ["clean", "mild", "moderate", "strong"]
    cols = st.columns(max(1, sum(r["robustness"] is not None for r in releases.values())))
    i = 0
    for name, rel in releases.items():
        if rel["robustness"] is None:
            continue
        rows = rel["robustness"]["results"]
        clean = next(r for r in rows if r["perturbation"] == "clean")[metric]
        frame = pd.DataFrame(index=levels)
        for pert in sorted({r["perturbation"] for r in rows} - {"clean"}):
            by = {r["severity"]: r[metric] for r in rows if r["perturbation"] == pert}
            frame[pert] = [clean] + [by.get(level) for level in levels[1:]]
        frame.index = pd.CategoricalIndex(frame.index, categories=levels, ordered=True)
        with cols[i]:
            st.markdown(f"**{label_of(name, releases)}**")
            st.line_chart(frame, height=260)
        i += 1
    st.caption("Brightness, contrast, Gaussian noise and blur at fixed severities; thresholds unchanged, "
               "as they would be in production.")

    st.subheader("Performance by acquisition condition")
    per_cond = {label_of(n, releases): {c: v["image_auroc"] for c, v in r["metrics"]["per_condition"].items()}
                for n, r in releases.items() if r["metrics"]}
    if per_cond:
        st.bar_chart(pd.DataFrame(per_cond), stack=False, height=280)
        st.caption("Image AUROC per condition (4 normal + 15 defective images each; small samples, wide uncertainty).")


def how_tab(demo: dict) -> None:
    st.markdown("""
**Question:** how well do different deep-learning approaches detect and localise defects when defect
labels are scarce or available at different granularities?
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="Helvetica", fontsize=11];
  data [label="MVTec AD 2\\nsheet_metal"]; pre [label="Preprocessing\\n256×1024, aspect kept"];
  ae [label="Autoencoder\\n(normal only)"]; pc [label="PatchCore\\n(normal only)"]; un [label="U-Net\\n(pixel masks)"];
  thr [label="Thresholds from\\nvalidation data"]; ev [label="Evaluation\\ndetection + localisation"];
  rob [label="Robustness &\\nlatency benchmark"]; mlf [label="MLflow\\ntracking"]; rel [label="Release\\nbundles"];
  app [label="This app\\n(Docker, CPU)"];
  data -> pre -> {ae pc un}; {ae pc} -> thr; {thr un} -> ev -> rob -> mlf -> rel -> app;
}""")
    st.markdown("""
| Regime | Labels | Model | How it decides |
|---|---|---|---|
| A | none, good parts only | Convolutional autoencoder | Reconstruction error |
| A | none, good parts only | PatchCore (ResNet-18) | Distance to the nearest normal patch feature |
| C | pixel-level defect masks | U-Net (ResNet-18 encoder) | Supervised per-pixel segmentation |

**Keeping the evaluation honest**
- The test data never influences training, checkpoints, thresholds or preprocessing.
- Each physical part appears in 6 photos (lighting/position variants). U-Net cross-validation folds are grouped by part,
  so a model is never tested on a part it saw during training.
- Models are compared on detection *and* localisation, robustness, latency and labelling cost, not on one metric.
""")
    st.link_button("Source code, protocol and full results on GitHub", demo.get("github_url", ""))


def monitoring_tab(releases: dict) -> None:
    history = st.session_state.history
    st.markdown("Live statistics for inspections run in this session, as a production dashboard would track them.")
    if not history:
        st.info("Run a few inspections to populate the monitoring view.")
        return
    df = pd.DataFrame(history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspections", len(df))
    c2.metric("FAIL rate", f"{(df['verdict'] == 'FAIL').mean():.0%}")
    c3.metric("Mean latency", f"{df['latency_ms'].mean():.0f} ms")
    c4.metric("p95 latency", f"{df['latency_ms'].quantile(0.95):.0f} ms")

    left, right = st.columns(2)
    with left:
        st.markdown("**Latency per inspection (ms)**")
        st.line_chart(df.pivot_table(index=df.index, columns="model", values="latency_ms"), height=240)
    with right:
        st.markdown("**Score relative to threshold** (above 1 means FAIL)")
        st.line_chart(df.pivot_table(index=df.index, columns="model", values="score / threshold"), height=240)

    st.markdown("**Input drift check:** mean brightness vs. the training images")
    drifted = df["brightness_z"].abs() > DRIFT_Z
    if drifted.any():
        st.warning(f"{int(drifted.sum())} input(s) differ from the training brightness by more than {DRIFT_Z:.0f} "
                   "standard deviations. Predictions on such inputs are less reliable (see Results → Robustness).")
    else:
        st.success(f"All inputs within ±{DRIFT_Z:.0f} standard deviations of the training brightness.")
    st.dataframe(df.iloc[::-1], hide_index=True, width="stretch",
                 column_config={"score / threshold": st.column_config.NumberColumn(format="%.2f"),
                                "latency_ms": st.column_config.NumberColumn(format="%.0f"),
                                "brightness": st.column_config.NumberColumn(format="%.3f"),
                                "brightness_z": st.column_config.NumberColumn(format="%+.1f")})


def main() -> None:
    demo, releases, samples = load_demo()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("predictions", {})
    st.title("Industrial Visual Defect Inspection")
    st.caption("Anomaly detection and defect localisation on MVTec AD 2 · three supervision regimes compared")
    if not releases:
        st.error("No release bundles found. Build the app with `python scripts/build_space.py`.")
        return

    with st.sidebar:
        st.header("About")
        st.markdown("A benchmark of deep-learning approaches to visual quality control, from models that "
                    "need only images of good parts to fully supervised segmentation.")
        st.link_button("GitHub repository", demo.get("github_url", ""), width="stretch")
        st.caption(demo.get("licence", ""))

    inspect, results, how, monitoring = st.tabs(["🔍 Inspect", "📊 Results", "⚙️ How it works", "📈 Monitoring"])
    with inspect:
        inspect_tab(releases, samples)
    with results:
        results_tab(releases)
    with how:
        how_tab(demo)
    with monitoring:
        monitoring_tab(releases)


main()
