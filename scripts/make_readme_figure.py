"""Render the README figure: one gallery part, ground truth and each model's heatmap and verdict.

Uses the first gallery sample of configs/demo.yaml (fixed rule, not chosen for performance) and
scores cross-validated models out-of-fold, exactly as the demo app does. Panels show the square
window (image height x image height) containing the most ground-truth pixels, so thin defects
stay visible. Green: ground truth; cyan: region above the model's pixel threshold.

    python scripts/make_readme_figure.py      # after scripts/build_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from defect_detection.evaluation.visualisation import overlay_heatmap  # noqa: E402
from defect_detection.inference.predictor import Predictor  # noqa: E402

ASSETS = Path("build/demo/assets")
OUT = Path("docs/demo_example.jpg")
LABELS = {"autoencoder": "Autoencoder", "patchcore": "PatchCore", "unet": "U-Net"}
PANEL = 480  # output pixels per square panel
GREEN, CYAN = (57, 255, 20), (0, 229, 255)


def densest_window(mask: np.ndarray, width: int) -> slice:
    """Column window of ``width`` containing the most mask pixels, re-centred on those pixels."""
    cols = np.concatenate([[0], np.cumsum(mask.sum(axis=0))])
    start = int(np.argmax(cols[width:] - cols[:-width]))
    xs = np.nonzero(mask[:, start:start + width])[1]
    if xs.size:
        centre = start + int(xs.mean())
        start = int(np.clip(centre - width // 2, 0, mask.shape[1] - width))
    return slice(start, start + width)


def to_panel(array: np.ndarray, resample) -> np.ndarray:
    mode = "F" if array.dtype == np.float32 else None
    return np.asarray(Image.fromarray(array, mode=mode).resize((PANEL, PANEL), resample))


def outline(image: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    from PIL import ImageFilter

    m = Image.fromarray((mask * 255).astype(np.uint8))
    edge = (np.asarray(m.filter(ImageFilter.MaxFilter(5))) > 0) & ~(np.asarray(m.filter(ImageFilter.MinFilter(3))) > 0)
    out = image.copy()
    out[edge] = colour
    return out


def main() -> None:
    demo = json.loads((ASSETS / "demo.json").read_text())
    sample = json.loads((ASSETS / "samples.json").read_text())[0]
    image = np.asarray(Image.open(ASSETS / sample["file"]).convert("RGB"))
    mask = np.asarray(Image.open(ASSETS / sample["mask"]).convert("L")) > 0 if sample["mask"] else None
    h, w = image.shape[:2]
    window = densest_window(mask if mask is not None else np.zeros((h, w), bool), h)
    crop = to_panel(np.ascontiguousarray(image[:, window]), Image.BILINEAR)
    gt = to_panel((mask[:, window] * 255).astype(np.uint8), Image.NEAREST) > 0 if mask is not None else None

    order = ["patchcore", "unet", "autoencoder"]
    truth = "defective" if sample["label"] else "normal"
    panels = [(f"Input (ground truth: {truth})", outline(crop, gt, GREEN) if gt is not None else crop)]
    for name in [n for n in order if n in demo["releases"]]:
        folder = ASSETS / "releases" / name
        folds_path = folder / "folds.json"
        model_dir = None
        if folds_path.is_file():  # score out-of-fold: the fold model never saw this part
            model_dir = folder / f"fold_{json.loads(folds_path.read_text())[sample['source']]}"
        meta = json.loads((folder / "release.json").read_text())
        result = Predictor.from_run(folder, device="cpu", model_dir=model_dir).predict(image)
        amap = to_panel(np.ascontiguousarray(result.anomaly_map[:, window]), Image.BILINEAR)
        panel = overlay_heatmap(crop, amap, meta["display"]["vmin"], meta["display"]["vmax"])
        panel = outline(panel, amap > result.pixel_threshold, CYAN)
        if gt is not None:
            panel = outline(panel, gt, GREEN)
        verdict = "FAIL" if result.is_anomalous else "PASS"
        note = "✓" if result.is_anomalous == bool(sample["label"]) else "✗"
        labels_needed = "no labels" if meta["supervision"] == "normal_only" else "pixel labels"
        panels.append((f"{LABELS[name]} ({labels_needed})\n{verdict} {note}", panel))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.75))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140, pil_kwargs={"quality": 85})
    print(f"saved {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
