"""Qualitative figures: original | ground truth | anomaly map | overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANEL_WIDTH = 5.0


def select_examples(labels: np.ndarray, num: int, num_normal: int = 2) -> list[int]:
    """Deterministic, evenly spaced picks: mostly anomalous plus a few normal samples."""
    picks: list[int] = []
    for label, k in ((1, num - num_normal), (0, num_normal)):
        idx = np.flatnonzero(labels == label)
        if len(idx) and k > 0:
            picks += idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int)].tolist()
    return picks


def plot_predictions(
    images: list[np.ndarray],
    masks: np.ndarray,
    maps: np.ndarray,
    scores: np.ndarray,
    paths: list[str],
    image_threshold: float,
    pixel_threshold: float,
    vmin: float,
    vmax: float,
) -> plt.Figure:
    """One row per sample. ``images`` are (H, W, 3) in [0, 1]; ``vmin``/``vmax`` fix a shared map colour scale."""
    height, width = masks.shape[1:]
    fig, axes = plt.subplots(
        len(images), 4, figsize=(4 * PANEL_WIDTH, len(images) * (PANEL_WIDTH * height / width + 0.5)), squeeze=False
    )
    for row, image, mask, amap, score, path in zip(axes, images, masks, maps, scores, paths):
        verdict = "FAIL" if score > image_threshold else "PASS"
        row[0].imshow(image)
        row[0].set_title(f"{Path(path).name}", fontsize=9)
        row[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        row[1].set_title("ground truth", fontsize=9)
        row[2].imshow(amap, cmap="inferno", vmin=vmin, vmax=vmax)
        row[2].set_title(f"anomaly map  score={score:.4g}  thr={image_threshold:.4g}  -> {verdict}", fontsize=9)
        row[3].imshow(image)
        row[3].imshow(amap, cmap="inferno", vmin=vmin, vmax=vmax, alpha=0.5)
        if (amap > pixel_threshold).any():
            row[3].contour(amap > pixel_threshold, levels=[0.5], colors="cyan", linewidths=0.6)
        if mask.any():
            row[3].contour(mask, levels=[0.5], colors="lime", linewidths=0.6)
        row[3].set_title("overlay (cyan: predicted, green: ground truth)", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    return fig


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, vmax: float, alpha: float = 0.5) -> np.ndarray:
    """Blend an 'inferno' heatmap (0..vmax) over an (H, W, 3) uint8 image; returns uint8."""
    normalised = np.clip(anomaly_map / max(vmax, 1e-12), 0.0, 1.0)
    heat = (plt.get_cmap("inferno")(normalised)[..., :3] * 255).astype(np.float32)
    weight = alpha * normalised[..., None]  # low scores stay transparent
    return (image.astype(np.float32) * (1 - weight) + heat * weight).astype(np.uint8)


def plot_robustness(rows: list[dict], metrics: tuple[str, ...] = ("image_auroc", "pixel_auroc", "f1")) -> plt.Figure:
    """Metric vs. severity (clean, mild, moderate, strong), one line per perturbation."""
    levels = ["clean", "mild", "moderate", "strong"]
    clean = next(r for r in rows if r["perturbation"] == "clean")
    perturbations = sorted({r["perturbation"] for r in rows} - {"clean"})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.6), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for name in perturbations:
            by_level = {r["severity"]: r[metric] for r in rows if r["perturbation"] == name}
            ys = [clean[metric]] + [by_level.get(level, np.nan) for level in levels[1:]]
            ax.plot(levels, ys, marker="o", label=name)
        ax.set_title(metric)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig
