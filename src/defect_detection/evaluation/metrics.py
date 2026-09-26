"""Image-level detection and pixel-level localisation metrics."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact, tie-aware AUROC via the Mann-Whitney U statistic.

    Memory-light enough for tens of millions of pixels. Returns NaN when only
    one class is present.
    """
    labels = np.asarray(labels).ravel().astype(bool)
    scores = np.asarray(scores).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)  # average ranks for ties
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Threshold-free (AUROC, AP) and thresholded (precision, recall, F1) detection metrics.

    A score strictly above the threshold is predicted anomalous.
    """
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(scores) > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "image_auroc": auroc(labels, scores),
        "image_ap": float(average_precision_score(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(preds[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Dice and IoU of two boolean arrays; both are 1.0 when both are empty."""
    intersection = np.logical_and(pred, target).sum(dtype=np.int64)
    p, t = pred.sum(dtype=np.int64), target.sum(dtype=np.int64)
    union = p + t - intersection
    if p + t == 0:
        return 1.0, 1.0
    return float(2 * intersection / (p + t)), float(intersection / union)


def pixel_metrics(masks: np.ndarray, maps: np.ndarray, threshold: float) -> dict[str, float]:
    """Pixel AUROC over all images, plus Dice/IoU at ``threshold``.

    ``dice``/``iou`` pool all pixels of all images; ``mean_dice_anomalous`` averages
    per-image Dice over anomalous images only (normal images have empty masks).
    """
    masks = masks.astype(bool)
    preds = maps > threshold
    dice, iou = dice_iou(preds, masks)
    anomalous = masks.reshape(len(masks), -1).any(axis=1)
    per_image = [dice_iou(preds[i], masks[i])[0] for i in np.flatnonzero(anomalous)]
    return {
        "pixel_auroc": auroc(masks, maps),
        "dice": dice,
        "iou": iou,
        "mean_dice_anomalous": float(np.mean(per_image)) if per_image else float("nan"),
    }


_CONDITION = re.compile(r"^\d+_(.+)$")


def condition_of(path: str) -> str:
    """Acquisition condition from an MVTec AD 2 filename, e.g. '003_shift_1.png' -> 'shift_1'."""
    stem = Path(path).stem
    match = _CONDITION.match(stem)
    return match.group(1) if match else "unknown"


def per_condition_image_auroc(labels: np.ndarray, scores: np.ndarray, paths: list[str]) -> dict[str, dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        groups[condition_of(p)].append(i)
    out = {}
    for cond, idx in sorted(groups.items()):
        lab = np.asarray(labels)[idx]
        out[cond] = {
            "image_auroc": auroc(lab, np.asarray(scores)[idx]),
            "n_normal": int((lab == 0).sum()),
            "n_anomalous": int((lab == 1).sum()),
        }
    return out
