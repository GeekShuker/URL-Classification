"""Threshold selection utilities."""
from __future__ import annotations

import numpy as np

from .metrics import compute_confusion


def find_threshold_max_recall_under_fpr(
    y_true: np.ndarray, y_proba: np.ndarray, max_fpr: float = 0.01
):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    # Candidate thresholds include unique probabilities plus boundaries
    unique_scores = np.unique(y_proba)
    candidates = np.concatenate(
        [unique_scores, np.linspace(0, 1, num=500, endpoint=True)]
    )
    candidates = np.clip(np.unique(candidates), 0.0, 1.0)

    best = None
    for thresh in candidates:
        preds = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = compute_confusion(y_true, preds)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr > max_fpr:
            continue
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        candidate = (recall, precision, thresh, fpr, tp, fp, fn, tn)
        if best is None:
            best = candidate
            continue
        if recall > best[0]:
            best = candidate
        elif recall == best[0]:
            if precision > best[1]:
                best = candidate
            elif precision == best[1] and thresh > best[2]:
                best = candidate
    if best is None:
        # No threshold satisfies constraint; pick highest threshold to minimize FPR
        best_thresh = float(np.max(candidates))
        preds = (y_proba >= best_thresh).astype(int)
        tn, fp, fn, tp = compute_confusion(y_true, preds)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        metrics = {
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "threshold": best_thresh,
        }
        return best_thresh, metrics

    recall, precision, thresh, fpr, tp, fp, fn, tn = best
    metrics = {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": thresh,
    }
    return float(thresh), metrics
