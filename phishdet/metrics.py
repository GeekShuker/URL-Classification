"""Metric helpers for phishing detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    metrics = {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics["auc"] = 0.0
    return metrics
