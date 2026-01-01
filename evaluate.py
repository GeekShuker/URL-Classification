#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from phishdet.features import build_feature_df
from phishdet.persistence import load_artifacts


def append_run_to_csv(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "model",
        "modeldir",
        "dataset",
        "split",
        "n_samples",
        "threshold",
        "recall",
        "precision",
        "accuracy",
        "fpr",
        "auc",
        "tp",
        "fp",
        "tn",
        "fn",
        "git_commit",
        "notes",
    ]

    for k in fieldnames:
        row.setdefault(k, "")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def model_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X)).astype(float)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = ""
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = ""

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained phishing URL detector.")
    p.add_argument("--modeldir", required=True, help="Path to trained model artifacts directory, e.g., artifacts/en_bag")
    p.add_argument("--data", required=True, help="CSV file with URLs and labels")
    p.add_argument("--url-col", default="url", help="Column name for URL")
    p.add_argument("--label-col", default="label", help="Column name for label (0/1)")
    p.add_argument("--threshold", type=float, default=None, help="Override threshold (otherwise use saved threshold)")

    # NEW:
    p.add_argument("--sample-size", type=int, default=0, help="Evaluate on a random subset of N rows; 0 = full file")
    p.add_argument("--random-state", type=int, default=42, help="Seed for subset sampling")

    p.add_argument("--out-csv", default=None, help="Append evaluation results to a CSV file (append-only)")
    p.add_argument("--split", default="external", help="Label for evaluation split (external/internal/etc.)")
    p.add_argument("--notes", default="", help="Optional notes to store in the CSV")
    p.add_argument("--git-commit", default="", help="Optional git commit hash to store in the CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.data)
    if args.url_col not in df.columns:
        raise KeyError(f"Missing url column '{args.url_col}' in {args.data}. Columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"Missing label column '{args.label_col}' in {args.data}. Columns: {list(df.columns)}")

    # NEW: subset sampling
    if args.sample_size and args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)

    urls = df[args.url_col].astype(str).tolist()
    y_true = df[args.label_col].astype(int).to_numpy()

    model, feature_cols, saved_threshold, _metadata = load_artifacts(args.modeldir)

    threshold = args.threshold if args.threshold is not None else saved_threshold

    feats_df = build_feature_df(urls)
    X = feats_df[feature_cols]
    y_prob = model_predict_proba(model, X)

    metrics = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))

    print("====================================")
    print("Evaluation")
    print("====================================")
    print(f"modeldir   : {args.modeldir}")
    print(f"data       : {args.data}")
    print(f"split      : {args.split}")
    print(f"n_samples  : {len(df)}")
    print(f"threshold  : {threshold}")
    print("------------------------------------")
    print(f"Recall     : {metrics['recall']:.6f}")
    print(f"Precision  : {metrics['precision']:.6f}")
    print(f"Accuracy   : {metrics['accuracy']:.6f}")
    print(f"FPR        : {metrics['fpr']:.6f}")
    print(f"AUC        : {metrics['auc']}")
    print("------------------------------------")
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    print("Confusion Matrix (Actual x Pred)")
    print(f"{'':>12} {'Pred 0':>10} {'Pred 1':>10}")
    print(f"{'Actual 0':>12} {tn:>10} {fp:>10}")
    print(f"{'Actual 1':>12} {fn:>10} {tp:>10}")
    print("====================================")

    if args.out_csv:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": os.path.basename(os.path.normpath(args.modeldir)),
            "modeldir": args.modeldir,
            "dataset": os.path.basename(args.data),
            "split": args.split,
            "n_samples": int(len(df)),
            "threshold": float(threshold),
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"],
            "fpr": metrics["fpr"],
            "auc": metrics["auc"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "git_commit": args.git_commit,
            "notes": args.notes,
        }
        append_run_to_csv(args.out_csv, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
