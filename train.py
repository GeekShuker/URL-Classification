#!/usr/bin/env python
"""Training script for phishing URL detection."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from phishdet.features import build_feature_df
from phishdet.models import build_en_bag, build_en_knn, build_rfe_xgb
from phishdet.thresholding import find_threshold_max_recall_under_fpr
from phishdet.metrics import compute_confusion, compute_metrics
from phishdet.persistence import save_artifacts
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_BUILDERS = {
    "en_bag": build_en_bag,
    "en_knn": build_en_knn,
    "rfe_xgb": build_rfe_xgb,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--data", default="balanced_urls.csv", help="Path to CSV dataset")
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--benign-value", type=int, default=0, help="Numeric value representing benign class")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional row cap for quick smoke tests; 0 uses full dataset",
    )
    parser.add_argument("--model", choices=list(MODEL_BUILDERS.keys()), default="en_bag")
    parser.add_argument("--max-fpr", type=float, default=0.01, help="Maximum allowed FPR for threshold selection")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", default=None, help="Directory to save artifacts")

    # Bagging params
    parser.add_argument("--bag-n-estimators", type=int, default=200)
    parser.add_argument("--bag-max-depth", type=int, default=8)
    parser.add_argument("--bag-min-samples-leaf", type=int, default=1)

    # kNN ensemble
    parser.add_argument("--knn-ks", default="3,5,7,9", help="Comma-separated k values")

    # RFE + XGB
    parser.add_argument("--rfe-k", type=int, default=30)
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)

    return parser.parse_args()


def build_model(args):
    common = {"random_state": args.random_state}
    if args.model == "en_bag":
        return build_en_bag(
            **common,
            n_estimators=args.bag_n_estimators,
            max_depth=args.bag_max_depth if args.bag_max_depth > 0 else None,
            min_samples_leaf=args.bag_min_samples_leaf,
        )
    if args.model == "en_knn":
        ks = [int(k.strip()) for k in args.knn_ks.split(",") if k.strip()]
        return build_en_knn(**common, ks=ks)
    if args.model == "rfe_xgb":
        return build_rfe_xgb(
            **common,
            rfe_k=args.rfe_k,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
        )
    raise ValueError(f"Unknown model {args.model}")


def main():
    total_start = time.perf_counter()
    args = parse_args()
    outdir = args.outdir or f"artifacts/{args.model}"
    data_path = Path(args.data)

    df = pd.read_csv(data_path, usecols=[args.url_col, args.label_col])
    if args.sample_size and args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)
    urls = df[args.url_col].astype(str).tolist()
    labels = (df[args.label_col].astype(int) != int(args.benign_value)).astype(int).values

    feature_start = time.perf_counter()
    features = build_feature_df(urls)
    feature_time = time.perf_counter() - feature_start
    feature_columns = list(features.columns)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train_val,
    )

    model = build_model(args)
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - training_start

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, val_metrics = find_threshold_max_recall_under_fpr(
        y_val, val_proba, max_fpr=args.max_fpr
    )

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    test_preds = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion(y_test, test_preds)
    try:
        roc_auc = float(roc_auc_score(y_test, test_proba))
    except Exception:
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(y_test, test_proba))
    except Exception:
        pr_auc = 0.0

    metadata_top_features = None
    if args.model == "rfe_xgb" and hasattr(model, "named_steps"):
        rfe_step = model.named_steps.get("rfe")
        xgb_step = model.named_steps.get("xgb")
        if rfe_step is not None and xgb_step is not None:
            support = getattr(rfe_step, "support_", None)
            importances = getattr(xgb_step, "feature_importances_", None)
            if support is not None and importances is not None:
                selected_features = [
                    feature_columns[i] for i, flag in enumerate(support) if flag
                ]
                pairs = list(zip(selected_features, importances))
                pairs.sort(key=lambda x: x[1], reverse=True)
                metadata_top_features = [
                    [name, float(score)] for name, score in pairs[:20]
                ]

    total_time = time.perf_counter() - total_start

    metadata = {
        "model": args.model,
        "params": vars(args),
        "max_fpr": args.max_fpr,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_columns": feature_columns,
        "splits": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "sample_size": args.sample_size,
        "n_samples": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "test_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "test_auc": {"roc_auc": roc_auc, "pr_auc": pr_auc},
        "random_state": args.random_state,
        "timing": {
            "feature_extraction_sec": float(feature_time),
            "training_sec": float(training_time),
            "total_sec": float(total_time),
        },
    }

    if metadata_top_features:
        metadata["top_features"] = metadata_top_features

    save_artifacts(
        outdir,
        model,
        feature_columns=feature_columns,
        threshold=threshold,
        metadata_dict=metadata,
    )

    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics, "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
