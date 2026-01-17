#!/usr/bin/env python
"""Training script for phishing URL detection."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from phishdet.features import build_feature_df
from phishdet.models import build_en_bag, build_en_knn, build_rfe_xgb, build_rf, build_xgb, build_catboost
from phishdet.thresholding import find_threshold_max_recall_under_fpr
from phishdet.metrics import compute_confusion, compute_metrics
from phishdet.persistence import save_artifacts


MODEL_BUILDERS = {
    "en_bag": build_en_bag,
    "en_knn": build_en_knn,
    "rfe_xgb": build_rfe_xgb,
    "rf": build_rf,
    "xgb": build_xgb,
    "catboost": build_catboost,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--data", default="balanced_urls.csv", help="Path to CSV dataset")
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--benign-value", type=int, default=0, help="Numeric value representing benign class")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional row cap for quick smoke tests; 0 uses full dataset")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS.keys()), default="en_bag")
    parser.add_argument("--max-fpr", type=float, default=0.01, help="Maximum allowed FPR for threshold selection")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="If set, use this threshold for val/test metrics instead of auto-selection on validation.",
    )

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
    parser.add_argument("--rfe-k", type=int, default=24)
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)

    # Plain XGBoost (without RFE)
    parser.add_argument("--plain-xgb-n-estimators", type=int, default=200)
    parser.add_argument("--plain-xgb-max-depth", type=int, default=6)
    parser.add_argument("--plain-xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--plain-xgb-subsample", type=float, default=0.8)
    parser.add_argument("--plain-xgb-colsample-bytree", type=float, default=0.8)

    # CatBoost
    parser.add_argument("--catboost-iterations", type=int, default=200)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.1)
    parser.add_argument("--catboost-subsample", type=float, default=0.8)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)

    # Random Forest
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="0 means None (unlimited)")
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", default="sqrt", help="sqrt, log2, int, or float")

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
    if args.model == "rf":
        # Parse max_features: can be "sqrt", "log2", int, or float
        max_features = args.rf_max_features
        if max_features not in ("sqrt", "log2"):
            try:
                # Try to parse as float first
                max_features = float(max_features)
                # If it's a whole number, convert to int
                if max_features.is_integer():
                    max_features = int(max_features)
            except (ValueError, AttributeError):
                # If parsing fails, keep as string (might be invalid, but let sklearn handle it)
                pass
        return build_rf(
            **common,
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=max_features,
        )
    if args.model == "xgb":
        return build_xgb(
            **common,
            n_estimators=args.plain_xgb_n_estimators,
            max_depth=args.plain_xgb_max_depth,
            learning_rate=args.plain_xgb_learning_rate,
            subsample=args.plain_xgb_subsample,
            colsample_bytree=args.plain_xgb_colsample_bytree,
        )
    if args.model == "catboost":
        return build_catboost(
            **common,
            iterations=args.catboost_iterations,
            depth=args.catboost_depth,
            learning_rate=args.catboost_learning_rate,
            subsample=args.catboost_subsample,
            l2_leaf_reg=args.catboost_l2_leaf_reg,
        )
    raise ValueError(f"Unknown model {args.model}")


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def _safe_pr_auc(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def main():
    total_start = time.perf_counter()
    args = parse_args()

    # Validate fixed threshold if provided
    if args.fixed_threshold is not None:
        if not (0.0 <= args.fixed_threshold <= 1.0):
            raise ValueError("--fixed-threshold must be in [0, 1]")


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

    if args.fixed_threshold is not None:
        threshold = float(args.fixed_threshold)
        val_metrics = compute_metrics(y_val, val_proba, threshold)
    else:
        threshold, val_metrics = find_threshold_max_recall_under_fpr(
            y_val, val_proba, max_fpr=args.max_fpr
        )

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    test_preds = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion(y_test, test_preds)

    roc_auc = _safe_auc(y_test, test_proba)
    pr_auc = _safe_pr_auc(y_test, test_proba)

    # RFE metadata: selected features + aligned top features
    selected_features = None
    metadata_top_features = None

    if args.model == "rfe_xgb" and hasattr(model, "named_steps"):
        rfe_step = model.named_steps.get("rfe")
        xgb_step = model.named_steps.get("xgb")
        if rfe_step is not None and xgb_step is not None:
            support = getattr(rfe_step, "support_", None)
            importances = getattr(xgb_step, "feature_importances_", None)

            if support is not None:
                selected_features = [feature_columns[i] for i, flag in enumerate(support) if flag]

            # xgb importances correspond to the post-RFE feature space (len == len(selected_features))
            if selected_features and importances is not None and len(importances) == len(selected_features):
                pairs = list(zip(selected_features, importances))
                pairs.sort(key=lambda x: float(x[1]), reverse=True)
                metadata_top_features = [[name, float(score)] for name, score in pairs[:20]]

    # Random Forest feature importance
    if args.model == "rf" and hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_columns):
            pairs = list(zip(feature_columns, importances))
            pairs.sort(key=lambda x: float(x[1]), reverse=True)
            metadata_top_features = [[name, float(score)] for name, score in pairs[:20]]

    # Plain XGBoost feature importance
    if args.model == "xgb" and hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_columns):
            pairs = list(zip(feature_columns, importances))
            pairs.sort(key=lambda x: float(x[1]), reverse=True)
            metadata_top_features = [[name, float(score)] for name, score in pairs[:20]]

    # CatBoost feature importance
    if args.model == "catboost" and hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_columns):
            pairs = list(zip(feature_columns, importances))
            pairs.sort(key=lambda x: float(x[1]), reverse=True)
            metadata_top_features = [[name, float(score)] for name, score in pairs[:20]]

    total_time = time.perf_counter() - total_start

    metadata = {
        "model": args.model,
        "params": vars(args),
        "max_fpr": args.max_fpr,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "selected_features": selected_features,  # None unless rfe_xgb
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

    out = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_auc": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "threshold": float(threshold),
    }
    print(json.dumps(out, indent=2))
    print("\nConfusion Matrix (TEST) (Actual x Pred)")
    print(f"{'':>12} {'Pred 0':>10} {'Pred 1':>10}")
    print(f"{'Actual 0':>12} {int(tn):>10} {int(fp):>10}")
    print(f"{'Actual 1':>12} {int(fn):>10} {int(tp):>10}")


if __name__ == "__main__":
    main()
