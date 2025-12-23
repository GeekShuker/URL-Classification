#!/usr/bin/env python
"""Evaluate a saved model on labeled data."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from phishdet.features import build_feature_df
from phishdet.metrics import compute_confusion, compute_metrics
from phishdet.persistence import load_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phishing model")
    parser.add_argument("--modeldir", required=True, help="Directory with saved artifacts")
    parser.add_argument("--data", default="balanced_urls.csv", help="Path to CSV dataset")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--benign-value", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model, feature_cols, threshold, metadata = load_artifacts(args.modeldir)
    df = pd.read_csv(args.data, usecols=[args.url_col, args.label_col])
    urls = df[args.url_col].astype(str).tolist()
    labels = (df[args.label_col].astype(int) != int(args.benign_value)).astype(int).values

    features = build_feature_df(urls)[feature_cols]
    proba = model.predict_proba(features)[:, 1]
    metrics = compute_metrics(labels, proba, threshold)
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion(labels, preds)
    output = {
        "threshold": threshold,
        "metrics": metrics,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
