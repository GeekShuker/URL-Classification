#!/usr/bin/env python
"""Predict phishing probabilities for URLs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from phishdet.features import build_feature_df
from phishdet.persistence import load_artifacts


LABELS = {0: "benign", 1: "malicious"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict phishing probabilities")
    parser.add_argument("--modeldir", required=True, help="Path to saved model directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single URL string")
    group.add_argument("--input", help="Path to text file with one URL per line")
    return parser.parse_args()


def load_urls(args) -> List[str]:
    if args.url is not None:
        return [args.url]
    input_path = Path(args.input)
    return [line.strip() for line in input_path.read_text().splitlines() if line.strip()]


def main():
    args = parse_args()
    model, feature_cols, threshold, metadata = load_artifacts(args.modeldir)
    urls = load_urls(args)
    features = build_feature_df(urls)[feature_cols]
    proba = model.predict_proba(features)[:, 1]

    for url, p in zip(urls, proba):
        pred_int = int(p >= threshold)
        output = {
            "url": url,
            "proba": float(p),
            "threshold": float(threshold),
            "predicted_int": pred_int,
            "predicted_label": LABELS.get(pred_int, "unknown"),
        }
        print(json.dumps(output))


if __name__ == "__main__":
    main()
