#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from phishdet.persistence import load_artifacts
from phishdet.features import build_feature_df


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-s))
    return np.asarray(model.predict(X)).astype(float)


def best_threshold_under_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float) -> dict:
    order = np.argsort(-p)
    p_sorted = p[order]
    y_sorted = y[order]

    P = int(y.sum())
    N = int((1 - y).sum())
    if P == 0 or N == 0:
        raise SystemExit("Calibration set must contain both classes (0 and 1).")

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    recall = tp_cum / P
    fpr = fp_cum / N

    ok = np.where(fpr <= max_fpr)[0]
    if len(ok) == 0:
        best_i = int(np.lexsort((-recall, fpr))[0])
    else:
        best_ok = ok[np.argmax(recall[ok])]
        best_i = int(best_ok)

    thr = float(p_sorted[best_i])

    y_hat = (p >= thr).astype(int)
    tn = int(((y == 0) & (y_hat == 0)).sum())
    fp = int(((y == 0) & (y_hat == 1)).sum())
    fn = int(((y == 1) & (y_hat == 0)).sum())
    tp = int(((y == 1) & (y_hat == 1)).sum())

    return {
        "threshold": thr,
        "max_fpr": float(max_fpr),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeldir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-fpr", type=float, default=0.005)
    ap.add_argument("--print-json", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=[args.url_col, args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)

    model, feature_cols, _thr, _meta = load_artifacts(args.modeldir)

    urls = df[args.url_col].astype(str).tolist()
    y = df[args.label_col].to_numpy(dtype=int)

    X = build_feature_df(urls)[feature_cols]
    p = predict_proba(model, X)

    res = best_threshold_under_fpr(y, p, args.max_fpr)
    if args.print_json:
        print(json.dumps(res, indent=2))
    else:
        print(res["threshold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
