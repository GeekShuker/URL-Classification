#!/usr/bin/env python3
"""
Generate reports/plots from artifacts/*/metadata.json.

Run:
  python report.py
Optional:
  python report.py --artifacts artifacts --out reports
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


@dataclass
class ModelRecord:
    name: str
    path: Path
    data: Dict[str, Any]


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_metadata_files(artifacts_dir: Path) -> List[ModelRecord]:
    files = sorted(artifacts_dir.glob("*/metadata.json"))
    records: List[ModelRecord] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            model_name = data.get("model") or fp.parent.name
            records.append(ModelRecord(name=str(model_name), path=fp, data=data))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return records


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)


def save_json(out_path: Path, obj: Any) -> None:
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_bar(values: Dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    names = list(values.keys())
    ys = [values[n] for n in names]

    plt.figure()
    plt.bar(names, ys)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(cm: Dict[str, int], title: str, out_path: Path) -> None:
    # expected keys: tn, fp, fn, tp
    tn = int(cm.get("tn", 0))
    fp = int(cm.get("fp", 0))
    fn = int(cm.get("fn", 0))
    tp = int(cm.get("tp", 0))
    mat = [[tn, fp], [fn, tp]]

    plt.figure()
    plt.imshow(mat, interpolation="nearest")
    plt.title(title)
    plt.xticks([0, 1], ["Pred Benign", "Pred Malicious"], rotation=15, ha="right")
    plt.yticks([0, 1], ["True Benign", "True Malicious"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(mat[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_timing(records: List[ModelRecord], out_path: Path) -> None:
    names = [r.name for r in records]
    feat = [float(safe_get(r.data, ["timing", "feature_extraction_sec"], 0.0)) for r in records]
    train = [float(safe_get(r.data, ["timing", "training_sec"], 0.0)) for r in records]
    total = [float(safe_get(r.data, ["timing", "total_sec"], 0.0)) for r in records]

    plt.figure()
    x = range(len(names))
    plt.bar(x, feat, label="feature_extraction_sec")
    plt.bar(x, train, bottom=feat, label="training_sec")
    bottom2 = [feat[i] + train[i] for i in range(len(names))]
    remainder = [max(0.0, total[i] - bottom2[i]) for i in range(len(names))]
    plt.bar(x, remainder, bottom=bottom2, label="other(total-feat-train)")

    plt.title("Timing breakdown (seconds)")
    plt.ylabel("Seconds")
    plt.xticks(list(x), names, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(top_features: List[List[Any]], title: str, out_path: Path, top_n: int = 20) -> None:
    # top_features is list of [feature_name, score]
    items: List[Tuple[str, float]] = []
    for it in top_features:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            try:
                items.append((str(it[0]), float(it[1])))
            except Exception:
                continue
    items = items[:top_n]
    if not items:
        return

    feats = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    plt.figure()
    plt.barh(feats, vals)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_summary_table(records: List[ModelRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in records:
        tm = safe_get(r.data, ["test_metrics"], {}) or {}
        ta = safe_get(r.data, ["test_auc"], {}) or {}
        cm = safe_get(r.data, ["test_confusion"], {}) or {}
        timing = safe_get(r.data, ["timing"], {}) or {}
        max_fpr = safe_get(r.data, ["max_fpr"], None)
        thresh = safe_get(r.data, ["threshold"], safe_get(tm, ["threshold"], None))

        tn = int(cm.get("tn", 0)) if isinstance(cm, dict) else 0
        fp = int(cm.get("fp", 0)) if isinstance(cm, dict) else 0
        fn = int(cm.get("fn", 0)) if isinstance(cm, dict) else 0
        tp = int(cm.get("tp", 0)) if isinstance(cm, dict) else 0
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else None
        fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else None

        rows.append({
            "model": r.name,
            "roc_auc": float(ta.get("roc_auc", tm.get("auc", float("nan")))),
            "pr_auc": float(ta.get("pr_auc", float("nan"))),
            "accuracy": float(tm.get("accuracy", float("nan"))),
            "f1": float(tm.get("f1", float("nan"))),
            "recall": float(tm.get("recall", float("nan"))),
            "precision": float(tm.get("precision", float("nan"))),
            "fpr": float(tm.get("fpr", fpr_rate if fpr_rate is not None else float("nan"))),
            "threshold": float(thresh) if thresh is not None else float("nan"),
            "max_fpr_policy": float(max_fpr) if max_fpr is not None else float("nan"),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fp_rate": fpr_rate,
            "fn_rate": fnr_rate,
            "feature_extraction_sec": float(timing.get("feature_extraction_sec", float("nan"))),
            "training_sec": float(timing.get("training_sec", float("nan"))),
            "total_sec": float(timing.get("total_sec", float("nan"))),
            "metadata_path": str(r.path.as_posix()),
        })
    return rows


def write_tables(summary_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    tables_dir = out_dir / "tables"
    if pd is not None:
        df = pd.DataFrame(summary_rows)
        df_sorted = df.sort_values(by=["roc_auc", "pr_auc"], ascending=False)
        df_sorted.to_csv(tables_dir / "model_comparison.csv", index=False)

        # Markdown table (compact)
        cols = ["model", "roc_auc", "pr_auc", "f1", "accuracy", "recall", "precision", "fpr", "threshold", "total_sec"]
        md = df_sorted[cols].to_markdown(index=False)
        (tables_dir / "model_comparison.md").write_text(md, encoding="utf-8")
    else:
        # Fallback: JSON only
        save_json(tables_dir / "model_comparison.json", summary_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory (default: artifacts)")
    ap.add_argument("--out", type=str, default="reports", help="Output reports dir (default: reports)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts)
    out_dir = Path(args.out)

    ensure_out(out_dir)

    records = load_metadata_files(artifacts_dir)
    if not records:
        raise SystemExit(f"No metadata.json found under: {artifacts_dir}/<model>/metadata.json")

    # Save a manifest
    manifest = [{"model": r.name, "metadata_path": str(r.path.as_posix())} for r in records]
    save_json(out_dir / "tables" / "manifest.json", manifest)

    # Summary table
    summary_rows = build_summary_table(records)
    write_tables(summary_rows, out_dir)

    # Prepare dicts for bar plots
    def as_map(key: str) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for row in summary_rows:
            try:
                m[str(row["model"])] = float(row.get(key, float("nan")))
            except Exception:
                m[str(row["model"])] = float("nan")
        return m

    plots_dir = out_dir / "plots"

    # Core metric plots
    plot_bar(as_map("roc_auc"), "ROC-AUC by model", "ROC-AUC", plots_dir / "roc_auc.png")
    plot_bar(as_map("pr_auc"), "PR-AUC by model", "PR-AUC", plots_dir / "pr_auc.png")
    plot_bar(as_map("f1"), "F1 by model", "F1", plots_dir / "f1.png")
    plot_bar(as_map("accuracy"), "Accuracy by model", "Accuracy", plots_dir / "accuracy.png")
    plot_bar(as_map("recall"), "Recall by model", "Recall", plots_dir / "recall.png")
    plot_bar(as_map("precision"), "Precision by model", "Precision", plots_dir / "precision.png")
    plot_bar(as_map("fpr"), "False Positive Rate (FPR) by model", "FPR", plots_dir / "fpr.png")
    plot_bar(as_map("threshold"), "Selected threshold by model (policy-driven)", "Threshold", plots_dir / "threshold.png")

    # Error-count plots (FN/FP)
    fp_counts: Dict[str, float] = {row["model"]: float(row["fp"]) for row in summary_rows}
    fn_counts: Dict[str, float] = {row["model"]: float(row["fn"]) for row in summary_rows}
    plot_bar(fp_counts, "False Positives (FP) count by model", "FP count", plots_dir / "fp_count.png")
    plot_bar(fn_counts, "False Negatives (FN) count by model", "FN count", plots_dir / "fn_count.png")

    # Error-rate plots (FP rate / FN rate)
    fp_rate: Dict[str, float] = {}
    fn_rate: Dict[str, float] = {}
    for row in summary_rows:
        m = str(row["model"])
        fp_rate[m] = float(row["fp_rate"]) if row["fp_rate"] is not None else float("nan")
        fn_rate[m] = float(row["fn_rate"]) if row["fn_rate"] is not None else float("nan")
    plot_bar(fp_rate, "FP rate (FP/(FP+TN)) by model", "FP rate", plots_dir / "fp_rate.png")
    plot_bar(fn_rate, "FN rate (FN/(FN+TP)) by model", "FN rate", plots_dir / "fn_rate.png")

    # Confusion matrices
    for r in records:
        cm = safe_get(r.data, ["test_confusion"], None)
        if isinstance(cm, dict):
            plot_confusion(cm, f"Confusion matrix (test) - {r.name}", plots_dir / f"confusion_{r.name}.png")

    # Timing breakdown
    plot_timing(records, plots_dir / "timing_breakdown.png")

    # Security-policy view: Recall under max_fpr policy
    # (your threshold is chosen to respect max_fpr on validation; still useful to present)
    policy = {}
    for row in summary_rows:
        policy[str(row["model"])] = float(row.get("recall", float("nan")))
    plot_bar(policy, "Recall under policy (threshold chosen with max_fpr constraint)", "Recall", plots_dir / "recall_under_policy.png")

    # Feature importance (if available)
    for r in records:
        top_feats = safe_get(r.data, ["top_features"], None)
        if isinstance(top_feats, list) and top_feats:
            plot_feature_importance(
                top_feats,
                title=f"Top features - {r.name}",
                out_path=plots_dir / f"top_features_{r.name}.png",
                top_n=20
            )

    # Save a short README in reports/
    readme = [
        "# Reports output",
        "",
        "## Tables",
        "- tables/model_comparison.csv (and .md if pandas installed)",
        "- tables/manifest.json",
        "",
        "## Plots (reports/plots)",
        "- roc_auc.png, pr_auc.png",
        "- f1.png, accuracy.png, recall.png, precision.png, fpr.png, threshold.png",
        "- fp_count.png, fn_count.png, fp_rate.png, fn_rate.png",
        "- confusion_<model>.png",
        "- timing_breakdown.png",
        "- recall_under_policy.png",
        "- top_features_<model>.png (only if metadata includes top_features)",
        "",
        "Note: ROC/PR curves require per-sample scores; this script plots AUC summary values from metadata.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"[OK] Wrote reports to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
