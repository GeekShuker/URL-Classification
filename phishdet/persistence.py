"""Persistence helpers for models and metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib


def save_artifacts(outdir: str | Path, model: Any, feature_columns: List[str], threshold: float, metadata_dict: Dict[str, Any]):
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path / "model.joblib")
    (out_path / "feature_columns.json").write_text(json.dumps(feature_columns))
    metadata = dict(metadata_dict)
    metadata["threshold"] = threshold
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_artifacts(modeldir: str | Path) -> Tuple[Any, List[str], float, Dict[str, Any]]:
    modeldir_path = Path(modeldir)
    model = joblib.load(modeldir_path / "model.joblib")
    feature_columns = json.loads((modeldir_path / "feature_columns.json").read_text())
    metadata = json.loads((modeldir_path / "metadata.json").read_text())
    threshold = metadata.get("threshold", 0.5)
    return model, feature_columns, float(threshold), metadata
