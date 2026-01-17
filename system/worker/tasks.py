import json, os, shlex, subprocess
from pathlib import Path

from phishdet.persistence import load_artifacts
from phishdet.features import build_feature_df
from system.api.database import SessionLocal
from system.api.models import Job

LABELS = {0: "benign", 1: "malicious"}

# Simple in-process cache per worker process to avoid re-loading artifacts for each URL
_MODEL_CACHE = {}  # model_name -> (model, feature_cols, default_thr)


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    modeldir = Path("/app") / "artifacts" / model_name
    if not modeldir.exists():
        raise FileNotFoundError(f"Model not found: {modeldir}")

    model, feature_cols, default_thr, _meta = load_artifacts(modeldir)
    _MODEL_CACHE[model_name] = (model, feature_cols, float(default_thr))
    return _MODEL_CACHE[model_name]

def _log(job: Job, text: str):
    full = os.path.join("/app", job.log_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def _set_status(db, job: Job, status: str, *, exit_code=0, error_message="", result=None):
    job.status = status
    job.exit_code = int(exit_code)
    job.error_message = error_message or ""
    if result is not None:
        job.result_json = json.dumps(result, ensure_ascii=False)
    db.commit()
    db.refresh(job)

def _run(job: Job, cmd: list[str]):
    _log(job, "==> Running: " + " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.Popen(cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        _log(job, line.rstrip("\n"))
    rc = p.wait()
    _log(job, f"==> Exit code: {rc}")
    return rc

def run_all_pipeline(job_id: str, payload: dict):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        _set_status(db, job, "started", result={"message":"started"})

        # Example: call your existing run_all.sh in repo root
        rc = _run(job, ["bash", "./run_all.sh"])

        if rc != 0:
            _set_status(db, job, "failed", exit_code=rc, error_message="run_all.sh failed")
            return

        _set_status(db, job, "finished", result={"message":"finished"})
    finally:
        db.close()


def classify_url(job_id: str, payload: dict):
    """Classify a single URL.

    This is designed for high load: the API enqueues one task per URL and returns
    immediately. Scaling is done by running multiple worker containers.
    """
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return

        _set_status(db, job, "started", result={"message": "started"})

        url = (payload.get("url") or "").strip()
        model_name = payload.get("model") or "rfe_xgb"
        thr_override = payload.get("threshold", None)

        if not url:
            _set_status(db, job, "failed", exit_code=1, error_message="Empty URL", result={"message": "failed"})
            return

        try:
            model, feature_cols, default_thr = _get_model(model_name)
            df = build_feature_df([url])
            X = df.reindex(columns=feature_cols, fill_value=0)
            proba = float(model.predict_proba(X)[:, 1][0])

            thr = float(thr_override) if thr_override is not None else float(default_thr)
            pred = int(proba >= thr)

            result = {
                "url": url,
                "model": model_name,
                "proba": proba,
                "threshold": thr,
                "predicted_int": pred,
                "predicted_label": LABELS.get(pred, "unknown"),
            }
            _set_status(db, job, "finished", result=result)
        except Exception as e:
            _log(job, f"ERROR: {e}")
            _set_status(db, job, "failed", exit_code=1, error_message=str(e), result={"message": "failed"})
    finally:
        db.close()
