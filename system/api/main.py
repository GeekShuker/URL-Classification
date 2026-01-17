# system/api/main.py

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Any, List

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from phishdet.persistence import load_artifacts
from phishdet.features import build_feature_df

from .config import settings
from .database import engine, SessionLocal, Base
from .models import Job
from .queue import get_queue

# -----------------------------
# App + DB init
# -----------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI(title="URL Classification System API", version="1.0.0")

LABELS = {0: "benign", 1: "malicious"}


# -----------------------------
# DB dependency
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Jobs helpers
# -----------------------------
def create_job(db: Session, job_type: str, request_obj: dict) -> Job:
    import uuid
    import datetime as dt

    job_id = str(uuid.uuid4())
    log_path = os.path.join("reports", "system_logs", f"{job_id}.log")

    j = Job(
        id=job_id,
        type=job_type,
        status="queued",
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
        request_json=json.dumps(request_obj, ensure_ascii=False),
        result_json=json.dumps({"message": "queued"}, ensure_ascii=False),
        log_path=log_path,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction endpoints
# -----------------------------
class PredictRequest(BaseModel):
    url: str
    model: str = "rfe_xgb"          # artifacts/<model>
    threshold: Optional[float] = None


class ClassifyAsyncRequest(BaseModel):
    url: str
    model: str = "rfe_xgb"          # artifacts/<model>
    threshold: Optional[float] = None


class ClassifyBatchRequest(BaseModel):
    urls: List[str]
    model: str = "rfe_xgb"          # artifacts/<model>
    threshold: Optional[float] = None


@app.get("/models")
def list_models():
    base = Path(settings.app_workdir) / "artifacts"
    if not base.exists():
        return {"models": []}

    models: List[str] = []
    for d in base.iterdir():
        if d.is_dir() and (d / "model.joblib").exists():
            models.append(d.name)

    return {"models": sorted(models)}


@app.post("/predict")
def predict(req: PredictRequest):
    modeldir = Path(settings.app_workdir) / "artifacts" / req.model
    if not modeldir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {modeldir}")

    # expected: model, feature_cols, default_thr, meta
    model, feature_cols, default_thr, _meta = load_artifacts(modeldir)

    df = build_feature_df([req.url])
    X = df.reindex(columns=feature_cols, fill_value=0)

    try:
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    thr = float(req.threshold) if req.threshold is not None else float(default_thr)
    pred = int(proba >= thr)

    return {
        "url": req.url,
        "model": req.model,
        "proba": proba,
        "threshold": thr,
        "predicted_int": pred,
        "predicted_label": LABELS.get(pred, "unknown"),
    }


# -----------------------------
# Jobs endpoints
# -----------------------------
@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db), limit: int = 200):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    out = []
    for j in jobs:
        out.append(
            {
                "job_id": j.id,
                "type": j.type,
                "status": j.status,
                "created_at": j.created_at.isoformat(),
                "updated_at": j.updated_at.isoformat(),
                "log_path": j.log_path,
                "exit_code": j.exit_code,
                "error_message": j.error_message,
                "request": json.loads(j.request_json or "{}"),
                "result": json.loads(j.result_json or "{}"),
            }
        )
    return out


@app.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": j.id,
        "type": j.type,
        "status": j.status,
        "created_at": j.created_at.isoformat(),
        "updated_at": j.updated_at.isoformat(),
        "log_path": j.log_path,
        "exit_code": j.exit_code,
        "error_message": j.error_message,
        "request": json.loads(j.request_json or "{}"),
        "result": json.loads(j.result_json or "{}"),
    }


@app.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_job_logs(job_id: str, db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "Job not found")

    if not j.log_path:
        return ""

    full = os.path.join(settings.app_workdir, j.log_path)
    if not os.path.exists(full):
        raise HTTPException(404, "Log file not found")

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


@app.post("/jobs/classify")
def enqueue_classify(req: ClassifyAsyncRequest, db: Session = Depends(get_db)):
    """Async URL classification.

    Returns immediately with job_id. The actual inference happens in the worker.
    """
    os.makedirs(os.path.join(settings.app_workdir, "reports", "system_logs"), exist_ok=True)

    payload = req.model_dump()
    job = create_job(db, "classify_url", payload)
    q = get_queue()
    q.enqueue("system.worker.tasks.classify_url", job.id, payload)

    return {"job_id": job.id, "status": "queued"}


@app.post("/jobs/classify_batch")
def enqueue_classify_batch(req: ClassifyBatchRequest, db: Session = Depends(get_db)):
    """Async batch URL classification.

    Submits one queue task per URL (simple + easy to scale via worker replicas).
    """
    os.makedirs(os.path.join(settings.app_workdir, "reports", "system_logs"), exist_ok=True)

    q = get_queue()
    job_ids: List[str] = []

    for url in req.urls:
        u = (url or "").strip()
        if not u:
            continue
        payload = {"url": u, "model": req.model, "threshold": req.threshold}
        job = create_job(db, "classify_url", payload)
        q.enqueue("system.worker.tasks.classify_url", job.id, payload)
        job_ids.append(job.id)

    return {"submitted": len(job_ids), "job_ids": job_ids}


@app.post("/jobs/run_all")
def enqueue_run_all(db: Session = Depends(get_db)):
    os.makedirs(os.path.join(settings.app_workdir, "reports", "system_logs"), exist_ok=True)

    job = create_job(db, "run_all", {})
    q = get_queue()

    # worker function path + args
    q.enqueue("system.worker.tasks.run_all_pipeline", job.id, {})

    return {"job_id": job.id, "status": "queued"}
