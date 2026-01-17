# URL Classification System – Docker & CLI Guide

This project provides a complete Docker-based system for training, evaluating, and running inference on URL classification models.
It supports both **synchronous prediction** (single URL, immediate result) and **asynchronous batch prediction** (multiple URLs via background jobs and workers).

---

## Architecture Overview

The system is orchestrated with **Docker Compose** and includes the following services:

- **api (FastAPI)**  
  Exposes REST endpoints for prediction, jobs, and pipeline execution.

- **worker (RQ workers)**  
  Executes long-running tasks such as batch classification and pipelines.

- **redis**  
  Message broker / queue backend for RQ.

- **db (PostgreSQL)**  
  Stores job metadata and system state.

- **ui (Streamlit)**  
  Web UI for interactive prediction and system control.

---

## Exposed Ports

- **API**: http://localhost:8000  
- **Swagger / OpenAPI**: http://localhost:8000/docs  
- **UI (Streamlit)**: http://localhost:8501  

---

## Prerequisites

- Docker Desktop
- Docker Compose v2

---

## Environment Setup

From the repository root:

```bash
cp system/.env.example system/.env
```

Default values are sufficient for local usage.

---

## Running the System

From the repository root:

```bash
docker compose -f system/docker-compose.yml up --build
```

Run in detached mode:

```bash
docker compose -f system/docker-compose.yml up --build -d
```

Check running services:

```bash
docker compose -f system/docker-compose.yml ps
```

Stop services:

```bash
docker compose -f system/docker-compose.yml stop
```

Stop and remove containers:

```bash
docker compose -f system/docker-compose.yml down
```

Remove containers and volumes (resets DB):

```bash
docker compose -f system/docker-compose.yml down -v
```

---

## Using the Web UI

Open:

```
http://localhost:8501
```

The UI allows:
- Single URL prediction
- Model selection
- Threshold configuration
- Pipeline execution

---

## Swagger / API Documentation

Open:

```
http://localhost:8000/docs
```

All REST endpoints are documented here.

---

## Prediction Modes

### 1. Synchronous Prediction (Single URL)

Endpoint:
```
POST /predict
```

This behaves exactly like the UI: one request → one immediate result.

#### PowerShell example:

```powershell
$body = '{"url":"https://example.com","model":"rfe_xgb"}'
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" `
  -Body $body
```

Response includes:
- predicted_label
- probability / proba
- threshold
- model

---

### 2. Asynchronous Batch Prediction (Multiple URLs)

Endpoint:
```
POST /jobs/classify_batch
```

This submits multiple URLs to the queue.  
The response **does not contain predictions**, only job IDs.

#### Example many.json

```json
{
  "items": [
    {"url": "https://example.com"},
    {"url": "https://test.com"}
  ]
}
```

#### Submit batch (PowerShell):

```powershell
$body = Get-Content .\many.json -Raw
$submit = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/jobs/classify_batch" `
  -ContentType "application/json" `
  -Body $body

$submit.job_ids
```

---

## Viewing Batch Results from the Terminal

### Poll job results

Each job must be queried individually:

```powershell
Invoke-RestMethod "http://localhost:8000/jobs/<JOB_ID>"
```

The result appears when:
```
status == "finished"
```

and is located under:
```
result
```

---

### Print all batch results (recommended)

```powershell
$ids = $submit.job_ids

foreach ($id in $ids) {
  $j = Invoke-RestMethod "http://localhost:8000/jobs/$id"

  if ($j.status -eq "finished" -and $j.result) {
    "{0}`t{1}`t{2}" -f `
      $j.result.url, `
      $j.result.predicted_label, `
      $j.result.proba
  } else {
    "{0}`tSTATUS={1}" -f $id, $j.status
  }
}
```

---

## Important: Result TTL

Workers log:
```
Result is kept for 500 seconds
```

This means:
- Results are available via `/jobs/<id>` for ~8 minutes
- After that, `result` may be empty even if status is `finished`

---

## Accessing Results After TTL (Logs)

Each job exposes a `log_path`:

```powershell
$j = Invoke-RestMethod "http://localhost:8000/jobs/<JOB_ID>"
$j.log_path
```

Open the log file:

```powershell
notepad $j.log_path
```

or print:

```powershell
Get-Content $j.log_path -Tail 200
```

Logs are stored under:
```
system/reports/system_logs/
```

---

## Queue & Workers Monitoring

### Number of workers

```bash
docker compose -f system/docker-compose.yml ps
```

Example:
```
worker-1 ... worker-8
```
→ 8 workers running concurrently.

---

### Queue length (Redis)

```bash
docker compose exec redis redis-cli LLEN rq:queue:default
```

- `0` → queue empty
- `>0` → pending jobs

---

## When to Use Each Mode

| Use Case | Recommended Endpoint |
|--------|----------------------|
| Single URL, immediate result | `/predict` |
| Large batch, parallel processing | `/jobs/classify_batch` |
| UI-based usage | Streamlit UI |
| Debug / inspection | Swagger (`/docs`) |

---

## Windows / PowerShell Notes

- Prefer `Invoke-RestMethod` over `curl`
- If using curl, always call `curl.exe`
- Avoid line-splitting unless using PowerShell backticks correctly

---

## Summary

- `/predict` → synchronous, immediate output
- `/jobs/classify_batch` → asynchronous, scalable, requires polling
- Results are temporary in API, permanent in logs
- Docker Compose manages all services consistently

---

