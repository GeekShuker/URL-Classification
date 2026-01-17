import os
import requests
import streamlit as st
import pandas as pd

API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="URL Classification UI", layout="wide")
st.title("URL Classification System UI")

# -------------------------
# Predict URL section
# -------------------------
st.header("Predict URL")

models = []
try:
    r = requests.get(f"{API_BASE_URL}/models", timeout=10)
    r.raise_for_status()
    models = r.json().get("models", [])
except Exception:
    models = []

model = st.selectbox("Model", options=models or ["rfe_xgb", "en_bag", "en_knn"])
url = st.text_input("URL", placeholder="https://example.com/login")
threshold_str = st.text_input("Threshold (optional)", value="")

if st.button("Predict"):
    if not url.strip():
        st.error("Please enter a URL")
    else:
        payload = {"url": url.strip(), "model": model}
        if threshold_str.strip():
            try:
                payload["threshold"] = float(threshold_str.strip())
            except ValueError:
                st.error("Threshold must be a number (e.g. 0.7775)")
                payload = None

        if payload:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} - {resp.text}")
            else:
                out = resp.json()
                st.success(f"Prediction: {out.get('predicted_label', 'N/A')}")
                st.json(out)

st.divider()

# -------------------------
# Async (queue-backed) URL classification
# -------------------------
st.header("Async URL Classification (Queue + Workers)")

st.caption("Designed for hundreds of concurrent URL requests. The API enqueues jobs and workers process them in parallel.")

async_model = st.selectbox("Model (async)", options=models or ["rfe_xgb", "en_bag", "en_knn"], key="async_model")
async_threshold_str = st.text_input("Threshold (optional) (async)", value="", key="async_thr")

st.subheader("Submit a single URL (async)")
async_url = st.text_input("URL (async)", placeholder="https://example.com/login", key="async_url")
if st.button("Enqueue single URL"):
    if not async_url.strip():
        st.error("Please enter a URL")
    else:
        payload = {"url": async_url.strip(), "model": async_model}
        if async_threshold_str.strip():
            try:
                payload["threshold"] = float(async_threshold_str.strip())
            except ValueError:
                st.error("Threshold must be a number")
                payload = None
        if payload:
            r = requests.post(f"{API_BASE_URL}/jobs/classify", json=payload, timeout=30)
            if r.status_code != 200:
                st.error(f"API error: {r.status_code} - {r.text}")
            else:
                st.success(r.json())

st.subheader("Submit multiple URLs (async batch)")
urls_text = st.text_area(
    "Paste one URL per line",
    height=150,
    placeholder="https://a.com\nhttps://b.com\nhttps://c.com",
    key="batch_urls",
)

if st.button("Enqueue batch"):
    urls = [ln.strip() for ln in urls_text.splitlines() if ln.strip()]
    if not urls:
        st.error("Please paste at least one URL")
    else:
        payload = {"urls": urls, "model": async_model}
        if async_threshold_str.strip():
            try:
                payload["threshold"] = float(async_threshold_str.strip())
            except ValueError:
                st.error("Threshold must be a number")
                payload = None
        if payload:
            r = requests.post(f"{API_BASE_URL}/jobs/classify_batch", json=payload, timeout=60)
            if r.status_code != 200:
                st.error(f"API error: {r.status_code} - {r.text}")
            else:
                out = r.json()
                st.success({"submitted": out.get("submitted"), "first_job_id": (out.get("job_ids") or [None])[0]})
                st.write("job_ids:")
                st.code("\n".join(out.get("job_ids") or []))

# -------------------------
# Jobs section (run_all + logs)
# -------------------------
st.header("Pipeline Jobs")

if st.button("Enqueue run_all"):
    r = requests.post(f"{API_BASE_URL}/jobs/run_all", timeout=60)
    if r.status_code != 200:
        st.error(f"API error: {r.status_code} - {r.text}")
    else:
        st.success(r.json())

st.subheader("Jobs")
jobs = requests.get(f"{API_BASE_URL}/jobs", timeout=30).json()
if jobs:
    st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)
    job_id = st.selectbox("job_id", [j["job_id"] for j in jobs])
    if job_id and st.button("Show logs"):
        logs = requests.get(f"{API_BASE_URL}/jobs/{job_id}/logs", timeout=60).text
        st.text(logs)
else:
    st.info("No jobs yet.")
