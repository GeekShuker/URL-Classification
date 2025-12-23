#!/usr/bin/env bash
set -euo pipefail

NTFY_URL="https://ntfy.sh/VLM-Attacks1790"
DATA="balanced_urls.csv"
URL_COL="url"
LABEL_COL="label"
BENIGN_VALUE="0"

# Prefer curl (git bash / windows), fallback to powershell if not available
notify() {
  local msg="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -s -d "$msg" "$NTFY_URL" >/dev/null || true
  elif command -v curl.exe >/dev/null 2>&1; then
    curl.exe -s -d "$msg" "$NTFY_URL" >/dev/null || true
  elif command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command \
      "try { Invoke-RestMethod -Uri '$NTFY_URL' -Method Post -Body '$msg' | Out-Null } catch { }" \
      >/dev/null 2>&1 || true
  else
    # No notifier available; do nothing
    true
  fi
}

notify_success () { notify "Finish"; }
notify_fail () { notify "FAILED"; }

trap notify_fail ERR

# Use venv python directly (avoid bash activate/uname issues)
PY=".venv/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then
  PY=".venv/Scripts/python"
fi

echo "==> Using python: $PY"

echo "==> Running tests"
"$PY" -m pytest

echo "==> Training En_Bag"
"$PY" train.py \
  --model en_bag \
  --data "$DATA" \
  --url-col "$URL_COL" \
  --label-col "$LABEL_COL" \
  --benign-value "$BENIGN_VALUE"

echo "==> Training En_kNN"
"$PY" train.py \
  --model en_knn \
  --data "$DATA" \
  --url-col "$URL_COL" \
  --label-col "$LABEL_COL" \
  --benign-value "$BENIGN_VALUE"

echo "==> Training RFE + XGBoost"
"$PY" train.py \
  --model rfe_xgb \
  --rfe-k 20 \
  --xgb-n-estimators 300 \
  --data "$DATA" \
  --url-col "$URL_COL" \
  --label-col "$LABEL_COL" \
  --benign-value "$BENIGN_VALUE"

notify_success
echo "==> DONE"
