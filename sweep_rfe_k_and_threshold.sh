#!/usr/bin/env bash
set -euo pipefail

# Pick venv python (Windows Git Bash or Linux)
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
if [[ ! -f "$PYTHON" ]]; then
  PYTHON=".venv/bin/python"
fi

# Inputs
TRAIN_DATA="${TRAIN_DATA:-balanced_urls.csv}"
TRAIN_URL_COL="${TRAIN_URL_COL:-url}"
TRAIN_LABEL_COL="${TRAIN_LABEL_COL:-label}"

EXT_DATA="${EXT_DATA:-phishing_simple.csv}"
URL_COL="${URL_COL:-url}"
LABEL_COL="${LABEL_COL:-label}"

# external split (no leakage)
SEED="${SEED:-42}"
CALIB_FRAC="${CALIB_FRAC:-0.2}"
EXT_CALIB="${EXT_CALIB:-reports/external_calib.csv}"
EXT_TEST="${EXT_TEST:-reports/external_test.csv}"

# optimization constraint
MAX_FPR="${MAX_FPR:-0.005}"

# sweep space
K_LIST="${K_LIST:-8 10 12 14 16 18 20 22 24}"

# output
OUT_CSV="${OUT_CSV:-reports/rfe_k_threshold_sweep.csv}"

mkdir -p reports artifacts
rm -f "$OUT_CSV"

echo "==> Splitting external into calib/test (seed=$SEED, calib_frac=$CALIB_FRAC)"
"$PYTHON" - <<'PY'
import os
import pandas as pd
from sklearn.model_selection import train_test_split

src=os.environ.get("EXT_DATA","phishing_simple.csv")
url_col=os.environ.get("URL_COL","url")
label_col=os.environ.get("LABEL_COL","label")
seed=int(os.environ.get("SEED","42"))
calib_frac=float(os.environ.get("CALIB_FRAC","0.2"))
out_calib=os.environ.get("EXT_CALIB","reports/external_calib.csv")
out_test=os.environ.get("EXT_TEST","reports/external_test.csv")

df=pd.read_csv(src).dropna(subset=[url_col,label_col]).reset_index(drop=True)
df[label_col]=df[label_col].astype(int)

calib,test=train_test_split(df, test_size=1.0-calib_frac, random_state=seed, stratify=df[label_col])
calib.to_csv(out_calib, index=False)
test.to_csv(out_test, index=False)

print("calib:", len(calib), "test:", len(test))
print("calib dist:", calib[label_col].value_counts().to_dict())
print("test  dist:", test[label_col].value_counts().to_dict())
PY

# CSV header
echo "timestamp,k,threshold,max_fpr,calib_recall,calib_precision,calib_fpr,test_recall,test_precision,test_fpr,test_accuracy" >> "$OUT_CSV"

for k in $K_LIST; do
  MODEL_DIR="artifacts/rfe_xgb_k${k}"
  JSONFILE="reports/_calib_k${k}.json"
  TMP_EVAL="reports/_tmp_eval_k${k}.csv"

  echo "==> Train rfe_xgb with rfe-k=$k -> $MODEL_DIR"
  "$PYTHON" train.py \
    --model rfe_xgb \
    --data "$TRAIN_DATA" --url-col "$TRAIN_URL_COL" --label-col "$TRAIN_LABEL_COL" \
    --outdir "$MODEL_DIR" --rfe-k "$k"

  echo "==> Calibrate threshold on external_calib (max_fpr=$MAX_FPR)"
  "$PYTHON" calibrate_threshold.py \
    --modeldir "$MODEL_DIR" \
    --data "$EXT_CALIB" --url-col "$URL_COL" --label-col "$LABEL_COL" \
    --max-fpr "$MAX_FPR" --print-json > "$JSONFILE"

  THR=$("$PYTHON" -c "import json; print(json.load(open('$JSONFILE','r',encoding='utf-8'))['threshold'])")
  CAL_REC=$("$PYTHON" -c "import json; print(json.load(open('$JSONFILE','r',encoding='utf-8'))['recall'])")
  CAL_PRE=$("$PYTHON" -c "import json; print(json.load(open('$JSONFILE','r',encoding='utf-8'))['precision'])")
  CAL_FPR=$("$PYTHON" -c "import json; print(json.load(open('$JSONFILE','r',encoding='utf-8'))['fpr'])")

  echo "==> Evaluate on external_test with threshold=$THR"
  rm -f "$TMP_EVAL"
  "$PYTHON" evaluate.py \
    --modeldir "$MODEL_DIR" \
    --data "$EXT_TEST" --url-col "$URL_COL" --label-col "$LABEL_COL" \
    --threshold "$THR" --split external_test --out-csv "$TMP_EVAL" --notes "k=${k}"

  # read last row from tmp eval csv
  TEST_LINE=$("$PYTHON" - <<PY
import pandas as pd
df=pd.read_csv("$TMP_EVAL")
row=df.iloc[-1]
print(f"{row['recall']},{row['precision']},{row['fpr']},{row.get('accuracy','')}")
PY
)

  TS=$("$PYTHON" -c "import datetime; print(datetime.datetime.now().isoformat(timespec='seconds'))")
  echo "${TS},${k},${THR},${MAX_FPR},${CAL_REC},${CAL_PRE},${CAL_FPR},${TEST_LINE}" >> "$OUT_CSV"
done

echo "==> Done. Wrote: $OUT_CSV"
