# URL-Classification

Phishing URL detector using lexical features only. The project trains CPU-only models that optimize recall under a strict false positive rate constraint (FPR ≤ 1%).

## Setup
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt

```
```bash
# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Training
All commands default to `balanced_urls.csv`, `url` column for URLs, and `label` column for labels.
Add `--sample-size 2000` (or similar) for a quick smoke test on a subset of rows.

### Available Models

#### Bagging ensemble
```bash
python train.py --model en_bag --data balanced_urls.csv --url-col url --label-col label
```

#### kNN ensemble
```bash
python train.py --model en_knn --knn-ks "3,5,7,9" --data balanced_urls.csv
```

#### RFE + XGBoost
```bash
python train.py --model rfe_xgb --rfe-k 24 --xgb-n-estimators 300 --data balanced_urls.csv
```

#### Random Forest
```bash
python train.py --model rf --data balanced_urls.csv --url-col url --label-col label
```
Optional parameters:
- `--rf-n-estimators` (default: 200)
- `--rf-max-depth` (default: 0, meaning unlimited)
- `--rf-min-samples-split` (default: 2)
- `--rf-min-samples-leaf` (default: 1)
- `--rf-max-features` (default: "sqrt", can be "sqrt", "log2", int, or float)

#### XGBoost (without RFE)
```bash
python train.py --model xgb --data balanced_urls.csv --url-col url --label-col label
```
Optional parameters:
- `--plain-xgb-n-estimators` (default: 200)
- `--plain-xgb-max-depth` (default: 6)
- `--plain-xgb-learning-rate` (default: 0.1)
- `--plain-xgb-subsample` (default: 0.8)
- `--plain-xgb-colsample-bytree` (default: 0.8)

#### CatBoost
```bash
python train.py --model catboost --data balanced_urls.csv --url-col url --label-col label
```
Optional parameters:
- `--catboost-iterations` (default: 200)
- `--catboost-depth` (default: 6)
- `--catboost-learning-rate` (default: 0.1)
- `--catboost-subsample` (default: 0.8)
- `--catboost-l2-leaf-reg` (default: 3.0)

## Prediction
Predict for a single URL:
```bash
python predict.py --modeldir artifacts/en_bag --url "http://example.com/login"
```

Predict from file:
```bash
python predict.py --modeldir artifacts/en_bag --input urls.txt
```

## Evaluation

Evaluate any trained model on a dataset:

```bash
# Bagging ensemble
python evaluate.py --modeldir artifacts/en_bag --data newDataSetKaggle.csv

# kNN ensemble
python evaluate.py --modeldir artifacts/en_knn --data newDataSetKaggle.csv

# RFE + XGBoost
python evaluate.py --modeldir artifacts/rfe_xgb --data newDataSetKaggle.csv

# Random Forest
python evaluate.py --modeldir artifacts/rf --data newDataSetKaggle.csv

# XGBoost (without RFE)
python evaluate.py --modeldir artifacts/xgb --data newDataSetKaggle.csv

# CatBoost
python evaluate.py --modeldir artifacts/catboost --data newDataSetKaggle.csv
```

You can also override the threshold:
```bash
python evaluate.py --modeldir artifacts/rfe_xgb --data newDataSetKaggle.csv --threshold 0.55
```
## Generating Reports

Generate comparison reports and plots for all trained models:

```bash
python report.py
```

This will:
- Scan all models in `artifacts/` directory
- Generate comparison tables (CSV and Markdown)
- Create plots for metrics (ROC-AUC, PR-AUC, F1, accuracy, recall, precision, FPR)
- Generate confusion matrices for each model
- Create feature importance plots (for models that support it: rf, xgb, catboost, rfe_xgb)
- Generate timing breakdown plots

Outputs are saved to `reports/` directory by default.

## Threshold tuning with FPR constraint
Validation probabilities are scanned to find the decision threshold that achieves the highest recall while keeping the false positive rate at or below the provided maximum (default 0.01). This threshold is then fixed and reused for test evaluation and downstream predictions.
