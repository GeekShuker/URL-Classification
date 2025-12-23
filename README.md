# URL-Classification

Phishing URL detector using lexical features only. The project trains CPU-only models that optimize recall under a strict false positive rate constraint (FPR ≤ 1%).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
All commands default to `balanced_urls.csv`, `url` column for URLs, and `label` column for labels.
Add `--sample-size 2000` (or similar) for a quick smoke test on a subset of rows.

### Bagging ensemble
```bash
python train.py --model en_bag --data balanced_urls.csv --url-col url --label-col label
```

### kNN ensemble
```bash
python train.py --model en_knn --knn-ks "3,5,7,9" --data balanced_urls.csv
```

### RFE + XGBoost
```bash
python train.py --model rfe_xgb --rfe-k 30 --xgb-n-estimators 300 --data balanced_urls.csv
```

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
```bash
python evaluate.py --modeldir artifacts/en_bag --data balanced_urls.csv
```

## Threshold tuning with FPR constraint
Validation probabilities are scanned to find the decision threshold that achieves the highest recall while keeping the false positive rate at or below the provided maximum (default 0.01). This threshold is then fixed and reused for test evaluation and downstream predictions.
