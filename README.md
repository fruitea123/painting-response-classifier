# Painting Response Classifier
A machine learning project for classifying paintings from human survey responses using text and structured features.

## Baseline Scaffold (Train/Eval Only)

This repo now includes a thin baseline scaffold for CSC311 course development:

- Reusable code in `src/`
- CLI entrypoints in `scripts/`
- Outputs in `reports/` and `artifacts/`

### File Overview

- `src/audit.py`: lightweight data audit (columns, types, missingness, group checks, label balance, suspicious values)
- `src/preprocess.py`: conservative column resolution + minimal cleaning/parsing
- `src/split.py`: grouped train/validation split by `unique_id`
- `src/features.py`: TF-IDF text features + small numeric/ordinal feature block
- `src/model.py`: logistic regression training and evaluation metrics
- `scripts/train_baseline.py`: train baseline model, save metrics and artifact
- `scripts/eval_baseline.py`: load artifact and evaluate on labeled data
- `starter/`: reference starter files, unchanged

### What v0 Includes

- Grouped splitting by `unique_id` (leakage-aware)
- Baseline `LogReg + TF-IDF`
- Metrics: accuracy and macro-F1
- Saved outputs:
  - `reports/train_metrics.json`
  - `reports/eval_metrics.json`
  - optional `reports/data_audit.json`
  - `artifacts/baseline_logreg_tfidf.pkl` (model + minimal reproducibility metadata)

### Intentionally Deferred

- Final competition/submission `pred.py` packaging
- Inference-time implementation without sklearn/torch
- Larger feature engineering (payment/multi-select/food parsing)
- Model-family sweep and advanced tuning

### Training Dependencies

Install from:

```bash
pip install -r requirements-train.txt
```

Run:

```bash
python scripts/train_baseline.py --train_csv data/training_data_202601.csv --val_size 0.2 --seed 311 --audit_json reports/data_audit.json
python scripts/eval_baseline.py --model artifacts/baseline_logreg_tfidf.pkl --data_csv data/training_data_202601.csv --subset saved_val
```
