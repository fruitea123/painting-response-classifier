# Revised Minimal Baseline Scaffold (Train/Eval Only)

## Summary
- Keep a neutral, reusable structure under `src/` (no `v0` package naming).
- Keep implementation intentionally small: lightweight audit, simple cleaning, grouped split, TF-IDF + small structured features, logistic regression baseline.
- Use `pandas + sklearn` for training/eval only.
- Defer final `pred.py` packaging (no sklearn at inference) to a later step.

## Revised File Tree
```text
painting-response-classifier/
├─ data/
│  └─ training_data_202601.csv
├─ docs/
│  ├─ CSC311_202601_Project_Instructions.pdf
│  ├─ CSC311_proposal_with_figs.pdf
│  └─ v0_plan.md
├─ starter/
│  ├─ pred_example.py
│  └─ project_baseline.py
├─ src/
│  ├─ __init__.py
│  ├─ audit.py
│  ├─ preprocess.py
│  ├─ pipeline.py
│  ├─ train_baseline.py
│  └─ eval_baseline.py
├─ reports/
│  └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ requirements-train.txt
└─ README.md
```

## What Each File Will Do
1. `src/audit.py`
- Run a lightweight audit on input CSV.
- Print summary to console and optionally write `reports/data_audit.json`.

2. `src/preprocess.py`
- Hold simple column resolution and constants (no separate `schema.py`).
- Implement minimal cleaning helpers:
  - text normalization
  - safe numeric conversion
  - Likert leading-digit parsing
  - simple validity checks for constrained fields

3. `src/pipeline.py`
- Shared core functions for:
  - loading labeled data
  - grouped split by `unique_id`
  - feature prep (text + small structured set)
  - model fit/predict
  - metric computation (accuracy, macro-F1)

4. `src/train_baseline.py`
- Baseline training entrypoint.
- Flow: load -> audit -> split -> train -> validate -> save metrics and model artifact.
- Save:
  - `reports/train_metrics.json`
  - `reports/val_predictions.csv` (optional simple trace)
  - `artifacts/baseline_logreg_tfidf.pkl`

5. `src/eval_baseline.py`
- Evaluation entrypoint.
- Load saved artifact and evaluate on labeled data.
- Default behavior: evaluate on IDs stored as validation IDs from training artifact.
- Save `reports/eval_metrics.json`.

6. `requirements-train.txt`
- Minimal training deps: `numpy`, `pandas`, `scikit-learn`, `scipy`.

7. `README.md` (update)
- Add short section: purpose of each created file/dir, what v0 includes, what is deferred.

## Lightweight Data Audit (Exact Checks)
1. Column names.
2. Inferred column types per column (`numeric_like`, `short_text`, `long_text`, `categorical_like`, `id_like`) using simple heuristics.
3. Missing value counts and rates.
4. Unique value counts and unique ratio.
5. `unique_id` group-size checks:
- number of unique IDs
- min/median/max rows per ID
- count of IDs not having expected size 3
6. Label (`Painting`) distribution counts and proportions.
7. Suspicious/irregular summary list:
- parse failures in expected numeric/Likert columns
- out-of-range counts for constrained numeric fields (e.g., intensity outside 1–10)
- high missing-rate columns (threshold 5%)
- unusually high-cardinality categorical columns
- obvious extreme numeric outliers (reported, not aggressively fixed)

## Minimal Cleaning and Feature Scope
1. Included features (small first pass):
- Text TF-IDF from:
  - `Describe how this painting makes you feel.`
  - `Imagine a soundtrack ...`
  (combined into one normalized text field)
- Structured numeric/ordinal:
  - intensity
  - prominent colours
  - objects caught eye
  - 4 Likert sentiment columns

2. Excluded for now (deferred):
- payment parsing
- multi-select categorical expansion (room/view-with/season)
- food field featureization
- advanced outlier handling or heavy parsing systems

3. Grouping rule:
- `unique_id` only for grouped splitting and evaluation subseting, never as a feature.

## Public Interfaces / Entrypoints
1. Training:
- `python -m src.train_baseline --train_csv data/training_data_202601.csv --val_size 0.2 --seed 311 --audit_json reports/data_audit.json`

2. Evaluation:
- `python -m src.eval_baseline --model artifacts/baseline_logreg_tfidf.pkl --data_csv data/training_data_202601.csv --subset saved_val`

## Test Plan
1. Smoke test train command end-to-end; confirm artifact + metrics files exist.
2. Confirm grouped split leakage guard: no overlap between train/val `unique_id`.
3. Confirm evaluation command reproduces metrics on saved validation subset.
4. Confirm audit JSON contains required sections (columns/types/missing/unique/groups/labels/suspicious).
5. Sanity check: validation accuracy exceeds random baseline (~0.33).

## Assumptions
1. Training CSV remains labeled with `Painting` and `unique_id`.
2. Minor column-name encoding oddities may occur, so simple keyword-based column resolution is used inside `preprocess.py`.
3. For v0, simple imputation and validity handling is sufficient; no full data-cleaning framework is required.

## TODOs / Intentionally Deferred
1. Final competition/submission `pred.py` packaging without sklearn/torch.
2. Broader feature set (payment/multi-select/food) after baseline is stable.
3. Hyperparameter search and model-family comparison beyond the single baseline.
4. More robust anomaly policies once audit findings are reviewed.
