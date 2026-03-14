# Final Revised v0 Scaffold Plan (Thin, Reusable, Train/Eval Only)

## Summary
- Keep reusable logic in `src/`, CLI entrypoints in `scripts/` for clarity.
- Keep modules small and explicit; no large all-in-one `pipeline.py`.
- Use grouped split by `unique_id`, train a single `LogReg + TF-IDF` baseline, report `accuracy` + `macro_f1`.
- Save metrics to `reports/` and save a model artifact with minimal reproducibility metadata.
- Defer final course `pred.py` packaging (no sklearn/torch at inference) to later.

## Proposed File Tree
```text
painting-response-classifier/
├─ src/
│  ├─ __init__.py
│  ├─ audit.py
│  ├─ preprocess.py
│  ├─ split.py
│  ├─ features.py
│  └─ model.py
├─ scripts/
│  ├─ train_baseline.py
│  └─ eval_baseline.py
├─ reports/
│  └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ starter/
│  ├─ pred_example.py
│  └─ project_baseline.py
├─ requirements-train.txt
└─ README.md
```

## File Responsibilities
- `src/audit.py`: lightweight dataset audit and optional JSON export.
- `src/preprocess.py`: explicit required-column mapping, safe text/numeric/Likert cleaning, train-time numeric fill stats.
- `src/split.py`: grouped train/validation split by `unique_id`.
- `src/features.py`: TF-IDF text features + small structured numeric block assembly.
- `src/model.py`: baseline model fit, predict, metric computation.
- `scripts/train_baseline.py`: orchestration for train+val evaluation, artifact/metrics saving.
- `scripts/eval_baseline.py`: load artifact, reproduce preprocessing/features, evaluate on labeled CSV (default subset = saved validation IDs).
- `README.md`: short baseline section (what exists now vs deferred later).

## Conservative Column Resolution
- Use explicit required column names matching current CSV.
- Allow only a small hardcoded alias list for known near-equivalent headers.
- No fuzzy/heuristic guessing of unknown columns.
- If required column is missing after explicit+alias lookup, fail with a clear error.

## Lightweight Data Audit Scope
- Column names.
- Inferred type per column (`numeric_like`, `text_like`, `categorical_like`, `id_like`) via simple rules.
- Missing counts and rates.
- Unique counts and unique ratios.
- `unique_id` group checks: unique IDs, min/median/max group size, count of non-3-sized groups.
- Label distribution for `Painting`.
- Suspicious summary list:
  - parse-failure rates for numeric/Likert fields
  - out-of-range counts for constrained fields (e.g., intensity outside 1–10)
  - high-missing columns (>=5%)
  - high-cardinality categorical columns

## Feature/Cleaning Scope (Intentionally Small)
- Included:
  - Combined normalized text from two main text fields -> TF-IDF.
  - Structured features: intensity, colour count, object count, and 4 Likert scores.
  - Missing handling: text -> empty string, numeric/Likert -> train median.
  - Simple validity checks: invalid constrained values -> `NaN` before imputation.
- Excluded for now:
  - Payment parsing.
  - Multi-select categorical expansion (room/view-with/season).
  - Food field featureization.
  - Complex anomaly correction logic.

## Artifact + Reproducibility Metadata
- Save `artifacts/baseline_logreg_tfidf.pkl` containing:
  - trained model
  - fitted TF-IDF vectorizer
  - numeric fill values
  - resolved column mapping actually used
  - feature config (text columns, structured columns, TF-IDF params)
  - split seed
  - validation `unique_id` list
- Save `reports/train_metrics.json` and `reports/eval_metrics.json`.
- Optional save `reports/data_audit.json`.

## Entrypoints
- Train:
  - `python scripts/train_baseline.py --train_csv data/training_data_202601.csv --val_size 0.2 --seed 311 --audit_json reports/data_audit.json`
- Eval:
  - `python scripts/eval_baseline.py --model artifacts/baseline_logreg_tfidf.pkl --data_csv data/training_data_202601.csv --subset saved_val`

## Test Plan
1. Train script runs end-to-end and writes artifact + metrics.
2. Group leakage guard: no overlap between train/val `unique_id`.
3. Eval script reproduces metrics on saved validation IDs.
4. Audit JSON contains all required sections.
5. Baseline metric sanity check: validation accuracy above random (~0.33).

## Assumptions and Deferred TODOs
- Assumptions:
  - Current training CSV schema is representative of labeled dev data.
  - Training/evaluation environment can install `pandas` and `scikit-learn`.
- Deferred TODOs:
  - Final submission `pred.py` inference path without sklearn/torch.
  - Broader feature set and model-family comparisons.
  - More robust handling of semi-structured/noisy columns.
