# Clean v0 Baseline Scaffold (Decision-Complete)

## Summary
- Goal: scaffold a minimal, readable training/evaluation baseline now; defer polished inference packaging.
- Chosen defaults from intent chat: `Train/Eval Only`, `pandas+sklearn` for training-time code, baseline family `LogReg + TF-IDF`.
- Course constraints extracted from [Instructions PDF](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/docs/CSC311_202601_Project_Instructions.pdf):
1. Final `pred.py` must expose `predict_all(csv_path)` and run with only stdlib + `numpy` + `pandas`.
2. Final prediction submission total size must be <= 10MB.
3. Training/dev code can use broader tooling (e.g., sklearn), and this is acceptable for v0.
- Proposal alignment from [Proposal PDF](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/docs/CSC311_proposal_with_figs.pdf): grouped split by `unique_id`, mixed structured+text features, accuracy + macro-F1.
- Current repo file purposes:
1. [README.md](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/README.md): one-line project description.
2. [starter/pred_example.py](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/starter/pred_example.py): required `predict_all` interface example.
3. [starter/project_baseline.py](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/starter/project_baseline.py): old kNN demo with non-current filename/columns and non-grouped split.
4. [data/training_data_202601.csv](c:/Users/菩提树/Desktop/ut/大二课程/csc311/painting-response-classifier/data/training_data_202601.csv): training data.
- Inferred data/task structure:
1. 1686 rows, 562 unique `unique_id`, exactly 3 rows per id.
2. Target `Painting` is perfectly balanced across 3 classes (562 each).
3. Prediction task is 3-class classification from mixed numeric/ordinal/categorical/multi-select/free-text responses.
4. Group leakage risk is real; split must be by `unique_id`.

## Proposed File Tree
```text
painting-response-classifier/
├─ data/
│  └─ training_data_202601.csv
├─ docs/
│  ├─ CSC311_202601_Project_Instructions.pdf
│  └─ CSC311_proposal_with_figs.pdf
├─ starter/
│  ├─ pred_example.py
│  └─ project_baseline.py
├─ baseline_v0/
│  ├─ __init__.py
│  ├─ schema.py
│  ├─ preprocess.py
│  ├─ split.py
│  ├─ features.py
│  ├─ model.py
│  └─ train_eval.py
├─ reports/
│  └─ .gitkeep
├─ requirements-train.txt
└─ README.md
```

## File/Directory Roles
1. `baseline_v0/schema.py`: resolve canonical column names from keyword matching (robust to encoding oddities like `1每10`).
2. `baseline_v0/preprocess.py`: parse helpers (`likert`, `payment`, multi-select tokenizer), missing handling, train-only fit stats.
3. `baseline_v0/split.py`: grouped train/val split by `unique_id` using fixed seed.
4. `baseline_v0/features.py`: fit/transform feature builders.
5. `baseline_v0/model.py`: train logistic regression baseline and compute metrics.
6. `baseline_v0/train_eval.py`: one script entrypoint to run full pipeline and print/save metrics.
7. `requirements-train.txt`: v0 training dependencies (`numpy`, `pandas`, `scikit-learn`, `scipy`).
8. `reports/`: output metrics JSON/CSV from v0 runs.
9. `starter/`: kept as reference only; not used by v0 pipeline.

## Interfaces and Implementation Changes
- New training entrypoint interface:
1. `python -m baseline_v0.train_eval --train_csv data/training_data_202601.csv --val_size 0.2 --seed 311`
2. Output: console metrics + `reports/v0_metrics.json`.
- Internal callable interfaces:
1. `split_grouped(df, group_col='unique_id', val_size=0.2, seed=311) -> (train_df, val_df)`.
2. `fit_featureizer(train_df) -> featureizer`.
3. `transform_features(featureizer, df) -> X`.
4. `train_baseline(X_train, y_train) -> model`.
5. `evaluate(model, X, y) -> {'accuracy': ..., 'macro_f1': ..., 'confusion_matrix': ...}`.
- Feature set for v0:
1. Text TF-IDF from `Describe how this painting makes you feel` + `Imagine a soundtrack...` (concatenated).
2. Numeric: intensity, prominent colours, object count.
3. Ordinal: 4 Likert columns parsed from leading digit.
4. Semi-structured numeric: payment parsed with regex first number, then clipped/log-transformed.
5. Multi-select categorical: room/view-with/season split on commas and multi-hot encoded.
- Explicitly out-of-scope in v0:
1. No `pred.py` final inference packaging yet.
2. No model-family sweep yet (single baseline only).
3. No hyperparameter search beyond minimal stable defaults.

## Test Plan
1. Smoke test: run training script end-to-end on full CSV and confirm metrics file is produced.
2. Leakage test: assert no overlap of `unique_id` between train/val.
3. Shape test: confirm `X_train.shape[1] == X_val.shape[1]`.
4. Robustness test: inject rows with missing values/odd payment strings and verify parser does not crash.
5. Metric sanity: verify accuracy is above random baseline (~0.33) on grouped validation split.

## Assumptions and TODOs
- Assumption: training CSV schema at dev time matches submission-time test CSV schema for shared predictor columns.
- Assumption: keeping `food` as free text (not categorical one-hot) is sufficient for v0.
- Assumption: single grouped holdout split (`val_size=0.2`) is enough for first checkpoint.
- TODO: finalize final-course `pred.py` path without sklearn/torch (likely via exported lightweight artifacts + numpy/pandas inference).
- TODO: decide final outlier policy for payment (clip rule) after first metric pass.
- TODO: verify any encoding-normalization needed for the intensity column name in future test files.
