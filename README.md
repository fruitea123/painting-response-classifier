# Painting Response Classifier

CSC311 project for classifying paintings from human survey responses using text and structured features.

## Current Final Path

The submission path is now based on a single model family only:

- training-time model: multinomial naive Bayes in sklearn
- submission-time model: numpy-only multinomial naive Bayes predictor in `src/PaintingClassifier.py`
- submission entrypoint: `pred.py`
- required artifact path for submission: `artifacts/baseline_mnb_tfidf.pkl`

Training and evaluation use `pandas`, `numpy`, and `scikit-learn`. Inference in `pred.py` uses only the Python standard library plus `numpy` and `pandas`.

## Relevant Files

- `scripts/data_processing.py`: clean the labeled CSV and create grouped `train.csv` / `test.csv`
- `scripts/train_baseline.py`: train the selected baseline model and export the lightweight inference artifact
- `scripts/eval_baseline.py`: evaluate the lightweight artifact and verify parity with a reconstructed sklearn reference model
- `pred.py`: final submission interface with `predict_all(filename)`
- `src/PaintingClassifier.py`: numpy inference implementations for the submission models
- `src/preprocess.py`: column resolution and cleaning
- `src/features.py`: training-time feature fitting
- `src/transform.py`: inference-time feature transform using exported `feature_state`

## Lightweight Artifact Format

`artifacts/baseline_mnb_tfidf.pkl` stores:

- `feature_state`
- `model_state`
- minimal metadata: `artifact_version`, `created_at_utc`, `seed`, `train_csv`, `model_module`, `model_type`

No sklearn model object is required at inference time.

## Verify End To End

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Create the grouped train/test split and audit outputs:

```bash
python scripts/data_processing.py --data data/training_data_202601.csv --train data/train.csv --test data/test.csv --audit_json reports/data_audit.json --metrics_out reports/preprocess_metrics.json
```

Retrain and export the lightweight artifact:

```bash
python scripts/train_baseline.py --train_csv data/train.csv --model src/mnb.py --artifact_out artifacts/baseline_mnb_tfidf.pkl --metrics_out reports/train_metrics.json
```

Evaluate the lightweight artifact and run the sklearn parity check:

```bash
python scripts/eval_baseline.py --model artifacts/baseline_mnb_tfidf.pkl --data_csv data/test.csv --subset all --metrics_out reports/eval_metrics.json --predictions_out reports/eval_predictions.csv
```

Run the final submission interface on the no-label CSV:

```bash
python -c "import pred; preds = pred.predict_all('data/training_data_202601_nolabel.csv'); print(len(preds)); print(preds[:5])"
```

Strict sklearn-free inference check in PowerShell:

```powershell
@'
import builtins

real_import = builtins.__import__

def blocked(name, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked

import pred

preds = pred.predict_all("data/training_data_202601_nolabel.csv")
print(len(preds))
print(preds[:5])
'@ | python -
```

If that final command succeeds, the submission path is not importing sklearn.

## Submission Notes

- `pred.py` exposes `predict_all(filename)` and returns a Python list of labels.
- `pred.py` expects the artifact at `artifacts/baseline_mnb_tfidf.pkl`.
- There is no debug `__main__` block in `pred.py`.
- Report and artifact outputs under `reports/` and `artifacts/` are gitignored local outputs.
