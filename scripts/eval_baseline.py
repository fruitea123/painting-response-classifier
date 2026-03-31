from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline import load_module_from_path
from src.BaseTrainer import *

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.PaintingClassifier import *
from src.transform import transform_features
from src.model import evaluate_model
from src.preprocess import (
    CATEGORICAL_MULTI_COLUMNS,
    GROUP_COLUMN,
    STRUCTURED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEXT_FEATURE_COLUMNS,
    clean_dataframe,
    resolve_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the lightweight inference artifact and verify sklearn parity."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained lightweight inference artifact (.pkl).",
    )
    parser.add_argument(
        "--data_csv",
        default="data/test.csv",
        help="Path to a labelled CSV used for evaluation.",
    )
    parser.add_argument(
        "--subset",
        choices=["all"],
        default="all",
        help="Evaluate on all rows in the provided labelled CSV.",
    )
    parser.add_argument(
        "--metrics_out",
        default="reports/eval_metrics.json",
        help="Output path for evaluation metrics JSON.",
    )
    parser.add_argument(
        "--predictions_out",
        default="reports/eval_predictions.csv",
        help="Output path for evaluation prediction trace CSV.",
    )
    return parser.parse_args()


def ensure_standardized_eval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    standardized_columns = {
        GROUP_COLUMN,
        TARGET_COLUMN,
        *TEXT_FEATURE_COLUMNS,
        *STRUCTURED_FEATURE_COLUMNS,
        *CATEGORICAL_MULTI_COLUMNS,
    }
    if standardized_columns.issubset(df.columns):
        return df.copy()

    column_mapping = resolve_columns(df.columns, require_label=True)
    return clean_dataframe(df, column_mapping=column_mapping, require_label=True)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_csv = Path(args.data_csv)

    with model_path.open("rb") as f:
        artifact = pickle.load(f)

    with open(args.data_csv, "r", encoding="utf-8") as file:
        raw_df = pd.read_csv(file)

    trainer = load_module_from_path(artifact["model_module"])

    df_clean = ensure_standardized_eval_dataframe(raw_df)

    eval_df = df_clean

    x_eval = transform_features(eval_df, artifact["feature_state"])
    y_eval = eval_df[TARGET_COLUMN].to_numpy()

    used_lightweight = False
    if artifact["model_state"] is not None:
        # Build lightweight model, if implemented

        if artifact["model_type"] == LOGREG:
            model = PaintingClassifierLogreg.from_artifact(artifact["model_state"])
        elif artifact["model_type"] == MNB:
            model = PaintingClassifierMNB.from_artifact(artifact["model_state"])
        else:
            raise NotImplementedError("lightweight model for selected model type not implemented.")

        used_lightweight = True
    else:
        # Assume if no saved model state, an sklearn model was saved.
        print("[eval] model state not found. using saved sklearn model.")
        model = artifact["model"]

    eval_metrics, eval_predictions = evaluate_model(model, x_eval, y_eval)
    sklearn_reference_model = trainer.build_sklearn_reference_model(artifact["model_state"])
    
    if sklearn_reference_model:
        sklearn_predictions = np.asarray(sklearn_reference_model.predict(x_eval))
        mismatch_count = int(np.sum(sklearn_predictions != eval_predictions))

        if mismatch_count > 0:
            raise AssertionError(
                "Lightweight inference path diverged from sklearn logistic regression "
                f"on {mismatch_count} evaluation rows."
            )
    else:
        mismatch_count = None

    metrics_payload = {
        "model_path": str(model_path),
        "data_csv": str(data_csv),
        "subset": args.subset,
        "n_eval_rows": int(eval_df.shape[0]),
        "n_eval_unique_ids": int(eval_df[GROUP_COLUMN].nunique()),
        "seed": artifact.get("seed"),
        "metrics": eval_metrics,
        "used_lightweight_model": used_lightweight,
        "parity": {
            "matches_sklearn_reference": True,
            "mismatch_count": mismatch_count,
        },
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(metrics_out, "r", encoding="utf-8-sig") as file:
            metrics_file = json.load(file)
    except FileNotFoundError:
        metrics_file = {}

    metrics_file[args.model] = metrics_payload
    metrics_out.write_text(
        json.dumps(metrics_file, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[eval] wrote metrics to {metrics_out}")

    predictions_out = Path(args.predictions_out)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            GROUP_COLUMN: eval_df[GROUP_COLUMN],
            "y_true": y_eval,
            "y_pred": eval_predictions,
        }
    ).to_csv(predictions_out, index=False)
    print(f"[eval] wrote predictions to {predictions_out}")


if __name__ == "__main__":
    main()
