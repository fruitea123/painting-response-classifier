from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import transform_features
from src.model import evaluate_model
from src.preprocess import GROUP_COLUMN, TARGET_COLUMN, clean_dataframe, validate_column_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved baseline artifact.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained baseline artifact (.pkl).",
    )
    parser.add_argument(
        "--data_csv",
        default="data/test.csv",
        help="Path to sanitized, labelled CSV used for evaluation.",
    )
    parser.add_argument(
        "--subset",
        choices=["saved_test", "all"],
        default="saved_test",
        help="Evaluate on saved test IDs or all rows.",
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


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_csv = Path(args.data_csv)

    with model_path.open("rb") as f:
        artifact = pickle.load(f)

    with open(args.data_csv, "r", encoding="utf-8") as file:
        df_clean = pd.read_csv(file)

    if args.subset == "saved_test":
        test_unique_ids = sorted(df_clean[GROUP_COLUMN].astype(str).unique().tolist())
        saved_ids = set(str(v) for v in test_unique_ids)
        eval_df = df_clean[df_clean[GROUP_COLUMN].astype(str).isin(saved_ids)].copy()
        if eval_df.empty:
            raise ValueError(
                "Subset 'saved_test' produced zero rows. "
                "Check whether evaluation CSV matches training artifact IDs."
            )
    else:
        eval_df = df_clean

    feature_state = {
        "vectorizer": artifact["vectorizer"],
        "fill_values": artifact["fill_values"],
        "text_columns": artifact["feature_config"]["text_columns"],
        "structured_columns": artifact["feature_config"]["structured_columns"],
        "categorical_columns": artifact["feature_config"].get("categorical_columns", []),
        "categorical_encoders": artifact["feature_config"].get("categorical_encoders", {}),
        "tfidf_config": artifact["feature_config"]["tfidf_config"],
    }

    x_eval = transform_features(eval_df, feature_state)
    y_eval = eval_df[TARGET_COLUMN].to_numpy()
    eval_metrics, eval_predictions = evaluate_model(artifact["model"], x_eval, y_eval)

    metrics_payload = {
        "model_path": str(model_path),
        "data_csv": str(data_csv),
        "subset": args.subset,
        "n_eval_rows": int(eval_df.shape[0]),
        "n_etest_unique_ids": int(eval_df[GROUP_COLUMN].nunique()),
        "seed": artifact.get("seed"),
        "metrics": eval_metrics,
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
