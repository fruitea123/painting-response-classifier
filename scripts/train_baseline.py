from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import print_audit_summary, run_data_audit, save_audit_json
from src.features import fit_features, transform_features
from src.model import (
    evaluate_model,
    train_logreg_baseline,
    train_multinomial_nb_baseline,
)
from src.preprocess import GROUP_COLUMN, TARGET_COLUMN, clean_dataframe, resolve_columns
from src.split import grouped_train_val_split, has_group_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped baseline model.")
    parser.add_argument("--train_csv", required=True, help="Path to labeled training CSV.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=311, help="Random seed for split/model.")
    parser.add_argument(
        "--model_family",
        choices=["logreg", "mnb"],
        default="logreg",
        help="Which baseline model family to train.",
    )
    parser.add_argument(
        "--mnb_alpha",
        type=float,
        default=1.0,
        help="Smoothing parameter for Multinomial Naive Bayes.",
    )
    parser.add_argument(
        "--artifact_out",
        default="artifacts/baseline_logreg_tfidf.pkl",
        help="Output path for serialized model artifact.",
    )
    parser.add_argument(
        "--metrics_out",
        default="reports/train_metrics.json",
        help="Output path for training/validation metrics JSON.",
    )
    parser.add_argument(
        "--val_predictions_out",
        default="reports/val_predictions.csv",
        help="Output path for validation prediction trace CSV.",
    )
    parser.add_argument(
        "--audit_json",
        default=None,
        help="Optional output path for data audit JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv)
    df_raw = pd.read_csv(train_csv)

    audit = run_data_audit(df_raw)
    print_audit_summary(audit)
    if args.audit_json:
        save_audit_json(audit, args.audit_json)
        print(f"[train] wrote audit JSON to {args.audit_json}")

    column_mapping = resolve_columns(df_raw.columns, require_label=True)
    df_clean = clean_dataframe(df_raw, column_mapping=column_mapping, require_label=True)

    train_df, val_df = grouped_train_val_split(
        df_clean, group_col=GROUP_COLUMN, val_size=args.val_size, seed=args.seed
    )
    if has_group_leakage(train_df, val_df, group_col=GROUP_COLUMN):
        raise RuntimeError("Group leakage detected between train/validation splits.")

    x_train, feature_state = fit_features(train_df)
    x_val = transform_features(val_df, feature_state)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_val = val_df[TARGET_COLUMN].to_numpy()

    if args.model_family == "logreg":
        model = train_logreg_baseline(x_train, y_train, seed=args.seed)
    else:
        model = train_multinomial_nb_baseline(x_train, y_train, alpha=args.mnb_alpha)

    train_metrics, _ = evaluate_model(model, x_train, y_train)
    val_metrics, val_predictions = evaluate_model(model, x_val, y_val)

    metrics_payload = {
        "train_csv": str(train_csv),
        "seed": args.seed,
        "val_size": args.val_size,
        "model_family": args.model_family,
        "train_rows": int(train_df.shape[0]),
        "val_rows": int(val_df.shape[0]),
        "train_unique_ids": int(train_df[GROUP_COLUMN].nunique()),
        "val_unique_ids": int(val_df[GROUP_COLUMN].nunique()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] wrote metrics to {metrics_out}")

    val_predictions_out = Path(args.val_predictions_out)
    val_predictions_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            GROUP_COLUMN: val_df[GROUP_COLUMN],
            "y_true": y_val,
            "y_pred": val_predictions,
        }
    ).to_csv(val_predictions_out, index=False)
    print(f"[train] wrote validation predictions to {val_predictions_out}")

    val_unique_ids = sorted(val_df[GROUP_COLUMN].astype(str).unique().tolist())
    artifact_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_family": args.model_family,
        "model": model,
        "vectorizer": feature_state["vectorizer"],
        "fill_values": feature_state["fill_values"],
        "column_mapping": column_mapping,
        "feature_config": {
            "text_columns": feature_state["text_columns"],
            "structured_columns": feature_state["structured_columns"],
            "categorical_columns": feature_state["categorical_columns"],
            "categorical_encoders": feature_state["categorical_encoders"],
            "tfidf_config": feature_state["tfidf_config"],
        },
        "val_unique_ids": val_unique_ids,
    }

    artifact_out = Path(args.artifact_out)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    with artifact_out.open("wb") as f:
        pickle.dump(artifact_payload, f)
    print(f"[train] wrote artifact to {artifact_out}")


if __name__ == "__main__":
    main()
