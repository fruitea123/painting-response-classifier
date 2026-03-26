from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import fit_features, transform_features
from src.model import evaluate_model
from src.preprocess import GROUP_COLUMN, TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped baseline model.")
    parser.add_argument("--train_csv", default="data/train.csv", help="Path to CSV containing sanitized training data.")
    parser.add_argument("--model", required=True, help="Python module containing the model training function.")
    parser.add_argument("--seed", type=int, default=311, help="Random seed for model.")
    parser.add_argument(
        "--artifact_out",
        default="artifacts/baseline_logreg_tfidf.pkl",
        help="Output path for serialized model artifact.",
    )
    parser.add_argument(
        "--metrics_out",
        default="reports/train_metrics.json",
        help="Output path for training metrics JSON.",
    )
    return parser.parse_args()


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("user_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main() -> None:
    args = parse_args()

    with open(args.train_csv, "r", encoding="utf-8") as file:
        train_df = pd.read_csv(file)

    x_train, feature_state = fit_features(train_df)
    # x_val = transform_features(val_df, feature_state)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    # y_val = val_df[TARGET_COLUMN].to_numpy()

    train = load_module_from_path(args.model).train
    model, stats = train(x_train, y_train, seed=args.seed)
    train_metrics, _ = evaluate_model(model, x_train, y_train)

    metrics_payload = {
        "train_csv": str(args.train_csv),
        "seed": args.seed,
        "train_rows": int(train_df.shape[0]),
        "train_unique_ids": int(train_df[GROUP_COLUMN].nunique()),
        "train_metrics": train_metrics,
        "tune_stats": stats 
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
    print(f"[train] wrote metrics to {metrics_out}")

    artifact_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model": model,
        "vectorizer": feature_state["vectorizer"],
        "fill_values": feature_state["fill_values"],
        "feature_config": {
            "text_columns": feature_state["text_columns"],
            "structured_columns": feature_state["structured_columns"],
            "tfidf_config": feature_state["tfidf_config"],
        }
    }

    artifact_out = Path(args.artifact_out)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    with artifact_out.open("wb") as f:
        pickle.dump(artifact_payload, f)
    print(f"[train] wrote artifact to {artifact_out}")


if __name__ == "__main__":
    main()