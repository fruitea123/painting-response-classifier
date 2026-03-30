from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import fit_features
from src.model import evaluate_model
from src.preprocess import GROUP_COLUMN, TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grouped logistic baseline and export a lightweight inference artifact."
    )
    parser.add_argument("--train_csv", default="data/train.csv", help="Path to CSV containing sanitized training data.")
    parser.add_argument("--model", required=True, help="Python module containing the model training function.")
    parser.add_argument("--seed", type=int, default=311, help="Random seed for model.")
    parser.add_argument(
        "--ngram_max_values",
        default="2",
        help="Comma-separated max n-gram values to try, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--artifact_out",
        default="artifacts/baseline_logreg_tfidf.pkl",
        help="Output path for the lightweight inference artifact.",
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Trainer


def parse_ngram_max_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one n-gram max value must be provided.")
    if any(value < 1 for value in values):
        raise ValueError(f"Invalid n-gram values: {values}")
    return values


def main() -> None:
    args = parse_args()

    with open(args.train_csv, "r", encoding="utf-8") as file:
        train_df = pd.read_csv(file)

    y_train = train_df[TARGET_COLUMN].to_numpy()

    Trainer = load_module_from_path(args.model)
    ngram_max_values = parse_ngram_max_values(args.ngram_max_values)

    best_score = float("-inf")
    best_model = None
    best_stats = None
    best_feature_state = None
    best_train_metrics = None
    feature_search_stats: list[dict] = []

    for ngram_max in ngram_max_values:
        tfidf_config = {"ngram_range": (1, ngram_max)}
        print(f"[feature_tuning] testing ngram_range=(1, {ngram_max})")
        x_train, feature_state = fit_features(train_df, tfidf_config=tfidf_config)
        model, stats = Trainer.train(x_train, y_train, seed=args.seed)
        train_metrics, _ = evaluate_model(model, x_train, y_train)

        final_stats = stats.get("final", {})
        selection_score = float(final_stats.get("macro_f1", train_metrics["macro_f1"]))
        feature_search_stats.append(
            {
                "ngram_range": [1, ngram_max],
                "selection_macro_f1": selection_score,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "model_stats": stats,
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_model = model
            best_stats = stats
            best_feature_state = feature_state
            best_train_metrics = train_metrics

    model = best_model
    stats = best_stats
    feature_state = best_feature_state
    train_metrics = best_train_metrics

    metrics_payload = {
        "train_csv": str(args.train_csv),
        "model": args.model,
        "seed": args.seed,
        "train_rows": int(train_df.shape[0]),
        "train_unique_ids": int(train_df[GROUP_COLUMN].nunique()),
        "train_metrics": train_metrics,
        "feature_search": feature_search_stats
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

    model_state = Trainer.extract_artifact_state(model)
    if model_state is not None:
        # If 'extract_artifact_state' is implemented, do not save sklearn model
        model = None

    artifact_payload = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "train_csv": str(args.train_csv),
        "model_module": args.model,
        "model_type": Trainer.model_type,
        "feature_state": feature_state,
        "model_state": model_state,
        "model": model
    }

    artifact_out = Path(args.artifact_out)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    with artifact_out.open("wb") as f:
        pickle.dump(artifact_payload, f)
    print(f"[train] wrote artifact to {artifact_out}")


if __name__ == "__main__":
    main()
