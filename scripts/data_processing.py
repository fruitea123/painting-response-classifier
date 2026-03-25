import argparse
import json
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import print_audit_summary, run_data_audit, save_audit_json
from src.preprocess import GROUP_COLUMN, TARGET_COLUMN, clean_dataframe, resolve_columns
from src.split import grouped_train_val_split, has_group_leakage

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize data and split into test, train, and validation sets.")
    parser.add_argument("--data", required=True, help="Path of CSV file containing labelled data")
    parser.add_argument("--train", default="data/train.csv", help="Path of CSV file containing training data")
    parser.add_argument("--test", default="data/test.csv", help="Path of CSV file containing test data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test set, as a porportionality to the size of the entire data set.")
    parser.add_argument("--seed", type=int, default=311, help="Random seed for split.")
    parser.add_argument(
        "--audit_json",
        default=None,
        help="Optional output path for data audit JSON.",
    )
    parser.add_argument(
        "--metrics_out",
        default="reports/preprocess_metrics.json",
        help="Output path for propress metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.data, "r", encoding="utf-8") as file:
        df_raw = pd.read_csv(file)
    
    audit = run_data_audit(df_raw)
    print_audit_summary(audit)
    if args.audit_json:
        save_audit_json(audit, args.audit_json)
        print(f"[preprocess] wrote audit JSON to {args.audit_json}")

    column_mapping = resolve_columns(df_raw.columns, require_label=True)
    df_clean = clean_dataframe(df_raw, column_mapping=column_mapping, require_label=True)

    train_df, val_df = grouped_train_val_split(
        df_clean, group_col=GROUP_COLUMN, val_size=args.test_size, seed=args.seed
    )

    if has_group_leakage(train_df, val_df, group_col=GROUP_COLUMN):
        raise RuntimeError("Group leakage detected between train/validation splits.")
    
    metrics_payload = {
        "data_csv": str(args.data),
        "seed": args.seed,
        "val_size": args.test_size,
        "train_rows": int(train_df.shape[0]),
        "val_rows": int(val_df.shape[0]),
        "train_unique_ids": int(train_df[GROUP_COLUMN].nunique()),
        "val_unique_ids": int(val_df[GROUP_COLUMN].nunique()),
        "counts_train": {
            "The Persistence of Memory": len(train_df[train_df[TARGET_COLUMN] == "The Persistence of Memory"]),
            "The Starry Night": len(train_df[train_df[TARGET_COLUMN] == "The Starry Night"]),
            "The Water Lily Pond": len(train_df[train_df[TARGET_COLUMN] == "The Water Lily Pond"])
        },
        "counts_val": {
            "The Persistence of Memory": len(val_df[val_df[TARGET_COLUMN] == "The Persistence of Memory"]),
            "The Starry Night": len(val_df[val_df[TARGET_COLUMN] == "The Starry Night"]),
            "The Water Lily Pond": len(val_df[val_df[TARGET_COLUMN] == "The Water Lily Pond"])
        }
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[preprocess] wrote metrics to {metrics_out}")
    
    with open(args.train, "w", encoding="utf-8", newline="") as file:
        train_df.to_csv(file, index=False)
    
    with open(args.test, "w", encoding="utf-8", newline="") as file:
        val_df.to_csv(file, index=False)


if __name__ == "__main__":
    main()