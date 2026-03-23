from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_val_split(
    df: pd.DataFrame,
    group_col: str = "unique_id",
    val_size: float = 0.2,
    seed: int = 311,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by group identifier to avoid leakage across splits."""
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' is not in DataFrame columns.")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def has_group_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, group_col: str = "unique_id"
) -> bool:
    train_groups = set(train_df[group_col].astype(str))
    val_groups = set(val_df[group_col].astype(str))
    return len(train_groups.intersection(val_groups)) > 0

