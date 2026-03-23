from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocess import parse_likert_series, resolve_columns, safe_numeric_series


def _missing_mask(series: pd.Series) -> pd.Series:
    string_view = series.fillna("").astype(str).str.strip()
    return series.isna() | (string_view == "")


def _non_missing_strings(series: pd.Series) -> pd.Series:
    mask = _missing_mask(series)
    return series[~mask].astype(str).str.strip()


def infer_column_type(series: pd.Series) -> str:
    non_missing = _non_missing_strings(series)
    if non_missing.empty:
        return "categorical_like"

    numeric_rate = pd.to_numeric(non_missing, errors="coerce").notna().mean()
    avg_len = non_missing.str.len().mean()
    unique_ratio = non_missing.nunique() / max(1, len(non_missing))
    name = (series.name or "").lower()

    if name == "unique_id" or (unique_ratio > 0.95 and numeric_rate > 0.9):
        return "id_like"
    if numeric_rate > 0.9:
        return "numeric_like"
    if avg_len >= 25:
        return "text_like"
    return "categorical_like"


def _parse_failure_count_numeric(series: pd.Series) -> int:
    non_missing_mask = ~_missing_mask(series)
    parsed = safe_numeric_series(series)
    return int((non_missing_mask & parsed.isna()).sum())


def _parse_failure_count_likert(series: pd.Series) -> int:
    non_missing_mask = ~_missing_mask(series)
    parsed = parse_likert_series(series)
    return int((non_missing_mask & parsed.isna()).sum())


def run_data_audit(df: pd.DataFrame) -> dict:
    n_rows, n_columns = df.shape
    column_names = list(df.columns)

    missing = {}
    unique = {}
    inferred_types = {}
    for col in column_names:
        missing_mask = _missing_mask(df[col])
        missing_count = int(missing_mask.sum())
        unique_count = int(_non_missing_strings(df[col]).nunique())
        missing[col] = {
            "count": missing_count,
            "rate": round(missing_count / max(1, n_rows), 6),
        }
        unique[col] = {
            "count": unique_count,
            "ratio": round(unique_count / max(1, n_rows), 6),
        }
        inferred_types[col] = infer_column_type(df[col])

    group_checks = None
    if "unique_id" in df.columns:
        group_sizes = df.groupby("unique_id").size()
        group_checks = {
            "n_unique_ids": int(group_sizes.shape[0]),
            "min_group_size": int(group_sizes.min()),
            "median_group_size": float(group_sizes.median()),
            "max_group_size": int(group_sizes.max()),
            "non_three_group_count": int((group_sizes != 3).sum()),
        }

    label_distribution = None
    if "Painting" in df.columns:
        label_counts = df["Painting"].fillna("<<MISSING>>").astype(str).value_counts()
        label_distribution = {
            label: {
                "count": int(count),
                "rate": round(float(count) / max(1, n_rows), 6),
            }
            for label, count in label_counts.items()
        }

    suspicious: list[dict[str, str | int | float]] = []

    for col, stats in missing.items():
        if stats["rate"] >= 0.05:
            suspicious.append(
                {
                    "column": col,
                    "issue": "high_missing_rate",
                    "value": stats["rate"],
                }
            )

    for col in column_names:
        if inferred_types[col] == "categorical_like":
            unique_count = unique[col]["count"]
            unique_ratio = unique[col]["ratio"]
            if unique_count >= 100 and unique_ratio >= 0.3:
                suspicious.append(
                    {
                        "column": col,
                        "issue": "high_cardinality_categorical",
                        "value": unique_count,
                    }
                )

    try:
        resolved = resolve_columns(df.columns, require_label=False)

        for key in ["intensity", "colour_count", "object_count"]:
            raw_col = resolved[key]
            failures = _parse_failure_count_numeric(df[raw_col])
            if failures > 0:
                suspicious.append(
                    {
                        "column": raw_col,
                        "issue": "numeric_parse_failures",
                        "value": failures,
                    }
                )

        for key in ["likert_sombre", "likert_content", "likert_calm", "likert_uneasy"]:
            raw_col = resolved[key]
            failures = _parse_failure_count_likert(df[raw_col])
            if failures > 0:
                suspicious.append(
                    {
                        "column": raw_col,
                        "issue": "likert_parse_failures",
                        "value": failures,
                    }
                )

        intensity_raw = resolved["intensity"]
        intensity = safe_numeric_series(df[intensity_raw])
        intensity_out_of_range = int(((intensity < 1) | (intensity > 10)).fillna(False).sum())
        if intensity_out_of_range > 0:
            suspicious.append(
                {
                    "column": intensity_raw,
                    "issue": "out_of_range_intensity_1_10",
                    "value": intensity_out_of_range,
                }
            )

        for key in ["colour_count", "object_count"]:
            raw_col = resolved[key]
            counts = safe_numeric_series(df[raw_col])
            negative_count = int((counts < 0).fillna(False).sum())
            if negative_count > 0:
                suspicious.append(
                    {
                        "column": raw_col,
                        "issue": "negative_count_values",
                        "value": negative_count,
                    }
                )
    except ValueError as exc:
        suspicious.append(
            {
                "column": "<<schema>>",
                "issue": "column_resolution_failed_for_audit_checks",
                "value": str(exc),
            }
        )

    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "column_names": column_names,
        "inferred_column_types": inferred_types,
        "missing": missing,
        "unique": unique,
        "group_checks": group_checks,
        "label_distribution": label_distribution,
        "suspicious": suspicious,
    }


def save_audit_json(audit: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def print_audit_summary(audit: dict) -> None:
    print(f"[audit] rows={audit['n_rows']} cols={audit['n_columns']}")
    print(f"[audit] columns={audit['column_names']}")

    if audit.get("group_checks"):
        g = audit["group_checks"]
        print(
            "[audit] group_checks="
            f"n_unique_ids={g['n_unique_ids']} "
            f"min/median/max={g['min_group_size']}/{g['median_group_size']}/{g['max_group_size']} "
            f"non_three={g['non_three_group_count']}"
        )

    if audit.get("label_distribution"):
        print(f"[audit] label_distribution={audit['label_distribution']}")

    suspicious = audit.get("suspicious", [])
    print(f"[audit] suspicious_count={len(suspicious)}")
    for item in suspicious[:8]:
        print(f"[audit] suspicious: {item}")

