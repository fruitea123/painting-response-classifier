from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# Standardized internal column names used by downstream modules.
GROUP_COLUMN = "unique_id"
TARGET_COLUMN = "label"

TEXT_FEEL_COLUMN = "text_feel"
TEXT_SOUNDTRACK_COLUMN = "text_soundtrack"
TEXT_FOOD_COLUMN = "text_food"
TEXT_FEATURE_COLUMNS = [TEXT_FEEL_COLUMN, TEXT_SOUNDTRACK_COLUMN, TEXT_FOOD_COLUMN]

INTENSITY_COLUMN = "intensity"
COLOUR_COUNT_COLUMN = "colour_count"
OBJECT_COUNT_COLUMN = "object_count"
LIKERT_SOMBRE_COLUMN = "likert_sombre"
LIKERT_CONTENT_COLUMN = "likert_content"
LIKERT_CALM_COLUMN = "likert_calm"
LIKERT_UNEASY_COLUMN = "likert_uneasy"
PAYMENT_COLUMN = "payment"

STRUCTURED_FEATURE_COLUMNS = [
    INTENSITY_COLUMN,
    COLOUR_COUNT_COLUMN,
    OBJECT_COUNT_COLUMN,
    LIKERT_SOMBRE_COLUMN,
    LIKERT_CONTENT_COLUMN,
    LIKERT_CALM_COLUMN,
    LIKERT_UNEASY_COLUMN,
    PAYMENT_COLUMN,
]

ROOM_COLUMN = "room"
VIEW_WITH_COLUMN = "view_with"
SEASON_COLUMN = "season"
CATEGORICAL_MULTI_COLUMNS = [ROOM_COLUMN, VIEW_WITH_COLUMN, SEASON_COLUMN]

# Conservative, explicit mappings based on current dataset columns.
# Only exact matches from canonical + small alias lists are allowed.
CANONICAL_COLUMNS = {
    "unique_id": "unique_id",
    "label": "Painting",
    "text_feel": "Describe how this painting makes you feel.",
    "text_soundtrack": (
        "Imagine a soundtrack for this painting. Describe that soundtrack without naming "
        "any objects in the painting."
    ),
    "intensity": "On a scale of 1\u6bcf10, how intense is the emotion conveyed by the artwork?",
    "colour_count": "How many prominent colours do you notice in this painting?",
    "object_count": "How many objects caught your eye in the painting?",
    "likert_sombre": "This art piece makes me feel sombre.",
    "likert_content": "This art piece makes me feel content.",
    "likert_calm": "This art piece makes me feel calm.",
    "likert_uneasy": "This art piece makes me feel uneasy.",
    "payment": "How much (in Canadian dollars) would you be willing to pay for this painting?",
    "room": "If you could purchase this painting, which room would you put that painting in?",
    "view_with": "If you could view this art in person, who would you want to view it with?",
    "season": "What season does this art piece remind you of?",
    "text_food": "If this painting was a food, what would be?",
}

COLUMN_ALIASES = {
    "intensity": [
        "On a scale of 1-10, how intense is the emotion conveyed by the artwork?",
        "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?",
        "On a scale of 1 to 10, how intense is the emotion conveyed by the artwork?",
    ],
}

_LIKERT_RE = re.compile(r"^\s*([1-5])")
_PAYMENT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def resolve_columns(columns: Iterable[str], require_label: bool = True) -> dict[str, str]:
    """Resolve required semantic keys to raw CSV column names."""
    available_columns = list(columns)
    available = set(available_columns)
    required_keys = [
        "unique_id",
        "text_feel",
        "text_soundtrack",
        "intensity",
        "colour_count",
        "object_count",
        "likert_sombre",
        "likert_content",
        "likert_calm",
        "likert_uneasy",
        "payment",
        "room",
        "view_with",
        "season",
        "text_food",
    ]
    if require_label:
        required_keys.insert(1, "label")

    resolved: dict[str, str] = {}
    missing: list[tuple[str, list[str]]] = []

    for key in required_keys:
        candidates = [CANONICAL_COLUMNS[key], *COLUMN_ALIASES.get(key, [])]
        hit = next((candidate for candidate in candidates if candidate in available), None)
        if hit is None:
            missing.append((key, candidates))
        else:
            resolved[key] = hit

    if missing:
        lines = ["Required columns are missing after explicit mapping:"]
        for key, candidates in missing:
            lines.append(f"- {key}: expected one of {candidates}")
        lines.append(f"Available columns: {available_columns}")
        raise ValueError("\n".join(lines))

    return resolved


def validate_column_mapping(
    columns: Iterable[str], column_mapping: dict[str, str], require_label: bool = True
) -> None:
    """Validate a pre-saved mapping against currently available columns."""
    available = set(list(columns))
    required_keys = [
        "unique_id",
        "text_feel",
        "text_soundtrack",
        "intensity",
        "colour_count",
        "object_count",
        "likert_sombre",
        "likert_content",
        "likert_calm",
        "likert_uneasy",
        "payment",
        "room",
        "view_with",
        "season",
        "text_food",
    ]
    if require_label:
        required_keys.insert(1, "label")

    missing_keys = [key for key in required_keys if key not in column_mapping]
    if missing_keys:
        raise ValueError(f"Column mapping is missing required semantic keys: {missing_keys}")

    missing_raw_columns = [
        column_mapping[key] for key in required_keys if column_mapping[key] not in available
    ]
    if missing_raw_columns:
        raise ValueError(
            "Mapped raw columns are not present in the provided CSV: "
            f"{missing_raw_columns}"
        )


def normalize_text_value(value: object) -> str:
    """Lowercase + whitespace normalization only (minimal text cleaning)."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def parse_likert_value(value: object) -> float:
    """Extract the leading Likert digit (1-5), otherwise NaN."""
    if pd.isna(value):
        return np.nan
    match = _LIKERT_RE.match(str(value))
    if not match:
        return np.nan
    return float(match.group(1))


def parse_likert_series(series: pd.Series) -> pd.Series:
    digits = series.fillna("").astype(str).str.extract(_LIKERT_RE, expand=False)
    return pd.to_numeric(digits, errors="coerce")


def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_payment_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    match = _PAYMENT_RE.search(text)
    if not match:
        return np.nan
    amount = float(match.group(0))
    if amount < 0:
        return np.nan
    return amount


def parse_payment_series(series: pd.Series) -> pd.Series:
    return series.map(parse_payment_value)


def normalize_multiselect_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parts = [part.strip().lower() for part in text.split(",") if part.strip()]
    return ",".join(parts)


def clean_dataframe(
    df: pd.DataFrame, column_mapping: dict[str, str], require_label: bool = True
) -> pd.DataFrame:
    """Build a standardized clean DataFrame used by train/eval code."""
    validate_column_mapping(df.columns, column_mapping, require_label=require_label)

    cleaned = pd.DataFrame(index=df.index)
    cleaned[GROUP_COLUMN] = df[column_mapping["unique_id"]].astype(str).str.strip()
    cleaned[TEXT_FEEL_COLUMN] = df[column_mapping["text_feel"]].map(normalize_text_value)
    cleaned[TEXT_SOUNDTRACK_COLUMN] = df[column_mapping["text_soundtrack"]].map(
        normalize_text_value
    )
    cleaned[TEXT_FOOD_COLUMN] = df[column_mapping["text_food"]].map(normalize_text_value)

    cleaned[INTENSITY_COLUMN] = safe_numeric_series(df[column_mapping["intensity"]])
    cleaned[COLOUR_COUNT_COLUMN] = safe_numeric_series(df[column_mapping["colour_count"]])
    cleaned[OBJECT_COUNT_COLUMN] = safe_numeric_series(df[column_mapping["object_count"]])
    cleaned[LIKERT_SOMBRE_COLUMN] = parse_likert_series(df[column_mapping["likert_sombre"]])
    cleaned[LIKERT_CONTENT_COLUMN] = parse_likert_series(df[column_mapping["likert_content"]])
    cleaned[LIKERT_CALM_COLUMN] = parse_likert_series(df[column_mapping["likert_calm"]])
    cleaned[LIKERT_UNEASY_COLUMN] = parse_likert_series(df[column_mapping["likert_uneasy"]])
    cleaned[PAYMENT_COLUMN] = parse_payment_series(df[column_mapping["payment"]])
    cleaned[ROOM_COLUMN] = df[column_mapping["room"]].map(normalize_multiselect_value)
    cleaned[VIEW_WITH_COLUMN] = df[column_mapping["view_with"]].map(normalize_multiselect_value)
    cleaned[SEASON_COLUMN] = df[column_mapping["season"]].map(normalize_multiselect_value)

    if require_label:
        cleaned[TARGET_COLUMN] = df[column_mapping["label"]].astype(str).str.strip()

    # Simple validity constraints for obviously bounded fields.
    cleaned.loc[
        (cleaned[INTENSITY_COLUMN] < 1) | (cleaned[INTENSITY_COLUMN] > 10), INTENSITY_COLUMN
    ] = np.nan
    for likert_col in [
        LIKERT_SOMBRE_COLUMN,
        LIKERT_CONTENT_COLUMN,
        LIKERT_CALM_COLUMN,
        LIKERT_UNEASY_COLUMN,
    ]:
        cleaned.loc[(cleaned[likert_col] < 1) | (cleaned[likert_col] > 5), likert_col] = np.nan

    for count_col in [COLOUR_COUNT_COLUMN, OBJECT_COUNT_COLUMN]:
        cleaned.loc[cleaned[count_col] < 0, count_col] = np.nan
    cleaned.loc[cleaned[PAYMENT_COLUMN] < 0, PAYMENT_COLUMN] = np.nan
    cleaned[PAYMENT_COLUMN] = np.log1p(cleaned[PAYMENT_COLUMN])
    return cleaned


def fit_numeric_fill_values(
    df: pd.DataFrame, numeric_columns: list[str] | None = None
) -> dict[str, float]:
    """Fit median fill values on train data only."""
    columns = numeric_columns or STRUCTURED_FEATURE_COLUMNS
    fill_values: dict[str, float] = {}
    for col in columns:
        median = df[col].median()
        fill_values[col] = float(median) if pd.notna(median) else 0.0
    return fill_values


def apply_numeric_fill_values(
    df: pd.DataFrame, fill_values: dict[str, float], numeric_columns: list[str] | None = None
) -> pd.DataFrame:
    columns = numeric_columns or list(fill_values.keys())
    filled = df.copy()
    for col in columns:
        filled[col] = filled[col].fillna(fill_values[col])
    return filled


def combine_text_columns(
    df: pd.DataFrame, text_columns: list[str] | None = None
) -> pd.Series:
    columns = text_columns or TEXT_FEATURE_COLUMNS
    combined = df[columns[0]].fillna("").astype(str)
    for col in columns[1:]:
        combined = combined.str.cat(df[col].fillna("").astype(str), sep=" ")
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()
