from __future__ import annotations

from typing import Any

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from src.preprocess import (
    CATEGORICAL_MULTI_COLUMNS,
    STRUCTURED_FEATURE_COLUMNS,
    TEXT_FEATURE_COLUMNS,
    apply_numeric_fill_values,
    combine_text_columns,
    fit_numeric_fill_values,
)

DEFAULT_TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
}


def _split_multiselect_value(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _fit_categorical_features(train_df, categorical_columns: list[str]):
    matrices = []
    encoders = {}
    for col in categorical_columns:
        encoder = MultiLabelBinarizer(sparse_output=True)
        values = train_df[col].fillna("").map(_split_multiselect_value)
        matrices.append(encoder.fit_transform(values))
        encoders[col] = encoder
    if matrices:
        return hstack(matrices, format="csr"), encoders
    return csr_matrix((train_df.shape[0], 0)), encoders


def _transform_categorical_features(df, encoders: dict[str, MultiLabelBinarizer]):
    matrices = []
    for col, encoder in encoders.items():
        values = df[col].fillna("").map(_split_multiselect_value)
        matrices.append(encoder.transform(values))
    if matrices:
        return hstack(matrices, format="csr")
    return csr_matrix((df.shape[0], 0))


def fit_features(
    train_df,
    text_columns: list[str] | None = None,
    structured_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    tfidf_config: dict[str, Any] | None = None,
):
    """Fit text vectorizer + numeric fill stats, then build training matrix."""
    use_text_columns = text_columns or TEXT_FEATURE_COLUMNS
    use_structured_columns = structured_columns or STRUCTURED_FEATURE_COLUMNS
    use_categorical_columns = categorical_columns or CATEGORICAL_MULTI_COLUMNS
    use_tfidf_config = dict(DEFAULT_TFIDF_CONFIG)
    if tfidf_config:
        use_tfidf_config.update(tfidf_config)

    fill_values = fit_numeric_fill_values(train_df, numeric_columns=use_structured_columns)
    filled = apply_numeric_fill_values(
        train_df, fill_values=fill_values, numeric_columns=use_structured_columns
    )

    combined_text = combine_text_columns(filled, text_columns=use_text_columns)
    vectorizer = TfidfVectorizer(**use_tfidf_config)
    x_text = vectorizer.fit_transform(combined_text)

    x_structured = csr_matrix(filled[use_structured_columns].to_numpy(dtype=float))
    x_categorical, categorical_encoders = _fit_categorical_features(
        filled, use_categorical_columns
    )
    x_all = hstack([x_text, x_structured, x_categorical], format="csr")

    feature_state = {
        "vectorizer": vectorizer,
        "fill_values": fill_values,
        "text_columns": list(use_text_columns),
        "structured_columns": list(use_structured_columns),
        "categorical_columns": list(use_categorical_columns),
        "categorical_encoders": categorical_encoders,
        "tfidf_config": use_tfidf_config,
    }
    return x_all, feature_state


def transform_features(df, feature_state: dict[str, Any]):
    """Transform arbitrary split/data using pre-fitted feature state."""
    filled = apply_numeric_fill_values(
        df,
        fill_values=feature_state["fill_values"],
        numeric_columns=feature_state["structured_columns"],
    )
    combined_text = combine_text_columns(filled, text_columns=feature_state["text_columns"])

    x_text = feature_state["vectorizer"].transform(combined_text)
    x_structured = csr_matrix(
        filled[feature_state["structured_columns"]].to_numpy(dtype=float)
    )
    x_categorical = _transform_categorical_features(
        filled, feature_state.get("categorical_encoders", {})
    )
    return hstack([x_text, x_structured, x_categorical], format="csr")
