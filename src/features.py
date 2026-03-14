from __future__ import annotations

from typing import Any

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import (
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


def fit_features(
    train_df,
    text_columns: list[str] | None = None,
    structured_columns: list[str] | None = None,
    tfidf_config: dict[str, Any] | None = None,
):
    """Fit text vectorizer + numeric fill stats, then build training matrix."""
    use_text_columns = text_columns or TEXT_FEATURE_COLUMNS
    use_structured_columns = structured_columns or STRUCTURED_FEATURE_COLUMNS
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
    x_all = hstack([x_text, x_structured], format="csr")

    feature_state = {
        "vectorizer": vectorizer,
        "fill_values": fill_values,
        "text_columns": list(use_text_columns),
        "structured_columns": list(use_structured_columns),
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
    return hstack([x_text, x_structured], format="csr")

