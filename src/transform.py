import re
import numpy as np
import numpy.typing as npt

from collections import Counter

from src.preprocess import apply_numeric_fill_values, combine_text_columns

def split_multiselect_value(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _transform_categorical_features(df, encoders: dict[str, dict[str, npt.NDArray]]):
    matrices = []
    for col, encoder in encoders.items():
        values = df[col].fillna("").map(split_multiselect_value)
        matrices.append(multi_label(values, encoder))
    if matrices:
        return np.hstack(matrices)
    return np.zeros((df.shape[0], 0))


def multi_label(values, encoder: dict[str, npt.NDArray]) -> npt.NDArray:
    x = np.zeros((len(values), len(encoder["classes_"])))
    
    # 1 if the i-th value list contains the j-th class
    for i, value_list in enumerate(values):
        for j, class_ in enumerate(encoder["classes_"]):
            if class_ in value_list:
                x[i][j] = 1
    
    return x


def tf_idf(values, vectorizer: dict[str, dict | npt.NDArray], config: dict[str, any]):
    x = np.zeros((len(values), len(vectorizer["vocabulary_"])))

    for i, text in enumerate(values):
        # Tokenize text
        tokens = re.findall(vectorizer["token_pattern"], text)

        # Generate ngrams
        ngrams = []
        min_n, max_n = config["ngram_range"]
        for n in range(min_n, max_n + 1):
            for j in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[j:j+n]))
        
        ngrams = Counter(ngrams)

        for token in ngrams:
            if token in vectorizer["vocabulary_"]:
                j = vectorizer["vocabulary_"][token]
                tf = ngrams[token]
                idf = vectorizer["idf_"][j]
                x[i][j] = float(tf) * idf
        
        # Normalize row
        row_norm = np.sqrt(np.sum(x[i] ** 2))
        if row_norm > 0:
            x[i] /= row_norm
    
    return x


def transform_features(df, feature_state: dict):
    """Transform arbitrary split/data using pre-fitted feature state."""
    filled = apply_numeric_fill_values(
        df,
        fill_values=feature_state["fill_values"],
        numeric_columns=feature_state["structured_columns"],
    )
    combined_text = combine_text_columns(filled, text_columns=feature_state["text_columns"])

    x_text = tf_idf(combined_text, feature_state["vectorizer"], feature_state["tfidf_config"])
    x_structured = filled[feature_state["structured_columns"]].to_numpy(dtype=float)
    x_categorical = _transform_categorical_features(
        filled, feature_state.get("categorical_encoders", {})
    )
    return np.hstack([x_text, x_structured, x_categorical])