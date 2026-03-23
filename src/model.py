from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train_logreg_baseline(x_train, y_train, seed: int = 311) -> LogisticRegression:
    """Train a simple multinomial logistic regression baseline."""
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_eval, y_true):
    """Return (metrics, predictions) with accuracy + macro-F1."""
    y_pred = model.predict(x_eval)
    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, np.asarray(y_pred)

