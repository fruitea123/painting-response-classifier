from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, x_eval, y_true):
    """Return (metrics, predictions) with accuracy + macro-F1."""
    y_pred = model.predict(x_eval)
    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion matrix": str(confusion_matrix(y_true, y_pred))
    }
    return metrics, np.asarray(y_pred)

