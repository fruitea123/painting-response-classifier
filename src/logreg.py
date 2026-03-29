import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
    """Train a simple multinomial logistic regression baseline."""
    # Cross-validate with 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Hyperparameters:
    cs = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


    stats = {}
    best = [-1, None, None, None, None]
    for i in range(len(cs)):
        c = cs[i]
        print(f"[tuning] testing c = {c}")

        model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        
        # Evaluate the model using cv and macro f1, keep track of accuracy as well
        f1s = []
        accs = []
        cm = np.zeros((3, 3))
        for train_idx, val_idx in cv.split(x_train, y_train):
            x_t, x_v = x_train[train_idx], x_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model.fit(x_t, y_t)
            preds = model.predict(x_v)

            f1s.append(f1_score(y_v, preds, average="macro"))
            accs.append(accuracy_score(y_v, preds))
            cm += confusion_matrix(y_v, preds)
        
        f1 = np.mean(f1s)
        acc = np.mean(accs)

        stats[str(i)] = {
            "c": c,
            "accuracy": acc,
            "macro_f1": f1,
            "confusion matrix": str(cm)
        }

        if f1 > best[0]:
            best_model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
            best_model.fit(x_train, y_train)
            best = (f1, best_model, c, acc, f1, str(cm))
    
    stats["final"] = {
        "c": best[2],
        "accuracy": best[3],
        "macro_f1": best[4],
        "confusion matrix": best[5]
    }

    return best[1], stats
