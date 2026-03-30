import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from src.BaseTrainer import MNB, BaseTrainer

class Trainer(BaseTrainer):
    model_type = MNB
    def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
        """Train Multinomial Naive Bayes with alpha tuning over TF-IDF features."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        alphas = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)

        stats = {}
        best = (-1.0, None, None, None, None, None)
        for i, alpha in enumerate(alphas):
            print(f"[tuning] testing alpha = {alpha}")
            model = MultinomialNB(alpha=alpha)

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

            f1 = float(np.mean(f1s))
            acc = float(np.mean(accs))
            stats[str(i)] = {
                "alpha": alpha,
                "accuracy": acc,
                "macro_f1": f1,
                "confusion matrix": str(cm),
            }

            if f1 > best[0]:
                best_model = MultinomialNB(alpha=alpha)
                best_model.fit(x_train, y_train)
                best = (f1, best_model, alpha, acc, f1, str(cm))

        stats["final"] = {
            "alpha": best[2],
            "accuracy": best[3],
            "macro_f1": best[4],
            "confusion matrix": best[5],
        }
        return best[1], stats