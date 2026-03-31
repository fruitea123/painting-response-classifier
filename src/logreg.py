import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.BaseTrainer import LOGREG, BaseTrainer

class Trainer(BaseTrainer):
    model_type = LOGREG
    def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
        """Train a simple multinomial logistic regression baseline."""
        # Cross-validate with 5 folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # Hyperparameters:
        cs = (0.1, 1, 10, 25, 50, 75, 100, 200, 300, 1000)

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
                best = (f1, c, acc, f1, str(cm))

        # Retrain the best model on the entire dataset
        model = LogisticRegression(C=best[1], max_iter=2000, random_state=seed)
        model.fit(x_train, y_train)
        
        stats["final"] = {
            "c": best[1],
            "accuracy": best[2],
            "macro_f1": best[3],
            "confusion matrix": best[4]
        }

        return model, stats


    def extract_artifact_state(model) -> dict:
        # required_attrs = ("classes_", "coef_", "intercept_")
        # missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]
        # if missing_attrs:
        #     raise TypeError(
        #         "Final inference artifact export currently supports logistic-regression-style "
        #         f"models with {required_attrs}. Missing: {missing_attrs}"
        #     )

        return {
            "classes": np.asarray(model.classes_).copy(),
            "coef": np.asarray(model.coef_, dtype=float).copy(),
            "intercept": np.asarray(model.intercept_, dtype=float).copy(),
        }


    def build_sklearn_reference_model(artifact: dict) -> LogisticRegression:
        model = LogisticRegression()
        model.classes_ = np.asarray(artifact["classes"])
        model.coef_ = np.asarray(artifact["coef"], dtype=float)
        model.intercept_ = np.asarray(artifact["intercept"], dtype=float)
        model.n_features_in_ = model.coef_.shape[1]
        model.n_iter_ = np.ones(model.coef_.shape[0], dtype=np.int32)
        return model