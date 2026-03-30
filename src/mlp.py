import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from src.BaseTrainer import MLP, BaseTrainer

class EncodedMLPClassifier(BaseEstimator):
    """Wrap MLPClassifier so training can use integer labels with early stopping."""

    def __init__(self, model: MLPClassifier):
        self.model = model
        self.label_encoder = LabelEncoder()

    def fit(self, x_train, y_train):
        y_encoded = self.label_encoder.fit_transform(y_train)
        self.model.fit(x_train, y_encoded)
        return self

    def predict(self, x_eval):
        y_encoded = self.model.predict(x_eval)
        return self.label_encoder.inverse_transform(y_encoded)

    @property
    def n_iter_(self) -> int:
        return int(self.model.n_iter_)


class Trainer(BaseTrainer):
    model_type = MLP
    def _build_model(
        hidden_size: int,
        learning_rate: float,
        max_epochs: int,
        seed: int,
    ) -> EncodedMLPClassifier:
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=max_epochs,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=seed,
        )
        return EncodedMLPClassifier(model)


    def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
        """Train a 1-hidden-layer MLP baseline with CV tuning."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        hidden_sizes = (96, 128, 160, 192)
        learning_rates = (0.002, 0.003, 0.004, 0.005)
        max_epochs_values = (100, 200)

        stats = {}
        best = (-1.0, None, None, None, None, None, None)
        trial_index = 0

        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                for max_epochs in max_epochs_values:
                    print(
                        "[tuning] testing "
                        f"hidden_size={hidden_size}, "
                        f"learning_rate={learning_rate}, "
                        f"max_epochs={max_epochs}"
                    )

                    f1s = []
                    accs = []
                    cm = np.zeros((3, 3))
                    fold_epochs = []

                    for train_idx, val_idx in cv.split(x_train, y_train):
                        x_t, x_v = x_train[train_idx], x_train[val_idx]
                        y_t, y_v = y_train[train_idx], y_train[val_idx]

                        model = _build_model(
                            hidden_size=hidden_size,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            seed=seed,
                        )
                        model.fit(x_t, y_t)
                        preds = model.predict(x_v)

                        f1s.append(f1_score(y_v, preds, average="macro"))
                        accs.append(accuracy_score(y_v, preds))
                        cm += confusion_matrix(y_v, preds)
                        fold_epochs.append(int(model.n_iter_))

                    f1 = float(np.mean(f1s))
                    acc = float(np.mean(accs))
                    mean_epochs = float(np.mean(fold_epochs))

                    stats[str(trial_index)] = {
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "max_epochs": max_epochs,
                        "early_stopping": True,
                        "mean_epochs_trained": mean_epochs,
                        "accuracy": acc,
                        "macro_f1": f1,
                        "confusion matrix": str(cm),
                    }

                    if f1 > best[0]:
                        best_model = _build_model(
                            hidden_size=hidden_size,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            seed=seed,
                        )
                        best_model.fit(x_train, y_train)
                        best = (
                            f1,
                            best_model,
                            hidden_size,
                            learning_rate,
                            max_epochs,
                            acc,
                            str(cm),
                        )

                    trial_index += 1

        stats["final"] = {
            "hidden_size": best[2],
            "learning_rate": best[3],
            "max_epochs": best[4],
            "early_stopping": True,
            "epochs_trained": int(best[1].n_iter_) if best[1] is not None else None,
            "accuracy": best[5],
            "macro_f1": best[0],
            "confusion matrix": best[6],
        }
        return best[1], stats