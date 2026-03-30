from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from src.BaseTrainer import LOGREG, BaseTrainer

class Trainer(BaseTrainer):
    model_type = LOGREG

    def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
        """Train a simple multinomial logistic regression baseline."""
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(x_train, y_train)
        return model, {}