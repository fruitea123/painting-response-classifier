from sklearn.base import BaseEstimator

# model types
DUMMY = "dummy"
LOGREG = "logreg"
MNB = "mnb"
MLP = "mlp"

class BaseTrainer:
    def __init__(self):
        pass

    def train(x_train, y_train, seed: int = 311) -> tuple[BaseEstimator, dict]:
        raise NotImplementedError("method 'train' not implemented for current trainer")

    def extract_artifact_state(model):
        # TODO: Not implemented
        print("[trainer] WARNING: method 'extract_artifact_state' not implemented for this model class")
        return None

    def build_sklearn_reference_model(artifact: dict) -> BaseEstimator:
        # TODO: not implemented
        print("[trainer] WARNING: method 'build_sklearn_reference_model' not implemented for this model class")
        return None