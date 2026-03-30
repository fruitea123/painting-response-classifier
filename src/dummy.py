from src.BaseTrainer import DUMMY, BaseTrainer
from src.PaintingClassifier import PaintingClassifier

class Trainer(BaseTrainer):
    model_type = DUMMY
    def train(x_train, y_train, seed: int = 311):
        model = PaintingClassifier()
        return model, {}