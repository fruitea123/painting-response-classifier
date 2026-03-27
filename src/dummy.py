from src.PaintingClassifier import PaintingClassifier

def train(x_train, y_train, seed: int = 311):
    model = PaintingClassifier()
    return model, {}