import numpy as np

class PaintingClassifier():
    def __init__(self, classes, coef, intercept):
        self.classes = np.asarray(classes)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)

        if self.coef.ndim != 2:
            raise ValueError(f"coef must be 2D, got shape {self.coef.shape}")
        if self.intercept.shape != (self.coef.shape[0],):
            raise ValueError(
                "intercept must have one value per class. "
                f"Expected {(self.coef.shape[0],)}, got {self.intercept.shape}"
            )
        if self.classes.shape != (self.coef.shape[0],):
            raise ValueError(
                "classes must have one label per class row. "
                f"Expected {(self.coef.shape[0],)}, got {self.classes.shape}"
            )

    @classmethod
    def from_artifact(cls, artifact: dict):
        return cls(
            classes=artifact["classes"],
            coef=artifact["coef"],
            intercept=artifact["intercept"],
        )

    def decision_function(self, data):
        x = np.asarray(data, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {x.shape}")
        if x.shape[1] != self.coef.shape[1]:
            raise ValueError(
                "Feature dimension mismatch. "
                f"Expected {self.coef.shape[1]}, got {x.shape[1]}"
            )
        return x @ self.coef.T + self.intercept

    def predict(self, data):
        scores = self.decision_function(data)
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]
