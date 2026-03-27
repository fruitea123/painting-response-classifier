import numpy as np

class PaintingClassifier():
    def __init__(self):
        pass

    def predict(self, data):
        classes = ["The Persistence of Memory", "The Starry Night", "The Water Lily Pond"]
        prediction = np.random.choice(classes, size=data.shape[0])
        
        # return the prediction
        return prediction