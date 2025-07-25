# models/dummy_model.py

import numpy as np

class DummyModel:
    def predict(self, data):
        batch_size = data.shape[0]
        num_classes = 2  # incendio, inundacion
        predictions = np.random.rand(batch_size, num_classes)
        predictions /= predictions.sum(axis=1, keepdims=True)
        return predictions

