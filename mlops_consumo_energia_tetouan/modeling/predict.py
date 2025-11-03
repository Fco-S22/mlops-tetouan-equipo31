import pandas as pd

class Predictor:
    """
    Envuelve el pipeline entrenado para predicción.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)
