# mlops_consumo_energia_tetouan/modeling/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


class ModelTrainer:
    """
    Entrena un Pipeline [preprocessor -> model] y calcula métricas.
    """

    def __init__(self, model=None):
        self.model = model or LinearRegression()
        self.pipeline = None

    def fit_eval(self, preprocessor, X_train, y_train, X_test, y_test):
        """
        Ajusta el pipeline, predice en el set de prueba y retorna el pipeline junto a métricas.
        """
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", self.model),
        ])

        # Entrenamiento
        self.pipeline.fit(X_train, y_train)

        # Predicción
        y_pred = self.pipeline.predict(X_test)

        # Métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # ✅ Compatible con todas las versiones de scikit-learn
        # Antes (puede fallar): mean_squared_error(y_test, y_pred, squared=False)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))

        print(f"[METRICS] R2={r2: .3f} | MAE={mae: .3f} | RMSE={rmse: .3f}")
        return self.pipeline, {"r2": r2, "mae": mae, "rmse": rmse}
