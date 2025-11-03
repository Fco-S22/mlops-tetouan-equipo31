# run_training.py

import os
import mlflow
import mlflow.sklearn

from mlops_consumo_energia_tetouan.dataset import DataLoader
from mlops_consumo_energia_tetouan.features import FeatureEngineer
from mlops_consumo_energia_tetouan.modeling.train import ModelTrainer

# --- Config MLflow ---
# Apunta a tu servidor local (según tu UI en http://127.0.0.1:5000)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Debe coincidir EXACTO con el nombre del experimento en la UI
mlflow.set_experiment("Consumo_Energia_Tetouan")

# --- Config de datos ---
CSV_PATH = "data/processed/power_tetouan_city_after_EDA.csv"
TARGET_COL = "total_power"


def _safe_git_commit():
    """Obtiene el SHA corto del commit actual para registrarlo como tag (si hay Git)."""
    try:
        return os.popen("git rev-parse --short HEAD").read().strip()
    except Exception:
        return ""


def main():
    # Se inicia un "run" o ejecución registrada en MLflow
    with mlflow.start_run(run_name="linear_regression_v1"):
        # --- Carga y preparación de datos ---
        df = DataLoader(CSV_PATH).load()

        fe = FeatureEngineer(target_col=TARGET_COL)
        X_train, X_test, y_train, y_test = fe.split(df)
        pre = fe.build_preprocessor(X_train)

        # --- Entrenamiento y evaluación ---
        trainer = ModelTrainer()
        pipeline, metrics = trainer.fit_eval(pre, X_train, y_train, X_test, y_test)

        # --- Registrar parámetros, métricas y metadatos ---
        mlflow.log_params({
            "model": "LinearRegression",
            "target": TARGET_COL,
            "test_size": fe.test_size,
            "random_state": fe.random_state
        })

        # Asegura que las métricas sean float (por si vienen como numpy types)
        metrics_to_log = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(metrics_to_log)

        # Tags útiles para reproducibilidad
        mlflow.set_tags({
            "git_commit": _safe_git_commit(),
            # Rellena si usas DVC y quieres enlazar el dataset:
            # "dataset_dvc_hash": "<HASH_DVC>"
        })

        # --- Guardar y registrar el modelo en el Model Registry ---
        # Se publica la versión bajo el nombre "ConsumoEnergia_Tetouan"
        input_example = X_test[:5]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="ConsumoEnergia_Tetouan",
            input_example=input_example
        )

        print("[OK] Entrenamiento registrado en MLflow. Métricas:", metrics_to_log)


if __name__ == "__main__":
    main()