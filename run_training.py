# run_training.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_consumo_energia_tetouan.dataset import DataLoader
from mlops_consumo_energia_tetouan.features import FeatureEngineer

# --- Config MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Consumo_Energia_Tetouan")

# --- Config de datos ---
CSV_PATH = "data/processed/power_tetouan_city_after_EDA.csv"
TARGET_COL = "total_power"

# Nombre en el Model Registry (todas las versiones caen bajo este nombre)
REGISTERED_MODEL_NAME = "ConsumoEnergia_Tetouan"


def _safe_git_commit():
    """Obtiene el SHA corto del commit actual para registrarlo como tag (si hay Git)."""
    try:
        return os.popen("git rev-parse --short HEAD").read().strip()
    except Exception:
        return ""


def _compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main():
    # --- Carga y preparación de datos ---
    df = DataLoader(CSV_PATH).load()
    fe = FeatureEngineer(target_col=TARGET_COL)
    X_train, X_test, y_train, y_test = fe.split(df)
    pre = fe.build_preprocessor(X_train)

    # (opcional) Mitigar warning por enteros con nulos
    try:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    except Exception:
        pass

    # --- Modelos a ejecutar ---
    models = {
        "linear_regression_v1": LinearRegression(),
        "ridge_regression_v1": Ridge(alpha=1.0),
        "random_forest_v1": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    os.makedirs("artifacts", exist_ok=True)

    for run_name, estimator in models.items():
        with mlflow.start_run(run_name=run_name):
            # Pipeline: preprocesamiento + modelo
            pipeline = Pipeline(steps=[
                ("preprocessor", pre),
                ("model", estimator)
            ])

            # Entrenamiento
            pipeline.fit(X_train, y_train)

            # Predicción y métricas
            y_pred = pipeline.predict(X_test)
            metrics = _compute_metrics(y_test, y_pred)

            # --- Log de parámetros, métricas y metadatos ---
            est_params = {f"model__{k}": v for k, v in estimator.get_params().items()}
            mlflow.log_params({
                "model": estimator.__class__.__name__,
                "target": TARGET_COL,
                "test_size": fe.test_size,
                "random_state": fe.random_state,
                **est_params
            })
            mlflow.log_metrics(metrics)

            mlflow.set_tags({
                "git_commit": _safe_git_commit(),
                "pipeline": "preprocessor + model",
            })

            # === Resultados relevantes (artefactos) ===
            # 1) Predicciones vs reales
            preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
            preds_path = f"artifacts/{run_name}_predicciones.csv"
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path, artifact_path="analysis")

            # 2) Gráfico de residuos
            resid = y_test - y_pred
            plt.figure()
            plt.scatter(y_pred, resid, alpha=0.5)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Predicción")
            plt.ylabel("Residuo")
            plt.title(f"Residuos - {estimator.__class__.__name__}")
            plt.tight_layout()
            resid_path = f"artifacts/{run_name}_residuos.png"
            plt.savefig(resid_path, dpi=150)
            plt.close()
            mlflow.log_artifact(resid_path, artifact_path="analysis")

            # 3) Importancias de características (si aplica)
            try:
                if hasattr(estimator, "feature_importances_"):
                    # Intentamos obtener nombres de features desde el preprocesador
                    try:
                        feature_names = pre.get_feature_names_out()
                    except Exception:
                        feature_names = [f"f{i}" for i in range(len(estimator.feature_importances_))]

                    imp_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": estimator.feature_importances_
                    })
                    imp_path = f"artifacts/{run_name}_importancias.csv"
                    imp_df.to_csv(imp_path, index=False)
                    mlflow.log_artifact(imp_path, artifact_path="analysis")
            except Exception:
                pass

            # 4) Coeficientes (modelos lineales)
            try:
                if hasattr(estimator, "coef_"):
                    try:
                        feature_names = pre.get_feature_names_out()
                    except Exception:
                        feature_names = [f"f{i}" for i in range(np.ravel(estimator.coef_).shape[0])]

                    coefs = np.ravel(estimator.coef_)
                    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
                    coef_path = f"artifacts/{run_name}_coeficientes.csv"
                    coef_df.to_csv(coef_path, index=False)
                    mlflow.log_artifact(coef_path, artifact_path="analysis")
            except Exception:
                pass
            # === Fin artefactos ===

            # --- Guardar y registrar el modelo en el Model Registry ---
            try:
                input_example = X_test[:5]  # DataFrame o ndarray
            except Exception:
                input_example = None

            # Nota: algunas versiones de MLflow avisan que 'artifact_path' está deprecado.
            # Si te aparece el warning, puedes cambiar 'artifact_path' por 'name'.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                input_example=input_example
            )

            print(f"[OK] {run_name} registrado. Métricas: {metrics}")

    print("\n[OK] Todos los modelos ejecutados y versionados en el Model Registry:")
    print(f"    - {REGISTERED_MODEL_NAME} v1/v2/v3 (según orden de registro).")


if __name__ == "__main__":
    main()