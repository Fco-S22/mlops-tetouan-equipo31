import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_consumo_energia_tetouan.dataset import DataLoader
from mlops_consumo_energia_tetouan.features import FeatureEngineer


def _make_processed_like_df():
    """
    Crea un DataFrame que se parece al dataset procesado:
    - features numéricas + categóricas
    - columna target total_power
    """
    n = 20
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "temperature": rng.normal(20, 2, size=n),
            "humidity": rng.normal(40, 5, size=n),
            "wind_speed": rng.normal(2, 0.5, size=n),
            "general_diffuse_flows": rng.normal(50, 10, size=n),
            "diffuse_flows": rng.normal(20, 5, size=n),
            "hour": np.arange(n) % 24,
            "dayofweek": np.arange(n) % 7,
            "city": ["Tetouan"] * n,
            "season": ["winter"] * (n // 2) + ["summer"] * (n - n // 2),
            "total_power": rng.normal(100, 15, size=n),
        }
    )


def _compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def test_full_pipeline_integration(tmp_path):
    """
    Prueba de integración end-to-end:
    - Guardar un CSV (carga de datos)
    - Preprocesar con FeatureEngineer
    - Entrenar LinearRegression
    - Obtener métricas
    """
    # 1) Crear y guardar CSV temporal
    df = _make_processed_like_df()
    csv_path = tmp_path / "processed_sample.csv"
    df.to_csv(csv_path, index=False)

    # 2) Carga de datos
    loader = DataLoader(str(csv_path))
    df_loaded = loader.load()

    # 3) Preprocesamiento + split
    fe = FeatureEngineer(target_col="total_power", test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = fe.split(df_loaded)
    fe.build_preprocessor(X_train)

    X_train_p = fe.fit_transform(X_train)
    X_test_p = fe.transform(X_test)

    # 4) Entrenamiento modelo simple
    model = LinearRegression()
    model.fit(X_train_p, y_train)

    # 5) Predicción y métricas
    y_pred = model.predict(X_test_p)
    metrics = _compute_metrics(y_test, y_pred)

    # 6) Asserts básicos de la prueba de integración
    #    (relajados según la instrucción de la fase final)
    # Las métricas deben existir
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}

    # Las métricas deben ser números finitos y positivas donde aplica
    assert np.isfinite(metrics["mae"]) and metrics["mae"] >= 0
    assert np.isfinite(metrics["rmse"]) and metrics["rmse"] >= 0
    assert np.isfinite(metrics["r2"])      # no imponemos rango [-1, 1] para evitar falsos fallos