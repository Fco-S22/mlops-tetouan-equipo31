"""
Servicio de inferencia para el modelo ConsumoEnergia_Tetouan usando FastAPI.

Para ejecutar localmente:
    uvicorn api:app --reload
"""

import os
import datetime as dt
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Configuración MLflow
# -------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# IMPORTANTE: ajusta la versión del modelo si corresponde
MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/ConsumoEnergia_Tetouan/16",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as exc:
    raise RuntimeError(
        f"No se pudo cargar el modelo desde MLflow: {MODEL_URI}. "
        f"Detalle: {exc}"
    )


# -------------------------------------------------------------------
# Modelos Pydantic (entrada / salida)
# -------------------------------------------------------------------

class EnergySample(BaseModel):
    # Usamos dt.datetime para evitar choque de nombres
    datetime: dt.datetime = Field(description="Fecha y hora en formato ISO.")
    temperature: float
    humidity: float
    wind_speed: float
    general_diffuse_flows: float
    diffuse_flows: float
    zone_1_power_consumption: float
    zone_2_power_consumption: float
    zone_3_power_consumption: float


class PredictionResponse(BaseModel):
    total_power: List[float]


# -------------------------------------------------------------------
# Construcción del DataFrame de entrada
# -------------------------------------------------------------------
def _build_input_dataframe(samples: List[EnergySample]) -> pd.DataFrame:
    """
    Transforma la lista de muestras Pydantic en un DataFrame con las
    mismas columnas que el modelo vio en el entrenamiento.
    """
    rows = []

    for s in samples:
        data = s.model_dump()
        dt_val: dt.datetime = data.pop("datetime")

        # Derivados temporales
        data["hour"] = dt_val.hour
        data["dayofweek"] = dt_val.weekday()

        rows.append(data)

    df = pd.DataFrame(rows)

    expected_cols = [
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flows",
        "diffuse_flows",
        "zone_1_power_consumption",
        "zone_2_power_consumption",
        "zone_3_power_consumption",
        "hour",
        "dayofweek",
    ]

    missing = set(expected_cols) - set(df.columns)
    for col in missing:
        df[col] = 0.0

    df = df[expected_cols]
    return df


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="Consumo Energia Tetouan - API de Predicción",
    description="API para servir el modelo registrado en MLflow.",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(samples: List[EnergySample]):
    """
    Recibe una lista de muestras (1..N) y devuelve las predicciones de total_power.
    """
    if not samples:
        raise HTTPException(status_code=400, detail="La lista de muestras está vacía.")

    try:
        df = _build_input_dataframe(samples)
        preds = model.predict(df)

        preds_list = [float(p) for p in getattr(preds, "ravel", lambda: preds)().tolist()]

        return PredictionResponse(total_power=preds_list)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        )