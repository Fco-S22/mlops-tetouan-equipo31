from pathlib import Path
import pandas as pd
from mlops_consumo_energia_tetouan.config import PROJ_ROOT

class DataLoader:
    """
    Cargador genérico de datasets CSV.
    """
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"[INFO] Cargado {self.data_path} -> {df.shape[0]} filas x {df.shape[1]} columnas")
        return df


# ---------------------------------------------------------------------
# NUEVO: Clase DataPreparer para el pipeline de manipulación y limpieza
# ---------------------------------------------------------------------

RAW_PATH = PROJ_ROOT / "data" / "raw" / "power_tetouan_city_modified.csv"
PROCESSED_PATH = PROJ_ROOT / "data" / "processed" / "power_tetouan_city_after_EDA.csv"

class DataPreparer:
    """
    Pipeline de preparación de datos para la fase final (raw -> processed):
    - Limpieza
    - Transformaciones iniciales
    - Creación de nuevas columnas
    - Guardado del dataset procesado
    """

    def load_raw(self) -> pd.DataFrame:
        """Carga el dataset original desde data/raw."""
        df = pd.read_csv(RAW_PATH)
        print(f"[INFO] Archivo RAW cargado: {RAW_PATH} ({df.shape[0]} filas)")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza y preparación del dataset."""

        # ------------------------
        # 1) Normalizar nombres
        # ------------------------
        df = df.rename(columns=lambda c: c.strip().replace(" ", "_").lower())

        # ------------------------
        # 2) Fecha y derivados
        # ------------------------
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
            df["hour"] = df["datetime"].dt.hour
            df["dayofweek"] = df["datetime"].dt.dayofweek

        # ------------------------
        # 3) Convertir columnas numéricas
        # ------------------------
        numeric_cols = [
            "temperature",
            "humidity",
            "wind_speed",
            "general_diffuse_flows",
            "diffuse_flows",
            "zone_1_power_consumption",
            "zone_2_power_consumption",
            "zone_3_power_consumption",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ------------------------
        # 4) Eliminar columnas inválidas
        # ------------------------
        if "mixed_type_col" in df.columns:
            df = df.drop(columns=["mixed_type_col"])

        # ------------------------
        # 5) Crear total_power
        # ------------------------
        if all(c in df.columns for c in [
            "zone_1_power_consumption",
            "zone_2_power_consumption",
            "zone_3_power_consumption"
        ]):
            df["total_power"] = (
                df["zone_1_power_consumption"]
                + df["zone_2_power_consumption"]
                + df["zone_3_power_consumption"]
            )

        # ------------------------
        # 6) Manejar nulos
        # ------------------------
        cols_to_check = [
            "temperature", "humidity", "wind_speed",
            "general_diffuse_flows", "diffuse_flows", "total_power"
        ]
        cols_to_check = [c for c in cols_to_check if c in df.columns]

        df = df.dropna(subset=cols_to_check)

        df = df.reset_index(drop=True)
        return df

    def save_processed(self, df: pd.DataFrame) -> None:
        """Guarda el dataset procesado en data/processed."""
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"[INFO] Dataset procesado guardado en {PROCESSED_PATH}")

    def run(self) -> None:
        """Ejecución completa: raw -> clean -> processed."""
        print("[INFO] Ejecutando pipeline de preparación de datos...")
        df_raw = self.load_raw()
        df_clean = self.clean(df_raw)
        self.save_processed(df_clean)
        print("[OK] Pipeline completado correctamente.")
