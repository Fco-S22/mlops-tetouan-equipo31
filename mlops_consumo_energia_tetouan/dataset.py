from pathlib import Path
import pandas as pd

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
