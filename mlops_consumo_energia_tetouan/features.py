# features.py

from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureEngineer:
    """
    Separa X/y y arma un preprocesador automático:

    - Numéricas: StandardScaler
    - Categóricas: OneHotEncoder(handle_unknown="ignore")

    Uso típico:
        fe = FeatureEngineer(target_col="total_power")
        X_train, X_test, y_train, y_test = fe.split(df)
        fe.build_preprocessor(X_train)
        X_train_p = fe.fit_transform(X_train)
        X_test_p = fe.transform(X_test)
    """

    def __init__(self, target_col: str, test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor: ColumnTransformer | None = None
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []

    # --------- División train / test ---------
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Separa el dataset en X/y y luego en train/test."""
        if self.target_col not in df.columns:
            raise ValueError(f"La columna target '{self.target_col}' no está en el DataFrame.")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test

    # --------- Construcción del preprocesador ---------
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Construye y guarda internamente un ColumnTransformer con:
        - StandardScaler para columnas numéricas
        - OneHotEncoder para columnas categóricas

        ADICIONAL:
        - Convierte las columnas categóricas a string para evitar mezcla de tipos
          (int/str) que provoca errores en OneHotEncoder.
        """
        # Detectar columnas numéricas y categóricas
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        # 👇 Paso clave: forzar categóricas a string
        if self.cat_cols:
            X[self.cat_cols] = X[self.cat_cols].astype(str)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
            ],
            remainder="drop",
        )
        return self.preprocessor

    # --------- Métodos de transformación ---------
    def fit_transform(self, X_train: pd.DataFrame):
        """
        Ajusta el preprocesador (si no existe, lo crea) y transforma X_train.
        """
        if self.preprocessor is None:
            self.build_preprocessor(X_train)

        return self.preprocessor.fit_transform(X_train)

    def transform(self, X: pd.DataFrame):
        """
        Solo transforma nuevos datos usando el preprocesador ya ajustado.
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "El preprocesador aún no ha sido ajustado. Llama primero a fit_transform()."
            )
        return self.preprocessor.transform(X)

    # --------- Pipeline completo conveniente ---------
    def prepare(
        self, df: pd.DataFrame
    ) -> Tuple[object, object, pd.Series, pd.Series]:
        """
        Atajo conveniente:
        - Separa train/test
        - Construye y ajusta el preprocesador
        - Devuelve X_train_p, X_test_p, y_train, y_test
        """
        X_train, X_test, y_train, y_test = self.split(df)
        self.build_preprocessor(X_train)
        X_train_p = self.fit_transform(X_train)
        X_test_p = self.transform(X_test)
        return X_train_p, X_test_p, y_train, y_test