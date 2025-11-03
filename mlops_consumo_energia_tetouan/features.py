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
    """
    def __init__(self, target_col: str, test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )
        return self.preprocessor
