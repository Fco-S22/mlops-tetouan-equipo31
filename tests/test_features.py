import pandas as pd

from mlops_consumo_energia_tetouan.features import FeatureEngineer


def _build_dummy_df():
    """
    Crea un DataFrame pequeño con:
    - columnas numéricas
    - columnas categóricas
    - columna target total_power
    """
    return pd.DataFrame(
        {
            "temperature": [20.0, 21.0, 22.0, 23.0],
            "humidity": [30.0, 35.0, 40.0, 45.0],
            "hour": [0, 1, 2, 3],
            "city": ["T1", "T1", "T2", "T2"],
            "season": ["winter", "winter", "summer", "summer"],
            "total_power": [100.0, 110.0, 120.0, 130.0],
        }
    )


def test_featureengineer_split_returns_correct_shapes():
    df = _build_dummy_df()
    fe = FeatureEngineer(target_col="total_power", test_size=0.25, random_state=42)

    X_train, X_test, y_train, y_test = fe.split(df)

    # 4 registros, test_size=0.25 -> 1 en test, 3 en train
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    # Columnas de X no deben incluir total_power
    assert "total_power" not in X_train.columns
    assert "total_power" not in X_test.columns


def test_featureengineer_build_preprocessor_sets_num_and_cat():
    df = _build_dummy_df().drop(columns=["total_power"])
    fe = FeatureEngineer(target_col="total_power")

    preprocessor = fe.build_preprocessor(df)

    # Debe haber identificado algunas numéricas y algunas categóricas
    assert len(fe.num_cols) > 0
    assert len(fe.cat_cols) > 0

    # El preprocesador debe ser un ColumnTransformer
    from sklearn.compose import ColumnTransformer as CT

    assert isinstance(preprocessor, CT)


def test_featureengineer_fit_transform_and_transform():
    df = _build_dummy_df()
    fe = FeatureEngineer(target_col="total_power", test_size=0.5, random_state=0)

    X_train, X_test, y_train, y_test = fe.split(df)
    fe.build_preprocessor(X_train)

    X_train_p = fe.fit_transform(X_train)
    X_test_p = fe.transform(X_test)

    # Debe devolver arrays (o matrices) con mismo número de filas
    assert X_train_p.shape[0] == X_train.shape[0]
    assert X_test_p.shape[0] == X_test.shape[0]