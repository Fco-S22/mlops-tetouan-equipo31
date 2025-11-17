import pandas as pd

from mlops_consumo_energia_tetouan.dataset import DataLoader, DataPreparer


def test_dataloader_loads_csv(tmp_path):
    """
    DataLoader debe cargar un CSV y devolver un DataFrame con la forma correcta.
    """
    # Arrange: crear un CSV pequeño temporal
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    csv_path = tmp_path / "dummy.csv"
    data.to_csv(csv_path, index=False)

    # Act
    loader = DataLoader(str(csv_path))
    df_loaded = loader.load()

    # Assert
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (3, 2)
    assert list(df_loaded.columns) == ["a", "b"]


def test_datapreparer_creates_total_power_and_cleans():
    """
    DataPreparer.clean debe:
    - normalizar nombres de columnas
    - crear total_power a partir de columnas de zona
    - eliminar filas con nulos en columnas críticas
    """
    # Arrange: DF que simula datos RAW mínimos
    raw = pd.DataFrame(
        {
            "DateTime": ["01-01-2017 00:00", "01-01-2017 01:00", "01-01-2017 02:00"],
            "Temperature": [20.5, 21.0, None],
            "Humidity": [30.0, 40.0, 50.0],
            "Wind Speed": [1.0, 2.0, 3.0],
            "General Diffuse Flows": [10.0, 20.0, 30.0],
            "Diffuse Flows": [5.0, 6.0, 7.0],
            "Zone 1 Power Consumption": [100.0, 150.0, 200.0],
            "Zone 2 Power Consumption": [80.0, 90.0, 100.0],
            "Zone 3 Power Consumption": [60.0, 70.0, 80.0],
        }
    )

    preparer = DataPreparer()

    # Act
    df_clean = preparer.clean(raw)

    # Assert
    # Debe existir total_power
    assert "total_power" in df_clean.columns

    # Todas las columnas críticas deben estar sin nulos
    for col in [
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flows",
        "diffuse_flows",
        "total_power",
    ]:
        assert df_clean[col].isna().sum() == 0

    # Chequear que total_power sea suma de las 3 zonas en la primera fila
    first_row = df_clean.iloc[0]
    expected_tp = (
        first_row["zone_1_power_consumption"]
        + first_row["zone_2_power_consumption"]
        + first_row["zone_3_power_consumption"]
    )
    assert first_row["total_power"] == expected_tp