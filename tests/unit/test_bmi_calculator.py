import numpy as np
import pandas as pd
import pytest
from src.features import BMICalculator

def test_bmi_fit_requires_cols():
    df = pd.DataFrame({"height": [1.70], "sex": ["M"]})
    bmi = BMICalculator()  # por defecto: weight, height
    with pytest.raises(ValueError):
        bmi.fit(df)

def test_bmi_transform_values(sample_raw_df_m):
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(sample_raw_df_m)
    out = bmi.transform(sample_raw_df_m)

    assert "bmi" in out.columns
    expected = sample_raw_df_m["weight"] / (sample_raw_df_m["height"] ** 2)
    assert np.allclose(out["bmi"].values, expected.values, atol=1e-9)
    # Valores aleatorios pero razonables para la data
    assert out["bmi"].between(10, 80).all()

def test_bmi_custom_columns():
    df = pd.DataFrame({
        "h_m": [1.80, 1.60],
        "w_kg": [90, 52],
    })
    bmi = BMICalculator(weight_col="w_kg", height_col="h_m", output_col="imc")
    bmi.fit(df)
    out = bmi.transform(df)
    assert "imc" in out.columns
