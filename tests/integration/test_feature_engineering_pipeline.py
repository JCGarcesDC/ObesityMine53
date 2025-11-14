import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.feature_engineering import FeatureEngineeringPipeline, BMICalculator


class AddColumnTransformer:
    """Transformer simple para tests: añade una columna calculada a partir de cada fila."""
    def __init__(self, col_name, func):
        self.col_name = col_name
        self.func = func
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col=None):
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.col_name] = out.apply(self.func, axis=1)
        return out


def test_pipeline_applies_transformers_in_sequence_and_produces_expected_columns():
    """Verifica que los transformadores se aplican en orden y las columnas finales contienen los valores esperados."""
    df = pd.DataFrame({"height": [1.70, 1.80], "weight": [70.0, 80.0]})
    bmi = BMICalculator(output_col="bmi")
    add = AddColumnTransformer("bmi_plus_one", lambda row: row["bmi"] + 1)
    pipeline = FeatureEngineeringPipeline([bmi, add])

    pipeline.fit(df)
    out = pipeline.transform(df)

    expected_bmi = df["weight"] / (df["height"] ** 2)
    assert "bmi" in out.columns and "bmi_plus_one" in out.columns
    assert np.allclose(out["bmi"].values, expected_bmi.values, equal_nan=True)
    assert np.allclose(out["bmi_plus_one"].values, (expected_bmi + 1).values, equal_nan=True)


def test_transform_raises_if_pipeline_not_fitted():
    """Transformar sin fit debe fallar con ValueError según la implementación."""
    df = pd.DataFrame({"height": [1.70], "weight": [70.0]})
    pipeline = FeatureEngineeringPipeline([BMICalculator(output_col="bmi")])
    with pytest.raises(ValueError):
        pipeline.transform(df)


def test_fit_transform_equals_fit_then_transform_and_preserves_input_dataframe():
    """fit_transform debe ser equivalente a fit() seguido de transform() y no mutar el DataFrame de entrada."""
    df = pd.DataFrame({"height": [1.70, 1.80], "weight": [70.0, 80.0]})
    bmi = BMICalculator(output_col="bmi")
    add = AddColumnTransformer("bmi_plus_one", lambda row: row["bmi"] + 1)
    pipeline = FeatureEngineeringPipeline([bmi, add])

    df_copy = df.copy(deep=True)
    out_ft = pipeline.fit_transform(df)
    # new pipeline to compare behaviour of fit() then transform()
    pipeline2 = FeatureEngineeringPipeline([BMICalculator(output_col="bmi"), add])
    pipeline2.fit(df)
    out_f_then_t = pipeline2.transform(df)

    pdt.assert_frame_equal(out_ft, out_f_then_t)
    pdt.assert_frame_equal(df, df_copy)  # input no debe cambiar


def test_pipeline_calls_fit_on_inner_transformers():
    """Comprueba que pipeline.fit() llama a fit() de cada transformer interno."""
    class SpyTransformer:
        def __init__(self):
            self.fitted = False

        def fit(self, df, target_col=None):
            self.fitted = True
            return self

        def transform(self, df):
            return df.copy()

    spy = SpyTransformer()
    pipeline = FeatureEngineeringPipeline([spy])
    df = pd.DataFrame({"height": [1.0], "weight": [1.0]})
    pipeline.fit(df)
    assert getattr(spy, "fitted", False) is True