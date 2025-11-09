import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from src.features.feature_engineering import BMICalculator

# Test that transform does not mutate the input DataFrame and preserves dtypes/index.
def test_transform_preserves_input_dataframe():
    """Transform should return a new DataFrame and must not modify the input in-p lace."""
    df = pd.DataFrame({"height": [1.70, 1.80], "weight": [70.0, 80.0]})
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    df_copy = df.copy(deep=True)
    _ = bmi.transform(df)
    pdt.assert_frame_equal(df, df_copy)

# Test NaN propagation: any row with NaN height or weight should yield NaN bmi.
def test_bmi_nan_values_propagate_to_bmi():
    """Rows with missing height or weight must result in NaN BMI; valid rows compute correctly."""
    df = pd.DataFrame({
        "height": [1.70, np.nan, 1.60],
        "weight": [70.0, 60.0, np.nan],
    })
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    # rows with either height or weight NaN should produce NaN bmi
    assert np.isnan(out.loc[1, "bmi"])
    assert np.isnan(out.loc[2, "bmi"])
    # valid row should match expected value within numerical tolerance
    expected = df.loc[0, "weight"] / (df.loc[0, "height"] ** 2)
    assert np.isclose(out.loc[0, "bmi"], expected, atol=1e-9)

# Test that integer inputs are handled and result BMI dtype is float with correct numeric values.
def test_transform_handles_integer_inputs_returns_float_values():
    """Integer input columns should produce float BMI values with correct computation."""
    df = pd.DataFrame({"height": [2, 1], "weight": [8, 3]})  # integer inputs
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    # result dtype should be float
    assert out["bmi"].dtype.kind == "f"
    expected = df["weight"] / (df["height"] ** 2)
    # compare numeric results (no NaNs expected in this case)
    assert np.allclose(out["bmi"].values, expected.values, atol=1e-9)

# Test that computation on a representative sample matches the expected vectorized formula.
def test_bmi_computation_matches_expected_with_sample_fixture(sample_raw_df_m):
    """
    Use the sample_raw_df_m fixture and verify elementwise equality between the transform output
    and the expected weight / height**2 computation. NaNs must compare equal at the same positions.
    """
    df = sample_raw_df_m
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    expected = df["weight"] / (df["height"] ** 2)
    # allow NaNs to compare equal in the same positions
    assert np.allclose(out["bmi"].values, expected.values, atol=1e-9, equal_nan=True)

def test_bmi_height_zero_yields_infinite_value():
    """
    When height is zero the mathematical result is division by zero and NumPy produces
    a positive infinity. Verify this behaviour and also that non-zero rows compute correctly.
    """
    df = pd.DataFrame({"height": [0.0, 1.75], "weight": [70.0, 70.0]})
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    # first row: infinite positive BMI
    assert np.isinf(out.loc[0, "bmi"]) and out.loc[0, "bmi"] > 0
    # second row: numeric correctness
    expected = df.loc[1, "weight"] / (df.loc[1, "height"] ** 2)
    assert np.isclose(out.loc[1, "bmi"], expected, atol=1e-9)


def test_bmi_negative_height_computes_positive_finite_value():
    """
    Negative heights (if present) are squared in the BMI formula, so the result should
    be a finite positive number equal to weight / (height**2).
    """
    df = pd.DataFrame({"height": [-1.70], "weight": [70.0]})
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    expected = df.loc[0, "weight"] / (df.loc[0, "height"] ** 2)
    assert np.isfinite(out.loc[0, "bmi"])
    assert out.loc[0, "bmi"] > 0
    assert np.isclose(out.loc[0, "bmi"], expected, atol=1e-9)


def test_bmi_pickle_roundtrip_preserves_behavior():
    """
    The transformer should be serializable with pickle and behave identically after
    a pickle roundtrip (useful for caching/CI).
    """
    import pickle
    df = pd.DataFrame({"height": [1.70, 1.80], "weight": [70.0, 80.0]})
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    serialized = pickle.dumps(bmi)
    bmi_unpickled = pickle.loads(serialized)
    out1 = bmi.transform(df)
    out2 = bmi_unpickled.transform(df)
    pdt.assert_frame_equal(out1, out2)


def test_bmi_compatible_with_sklearn_pipeline():
    """
    Ensure BMICalculator can be used inside an sklearn.pipeline.Pipeline. If scikit-learn
    is not installed the test is skipped.
    """
    pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    df = pd.DataFrame({"height": [1.70], "weight": [70.0]})
    pipe = Pipeline([("bmi", BMICalculator(output_col="bmi"))])
    result = pipe.fit_transform(df)
    # accept either a DataFrame with the new column or any non-None result
    if isinstance(result, pd.DataFrame):
        assert "bmi" in result.columns
    else:
        assert result is not None


def test_bmi_vectorized_on_large_dataframe_smoke():
    """
    Smoke test to ensure the implementation is vectorized and performs on a larger input.
    This verifies correctness against the vectorized formula using equal_nan comparison.
    """
    rng = np.random.default_rng(0)
    n = 5000
    heights = rng.uniform(0.5, 2.2, size=n)
    weights = rng.uniform(30.0, 150.0, size=n)
    # inject a few NaNs to validate NaN propagation
    heights[10] = np.nan
    weights[20] = np.nan
    df = pd.DataFrame({"height": heights, "weight": weights})
    bmi = BMICalculator(output_col="bmi")
    bmi.fit(df)
    out = bmi.transform(df)
    expected = df["weight"] / (df["height"] ** 2)
    assert np.allclose(out["bmi"].values, expected.values, atol=1e-9, equal_nan=True)