import pandas as pd
import pandas.testing as pdt
import pytest
from pathlib import Path

from src.data.data_loader import CSVDataLoader, DataFrameAnalyzer, VariableManager

pytestmark = pytest.mark.unit


def test_csv_data_loader_reads_csv_file(tmp_path):
    """CSVDataLoader.load should read a CSV and return the expected DataFrame."""
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_csv(csv_path, index=False)

    loader = CSVDataLoader(str(csv_path))
    assert loader.validate_source() is True
    out = loader.load()
    pdt.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_csv_data_loader_raises_on_missing_file():
    """Loading a non-existent CSV should raise FileNotFoundError."""
    loader = CSVDataLoader("no/such/file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_csv_data_loader_propagates_parser_error(monkeypatch, tmp_path):
    """If pandas.read_csv raises a ParserError, the loader should propagate it."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col\n1\n2\n")  # content won't matter; we monkeypatch read_csv

    def _raise(*args, **kwargs):
        raise pd.errors.ParserError("mock parser error")

    monkeypatch.setattr(pd, "read_csv", _raise)
    loader = CSVDataLoader(str(csv_path))
    with pytest.raises(pd.errors.ParserError):
        loader.load()


def test_dataframe_analyzer_basic_info_and_columns():
    """DataFrameAnalyzer.get_basic_info and column selectors should return expected values."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, None],
        "num2": [3, 4, 5],
        "cat": ["a", "b", "a"]
    })
    analyzer = DataFrameAnalyzer(df)
    info = analyzer.get_basic_info()
    assert info["shape"] == df.shape
    assert "num1" in info["columns"] and "cat" in info["columns"]

    numeric = analyzer.get_numeric_columns()
    categorical = analyzer.get_categorical_columns()
    assert set(numeric) >= {"num1", "num2"}
    assert "cat" in categorical


def test_dataframe_analyzer_missing_and_duplicate_summary():
    """Missing values summary and duplicate detection should report correct counts."""
    df = pd.DataFrame({
        "x": [1, None, 1],
        "y": ["a", "b", "a"]
    })
    # create a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    analyzer = DataFrameAnalyzer(df)
    missing = analyzer.get_missing_values_summary()
    assert "x" in missing.index
    dup = analyzer.get_duplicate_summary()
    assert dup["n_duplicates"] >= 1


def test_get_summary_statistics_returns_empty_when_no_numeric_columns():
    """If there are no numeric columns, get_summary_statistics should return an empty DataFrame."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    analyzer = DataFrameAnalyzer(df)
    stats = analyzer.get_summary_statistics()
    assert isinstance(stats, pd.DataFrame)
    assert stats.empty


def test_variable_manager_defaults_and_to_lower():
    """VariableManager should expose default lists and support to_lower conversion."""
    vm = VariableManager()
    all_features = vm.get_all_feature_vars()
    assert isinstance(all_features, list)
    assert vm.DEFAULT_TARGET_VAR in vm.get_all_vars()

    vm_lower = VariableManager(to_lower=True)
    assert all(v.islower() for v in vm_lower.get_all_feature_vars())


def test_variable_manager_exclude_variables_returns_new_manager():
    """exclude_variables should return a new manager without the excluded variables."""
    vm = VariableManager(numeric_vars=["A", "B"], categorical_vars=["C"], target_var="T")
    new_vm = vm.exclude_variables(["B", "C"])
    assert "B" not in new_vm.numeric_vars
    assert "C" not in new_vm.categorical_vars
    # original must remain unchanged
    assert "B" in vm.numeric_vars and "C" in vm.categorical_vars


def test_variable_manager_to_dict_contains_expected_keys():
    vm = VariableManager()
    d = vm.to_dict()
    assert set(d.keys()) >= {"numeric_vars", "categorical_vars", "target_var", "n_numeric", "n_categorical"}