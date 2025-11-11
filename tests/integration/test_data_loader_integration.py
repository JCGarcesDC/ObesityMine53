import pytest
import pandas as pd
import pandas.testing as pdt
from pathlib import Path
from src.data.data_loader import CSVDataLoader, DataFrameAnalyzer

@pytest.mark.integration
def test_csvdataloader_loads_project_raw_file_if_present():
    """
    Integration test: use a real project raw CSV if present at data/raw/sample_raw_m.csv.
    The test is skipped when the file is not available.
    """
    project_file = Path.cwd() / "data" / "raw" / "sample_raw_m.csv"
    if not project_file.exists():
        pytest.skip("Project raw data not present; skipping integration test.")

    loader = CSVDataLoader(str(project_file))
    assert loader.validate_source() is True
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0

    analyzer = DataFrameAnalyzer(df)
    report = analyzer.get_basic_info()
    assert report["n_rows"] == df.shape[0]