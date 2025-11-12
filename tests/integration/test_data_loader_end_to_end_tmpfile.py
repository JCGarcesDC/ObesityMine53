import pytest
import pandas as pd
import pandas.testing as pdt
from src.data.data_loader import CSVDataLoader, DataFrameAnalyzer

@pytest.mark.integration
def test_csvdataloader_end_to_end_with_tmpfile(tmp_path):
    """
    End-to-end test writing a CSV to disk (tmp_path) and reading it with the real loader.
    This isolates filesystem effects and does not modify repo files.
    """
    csv = tmp_path / "tests-integration.csv"
    df = pd.DataFrame({"height": [1.7, 1.8], "weight": [70, 80]})
    df.to_csv(csv, index=False)

    loader = CSVDataLoader(str(csv))
    assert loader.validate_source() is True
    out = loader.load()
    pdt.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))

    analyzer = DataFrameAnalyzer(out)
    missing = analyzer.get_missing_values_summary()
    assert missing.empty
    summary = analyzer.get_summary_statistics()
    assert "height" in summary.index