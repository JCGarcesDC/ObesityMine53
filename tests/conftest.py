import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# añade la raíz del repo para poder importar `src`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def sample_raw_df_m():
    """DataFrame de ejemplo usado por las pruebas unitarias."""
    return pd.DataFrame({
        "height": [1.70, np.nan, 1.60],
        "weight": [70.0, 60.0, np.nan],
    })

DATA_DIR = Path(__file__).resolve().parent / "data"

@pytest.fixture
def sample_raw_df_m_from_file():
    path = DATA_DIR / "sample_raw_m.csv"
    return pd.read_csv(path)