import pytest
import pandas as pd
from pathlib import Path

@pytest.mark.integration
def test_nhanes_schema_and_sanity():
    """
    Sanity check para NHANES 2017–2018 (height/weight):
    - columnas presentes (height en cm, weight en kg)
    - tipos numéricos
    - rangos razonables tolerando niños y adultos
    - BMI dentro de bandas realistas
    """
    file_path = Path("data/raw/tests-NHANES-2017-2018-height-weight.csv")
    if not file_path.exists():
        pytest.skip(f"{file_path} not found; skipping sanity check.")

    df = pd.read_csv(file_path)

    # ---- Columnas esperadas ----
    cols = {c.lower() for c in df.columns}
    assert "height" in cols and "weight" in cols, f"Columns present: {df.columns.tolist()}"

    # Columnas útiles
    height_cm = df["height"]
    weight_kg = df["weight"]

    # ---- Tipos y faltantes ----
    assert pd.api.types.is_numeric_dtype(height_cm)
    assert pd.api.types.is_numeric_dtype(weight_kg)
    assert height_cm.isna().mean() < 0.02, "Too many NaNs in height"
    assert weight_kg.isna().mean() < 0.02, "Too many NaNs in weight"

    # ---- Tamaño muestral mínimo ----
    assert len(df) >= 1000, f"Unexpectedly small dataset: {len(df)} rows"

    # ---- Chequeos de rangos (tolerantes a niños y adultos) ----
    # Altura en cm: casi todos entre 60 y 220 cm
    prop_height_ok = height_cm.between(60, 220).mean()
    assert prop_height_ok > 0.98, f"Height outliers too many (prop ok={prop_height_ok:.3f})"

    # Peso en kg: casi todos entre 5 y 250 kg (niños incluidos)
    prop_weight_ok = weight_kg.between(5, 250).mean()
    assert prop_weight_ok > 0.99, f"Weight outliers too many (prop ok={prop_weight_ok:.3f})"

    # Medianas razonables (confirmar unidades cm/kg)
    assert height_cm.median() > 100, f"Height median too low for cm: {height_cm.median():.1f}"
    assert 35 <= weight_kg.median() <= 120, f"Weight median out of expected kg range: {weight_kg.median():.1f}"

    # ---- BMI (usando height en m) ----
    bmi = weight_kg / (height_cm / 100.0) ** 2
    # Usamos cuantiles para robustez ante outliers
    q01, q50, q99 = bmi.quantile([0.01, 0.50, 0.99])
    assert 10 <= q01 <= 16,  f"q01(BMI) out of range: {q01:.2f}"
    assert 16 <= q50 <= 35,  f"median(BMI) out of range: {q50:.2f}"
    assert 30 <= q99 <= 60,  f"q99(BMI) out of range: {q99:.2f}"