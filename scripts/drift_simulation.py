"""
Script de ejemplo: Simulación de Data Drift y Detección de Pérdida de Performance

Este script realiza:
- Carga del dataset `data/obesity_estimation_cleaned.csv` como baseline
- Establece baseline estadístico y de desempeño (entrenando un modelo si no existe uno)
- Genera datasets con drift sintético (desplazamiento de media, missing features, estacionalidad)
- Evalúa métricas en el dataset de monitoring y las compara con la línea base
- Detecta drift por feature y degradación de performance
- Guarda reportes JSON y gráficos en `artefactos/`

Ejecutar: `python scripts/drift_simulation.py`
"""

import os
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Importar utilidades de monitoreo creadas
from src.monitoring import (
    BaselineEstablisher,
    SyntheticDriftGenerator,
    PerformanceComparator,
    DataDriftMonitor,
    DriftVisualizations
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = Path("data/obesity_estimation_cleaned.csv")
MODEL_ARTIFACT = Path("artefactos/pipelines_optimizados.joblib")
OUTPUT_DIR = Path("artefactos/drift_monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    # Normalizar nombres de columna a minúsculas
    df.columns = [c.lower() for c in df.columns]
    return df


def build_simple_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    return clf


def train_or_load_model(X_train, y_train):
    # Intentar cargar artefacto existente
    if MODEL_ARTIFACT.exists():
        try:
            logger.info(f"Cargando modelo desde {MODEL_ARTIFACT}")
            model = joblib.load(MODEL_ARTIFACT)
            return model
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo: {e}")

    # Si no existe, entrenar un pipeline simple de regresión logística
    numeric_features = ['age', 'height', 'weight', 'imc'] if 'imc' in X_train.columns else ['age', 'height', 'weight']
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    model = build_simple_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    # Guardar artefacto para reutilización
    try:
        MODEL_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_ARTIFACT)
        logger.info(f"Modelo entrenado y guardado en {MODEL_ARTIFACT}")
    except Exception as e:
        logger.warning(f"No se pudo guardar el modelo: {e}")

    return model


def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {"accuracy": float(acc), "f1_weighted": float(f1)}, y_pred


def main():
    df = load_data(DATA_PATH)
    logger.info(f"Datos cargados: {df.shape}")

    # Barajar/seleccionar datos (opcional para velocidad)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Target: usar 'nobeyesdad' si existe, si no usar umbral sobre 'imc'
    if 'nobeyesdad' in df.columns:
        target_col = 'nobeyesdad'
        y = df[target_col]
    else:
        target_col = 'imc'
        y = (df['imc'] > 25).astype(int)

    # División simple train/validation para establecer la baseline
    features = [c for c in df.columns if c not in ['nobeyesdad', 'imc']]

    X = df[features].copy()
    y = y.copy()

    # Para simplicidad convertir target a etiquetas numéricas si es necesario
    try:
        y = pd.factorize(y)[0]
    except Exception:
        pass

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Entrenar o cargar modelo
    model = train_or_load_model(X_train, y_train)

    # Calcular métricas baseline
    baseline_metrics, _ = compute_metrics(model, X_val, y_val)
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # Establecer estadísticas de baseline
    baseline_df = X_val.copy()
    baseline_df[target_col] = y_val.values

    establisher = BaselineEstablisher(validation_data=baseline_df)
    baseline_stats = establisher.compute_feature_statistics()

    # Crear datasets sintéticos con drift
    generator = SyntheticDriftGenerator(baseline_data=baseline_df, random_state=7)

    # Desplazamiento de media en 'weight' y 'height'
    drift_mean = generator.create_mean_shift_drift(features=['weight', 'height'], shift_magnitude=1.0, sample_size=len(baseline_df))

    # Drift por valores faltantes: introducir NaN en 'age'
    drift_missing = generator.create_missing_feature_drift(features=['age'], missing_percentage=0.4, sample_size=len(baseline_df))

    # Drift estacional en 'weight'
    drift_seasonal = generator.create_seasonal_drift(features=['weight'], amplitude=0.6, period=12, sample_size=len(baseline_df))

    # Seleccionar un escenario de monitoreo para evaluar
    monitoring_df = drift_mean.copy()
    monitoring_df[target_col] = y_val.values  # mantener las etiquetas para evaluación

    # Evaluar sobre el dataset de monitoreo
    monitoring_metrics, y_pred_monitor = compute_metrics(model, monitoring_df.drop(columns=[target_col]), monitoring_df[target_col])
    logger.info(f"Monitoring metrics (mean-shift): {monitoring_metrics}")

    # Detectar drift
    drift_monitor = DataDriftMonitor()
    numeric_features = [f for f in baseline_df.select_dtypes(include=[np.number]).columns if f != target_col]
    categorical_features = [f for f in baseline_df.select_dtypes(include=['object', 'category']).columns]

    drift_results, alerts = drift_monitor.detect_feature_drift(baseline_df, monitoring_df, numeric_features=numeric_features, categorical_features=categorical_features)

    perf_drift, perf_alert = drift_monitor.detect_performance_drift(
        baseline_metrics={'f1_score': baseline_metrics.get('f1_weighted', baseline_metrics.get('f1', 0.0))},
        monitoring_metrics={'f1_score': monitoring_metrics.get('f1_weighted', monitoring_metrics.get('f1', 0.0))},
        metric_name='f1_score'
    )

    # Guardar reporte
    report = {
        'baseline_metrics': baseline_metrics,
        'monitoring_metrics': monitoring_metrics,
        'feature_drift_count': len([r for r in drift_results if r.drift_detected]),
        'alerts': [a.to_dict() for a in alerts],
        'performance_alert': perf_alert.to_dict() if perf_alert is not None else None
    }

    with open(OUTPUT_DIR / 'drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {OUTPUT_DIR / 'drift_report.json'}")

    # Visualizaciones
    viz = DriftVisualizations()
    # Graficar comparación de la primera feature numérica como ejemplo
    if numeric_features:
        feat = numeric_features[0]
        plot_path = OUTPUT_DIR / f'dist_{feat}.png'
        viz.plot_feature_distribution_comparison(baseline_df, monitoring_df, feat, output_path=plot_path)

    # Gráfico de desempeño
    perf_plot = OUTPUT_DIR / 'performance_comparison.png'
    viz.plot_performance_metrics(baseline_metrics, monitoring_metrics, output_path=perf_plot)

    logger.info('Drift simulation completed')


if __name__ == '__main__':
    main()
