"""
Utilidades de monitoreo para seguimiento de modelos en producción.

Proporciona utilidades para:
- Generar datos sintéticos con drift
- Establecer baseline estadístico y de desempeño
- Comparar rendimiento entre periodos
- Agregar métricas y obtener resúmenes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BaselineEstablisher:
    """Calcula y almacena estadísticas de referencia (baseline) para comparación."""

    def __init__(self, validation_data: pd.DataFrame):
        """
        Inicializa el establecedor de baseline.

        Args:
            validation_data: Conjunto de validación para derivar la baseline
        """
        self.validation_data = validation_data
        self.baseline_stats = {}
        self.baseline_metrics = {}
        
        logger.info(f"Initialized BaselineEstablisher with {len(validation_data)} samples")

    def compute_feature_statistics(
        self,
        numeric_features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas básicas para features numéricos de la baseline.

        Args:
            numeric_features: Lista de features numéricos (se detectan automáticamente si es None)

        Returns:
            Diccionario con estadísticas por feature
        """
        if numeric_features is None:
            numeric_features = self.validation_data.select_dtypes(include=[np.number]).columns.tolist()

        stats_dict = {}

        for feature in numeric_features:
            if feature not in self.validation_data.columns:
                logger.warning(f"Feature {feature} not found")
                continue

            data = self.validation_data[feature].dropna()

            stats_dict[feature] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "median": float(np.median(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "q25": float(np.percentile(data, 25)),
                "q75": float(np.percentile(data, 75)),
                "skewness": float(self._calculate_skewness(data)),
                "kurtosis": float(self._calculate_kurtosis(data))
            }

        self.baseline_stats = stats_dict
        return stats_dict

    def compute_performance_baseline(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcula métricas de desempeño a partir de etiquetas verdaderas y predichas.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones de etiquetas
            y_pred_proba: Predicciones de probabilidad (opcional)

        Returns:
            Diccionario con métricas de baseline
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

        # Add ROC-AUC if binary classification and probabilities available
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        self.baseline_metrics = metrics
        logger.info(f"Computed baseline metrics: {metrics}")
        return metrics

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calcular asimetría (skewness)."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.sum(((data - mean) / std) ** 3) / n)

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calcular curtosis excesiva (kurtosis)."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.sum(((data - mean) / std) ** 4) / n - 3)


class SyntheticDriftGenerator:
    """Generador de escenarios sintéticos de drift para pruebas y simulación."""

    def __init__(self, baseline_data: pd.DataFrame, random_state: int = 42):
        """
        Inicializa el generador de drift.

        Args:
            baseline_data: Datos de referencia (baseline)
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.baseline_data = baseline_data
        self.random_state = random_state
        np.random.seed(random_state)
        
        logger.info(f"Initialized SyntheticDriftGenerator with {len(baseline_data)} baseline samples")

    def create_mean_shift_drift(
        self,
        features: List[str],
        shift_magnitude: float = 0.5,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Genera drift mediante un desplazamiento de la media de las features.

        Args:
            features: Features a desplazar
            shift_magnitude: Magnitud del desplazamiento (en desviaciones estándar)
            sample_size: Tamaño del dataset a generar

        Returns:
            DataFrame con features desplazadas
        """
        drift_data = self.baseline_data.sample(n=sample_size, replace=True, random_state=self.random_state).copy()

        for feature in features:
            if feature not in drift_data.columns:
                logger.warning(f"Feature {feature} not found")
                continue

            # Calcular la magnitud del desplazamiento
            std = drift_data[feature].std()
            shift = shift_magnitude * std
            # Añadir ruido para simular un drift gradual
            noise = np.random.normal(0, std * 0.1, len(drift_data))
            drift_data[feature] = drift_data[feature] + shift + noise

        logger.info(f"Created mean-shift drift with {sample_size} samples")
        return drift_data

    def create_variance_shift_drift(
        self,
        features: List[str],
        variance_multiplier: float = 2.0,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Create drift by changing feature variance.

        Args:
            features: Features to modify
            variance_multiplier: Multiply variance by this factor
            sample_size: Number of samples

        Returns:
            DataFrame with variance-shifted features
        """
        drift_data = self.baseline_data.sample(n=sample_size, replace=True, random_state=self.random_state).copy()


        for feature in features:
            if feature not in drift_data.columns:
                logger.warning(f"Feature {feature} not found")
                continue

            mean = drift_data[feature].mean()
            std = drift_data[feature].std()

            # Crear datos con varianza distinta
            drift_data[feature] = mean + np.random.normal(0, std * np.sqrt(variance_multiplier), len(drift_data))

        logger.info(f"Created variance-shift drift with {sample_size} samples")
        return drift_data

    def create_missing_feature_drift(
        self,
        features: List[str],
        missing_percentage: float = 0.3,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Create drift by introducing missing values.

        Args:
            features: Features to make missing
            missing_percentage: Percentage of values to make missing
            sample_size: Number of samples

        Returns:
            DataFrame with missing values
        """
        drift_data = self.baseline_data.sample(n=sample_size, replace=True, random_state=self.random_state).copy()

        for feature in features:
            if feature not in drift_data.columns:
                logger.warning(f"Feature {feature} not found")
                continue

            # Seleccionar índices aleatorios y poner NaN
            missing_indices = np.random.choice(len(drift_data), int(len(drift_data) * missing_percentage), replace=False)
            drift_data.loc[missing_indices, feature] = np.nan

        logger.info(f"Created missing-value drift with {missing_percentage:.1%} missing values")
        return drift_data

    def create_seasonal_drift(
        self,
        features: List[str],
        amplitude: float = 0.3,
        period: int = 12,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Create drift with seasonal patterns.

        Args:
            features: Features to apply seasonal effect to
            amplitude: Seasonal amplitude (relative to std)
            period: Seasonal period
            sample_size: Number of samples

        Returns:
            DataFrame with seasonal drift
        """
        drift_data = self.baseline_data.sample(n=sample_size, replace=True, random_state=self.random_state).copy()

        for idx, feature in enumerate(features):
            if feature not in drift_data.columns:
                logger.warning(f"Feature {feature} not found")
                continue


            std = drift_data[feature].std()
            # Componente estacional (senoidal) para simular variación temporal
            seasonal_component = amplitude * std * np.sin(2 * np.pi * np.arange(len(drift_data)) / period)
            drift_data[feature] = drift_data[feature] + seasonal_component

        logger.info(f"Drift estacional creado (periodo={period}, amplitud={amplitude})")
        return drift_data

    def create_multimodal_drift(
        self,
        features: List[str],
        sample_size: int = 500,
        n_modes: int = 2
    ) -> pd.DataFrame:
        """
        Create drift with multimodal distribution.

        Args:
            features: Features to apply multimodality to
            sample_size: Number of samples
            n_modes: Number of modes

        Returns:
            DataFrame with multimodal distribution
        """
        # Generate from multiple Gaussian components
        drift_data_parts = []
        samples_per_mode = sample_size // n_modes

        for mode in range(n_modes):
            mode_data = self.baseline_data.sample(
                n=samples_per_mode,
                replace=True,
                random_state=self.random_state + mode
            ).copy()

            for feature in features:
                if feature not in mode_data.columns:
                    continue

                std = mode_data[feature].std()
                mode_data[feature] = mode_data[feature] + (mode - n_modes/2) * std

            drift_data_parts.append(mode_data)

        drift_data = pd.concat(drift_data_parts, ignore_index=True)
        logger.info(f"Created multimodal drift with {n_modes} modes")
        return drift_data

    def create_combined_drift(
        self,
        drift_config: Dict[str, Any],
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Create drift with combination of effects.

        Args:
            drift_config: Dict specifying multiple drift types
                Example: {
                    'mean_shift': {'features': ['feature1'], 'shift_magnitude': 0.5},
                    'variance_shift': {'features': ['feature2'], 'variance_multiplier': 2.0},
                    'missing': {'features': ['feature3'], 'missing_percentage': 0.2}
                }
            sample_size: Number of samples

        Returns:
            DataFrame with combined drift
        """
        drift_data = self.baseline_data.sample(n=sample_size, replace=True, random_state=self.random_state).copy()

        # Aplicar desplazamientos de media
        if 'mean_shift' in drift_config:
            config = drift_config['mean_shift']
            for feature in config.get('features', []):
                if feature not in drift_data.columns:
                    continue
                std = drift_data[feature].std()
                shift = config.get('shift_magnitude', 0.5) * std
                drift_data[feature] = drift_data[feature] + shift

        # Aplicar cambios de varianza
        if 'variance_shift' in drift_config:
            config = drift_config['variance_shift']
            for feature in config.get('features', []):
                if feature not in drift_data.columns:
                    continue
                mean = drift_data[feature].mean()
                std = drift_data[feature].std()
                multiplier = config.get('variance_multiplier', 2.0)
                drift_data[feature] = mean + np.random.normal(0, std * np.sqrt(multiplier), len(drift_data))

        # Aplicar valores faltantes
        if 'missing' in drift_config:
            config = drift_config['missing']
            for feature in config.get('features', []):
                if feature not in drift_data.columns:
                    continue
                missing_pct = config.get('missing_percentage', 0.3)
                missing_indices = np.random.choice(len(drift_data), int(len(drift_data) * missing_pct), replace=False)
                drift_data.loc[missing_indices, feature] = np.nan

        logger.info(f"Drift combinado creado con {len(drift_config)} efectos")
        return drift_data


class PerformanceComparator:
    """Comparador de desempeño del modelo entre distintas distribuciones de datos."""

    def __init__(self, baseline_metrics: Dict[str, float]):
        """
        Inicializa el comparador de desempeño.

        Args:
            baseline_metrics: Métricas de referencia (baseline)
        """
        self.baseline_metrics = baseline_metrics
        self.monitoring_history = []
        
        logger.info(f"Initialized PerformanceComparator with baseline metrics: {baseline_metrics}")

    def compute_monitoring_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcula las métricas del periodo de monitoreo y las compara con la baseline.

        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones de etiquetas
            y_pred_proba: Predicciones de probabilidad (opcional)
            timestamp: Marca temporal para la medición

        Returns:
            Diccionario con métricas calculadas y comparaciones frente a la baseline
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "timestamp": timestamp or datetime.now().isoformat()
        }

        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        # Comparar con la baseline
        metrics["degradation"] = {}
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in metrics:
                degradation = baseline_value - metrics[metric_name]
                degradation_pct = (degradation / baseline_value * 100) if baseline_value != 0 else 0
                metrics["degradation"][metric_name] = {
                    "absolute": degradation,
                    "percentage": degradation_pct
                }

        self.monitoring_history.append(metrics)
        logger.info(f"Computed monitoring metrics at {metrics['timestamp']}")
        return metrics

    def get_degradation_summary(self) -> Dict[str, Any]:
        """Get summary of performance degradation over time."""
        if not self.monitoring_history:
            return {}

        summary = {
            "measurements_count": len(self.monitoring_history),
            "metric_trends": {}
        }

        for metric_name in self.baseline_metrics.keys():
            values = []
            for record in self.monitoring_history:
                if "degradation" in record and metric_name in record["degradation"]:
                    values.append(record["degradation"][metric_name]["absolute"])

            if values:
                summary["metric_trends"][metric_name] = {
                    "mean_degradation": float(np.mean(values)),
                    "max_degradation": float(np.max(values)),
                    "min_degradation": float(np.min(values)),
                    "trend": "worsening" if np.mean(values[-3:]) > np.mean(values[:3]) else "improving"
                }

        return summary
