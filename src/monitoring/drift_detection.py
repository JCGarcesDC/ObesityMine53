"""
Módulo de Detección de Data Drift para Monitoreo en Producción.

Este módulo ofrece capacidades para detectar drift de forma completa, incluyendo:
- Detección estadística de drift (Kolmogorov-Smirnov, Wasserstein, Hellinger)
- Monitoreo de distribuciones por feature
- Detección de degradación de desempeño
- Mecanismos de alertas e informes

Responsables:
- Data Scientist: Usa para monitorizar el modelo en producción
- ML Engineer: Implementa las estrategias de detección
- DevOps: Integra en las canalizaciones de monitoreo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alerta que se genera cuando se detecta drift.

    Atributos principales:
    - timestamp: marca temporal de la alerta
    - feature_name: nombre de la característica afectada
    - drift_type: tipo de drift ('statistical', 'performance', 'feature_missing')
    - severity: severidad ('low','medium','high','critical')
    - drift_score: puntaje que mide la magnitud del drift
    - threshold: umbral usado para disparar la alerta
    - suggested_action: acción recomendada para el equipo
    - details: metadatos adicionales
    """
    timestamp: str
    feature_name: str
    drift_type: str  # 'statistical', 'performance', 'feature_missing'
    severity: str  # 'low', 'medium', 'high', 'critical'
    drift_score: float
    threshold: float
    suggested_action: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convertir a diccionario."""
        return asdict(self)

    def __str__(self) -> str:
        """Representación en cadena para logging."""
        return (
            f"[{self.severity.upper()}] {self.drift_type} drift en '{self.feature_name}' "
            f"(score: {self.drift_score:.4f}, umbral: {self.threshold:.4f})\n"
            f"Acción: {self.suggested_action}"
        )


@dataclass
class DriftMetrics:
    """Contenedor para métricas resultantes de la detección de drift."""
    feature_name: str
    test_statistic: float
    p_value: float
    drift_detected: bool
    drift_score: float
    baseline_distribution: Optional[Dict[str, float]] = None
    monitoring_distribution: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convertir a diccionario."""
        return asdict(self)


class DriftDetector(ABC):
    """Clase base abstracta para estrategias de detección de drift."""

    def __init__(self, name: str, threshold: float = 0.05):
        """
        Inicializa el detector de drift.

        Args:
            name: Nombre del detector
            threshold: Umbral de significancia para considerar drift
        """
        self.name = name
        self.threshold = threshold
        logger.info(f"Initialized {name} with threshold={threshold}")

    @abstractmethod
    def detect(
        self,
        baseline: np.ndarray,
        monitoring: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftMetrics:
        """
        Detectar drift entre la distribución baseline y la de monitoreo.

        Args:
            baseline: Datos de referencia (baseline)
            monitoring: Datos de periodo de monitoreo
            feature_name: Nombre de la característica (para logs)

        Returns:
            DriftMetrics: Resultado de la detección
        """
        pass


class KolmogorovSmirnovDetector(DriftDetector):
    """Detector basado en la prueba Kolmogorov-Smirnov para drift."""

    def __init__(self, threshold: float = 0.05):
        """Initialize KS detector."""
        super().__init__("Kolmogorov-Smirnov", threshold)

    def detect(
        self,
        baseline: np.ndarray,
        monitoring: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftMetrics:
        """
        Ejecuta la prueba Kolmogorov-Smirnov.

        Adecuada para: features continuos y detección general de shift.
        """
        # Eliminar valores NaN
        baseline = baseline[~np.isnan(baseline)]
        monitoring = monitoring[~np.isnan(monitoring)]

        if len(baseline) == 0 or len(monitoring) == 0:
            logger.warning(f"Datos vacíos para {feature_name}")
            return DriftMetrics(
                feature_name=feature_name,
                test_statistic=np.nan,
                p_value=1.0,
                drift_detected=False,
                drift_score=0.0,
                metadata={"error": "Empty data"}
            )

        statistic, p_value = stats.ks_2samp(baseline, monitoring)

        return DriftMetrics(
            feature_name=feature_name,
            test_statistic=statistic,
            p_value=p_value,
            drift_detected=p_value < self.threshold,
            drift_score=statistic,
            baseline_distribution={
                "mean": float(np.mean(baseline)),
                "std": float(np.std(baseline)),
                "min": float(np.min(baseline)),
                "max": float(np.max(baseline))
            },
            monitoring_distribution={
                "mean": float(np.mean(monitoring)),
                "std": float(np.std(monitoring)),
                "min": float(np.min(monitoring)),
                "max": float(np.max(monitoring))
            },
            metadata={"method": "ks_2samp"}
        )


class WassersteinDetector(DriftDetector):
    """Detector basado en la distancia de Wasserstein (Earth Mover)."""

    def __init__(self, threshold: float = 0.5):
        """Initialize Wasserstein detector."""
        super().__init__("Wasserstein", threshold)

    def detect(
        self,
        baseline: np.ndarray,
        monitoring: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftMetrics:
        """
        Calcula la distancia de Wasserstein.

        Útil para medir la magnitud del cambio entre distribuciones.
        """
        baseline = baseline[~np.isnan(baseline)]
        monitoring = monitoring[~np.isnan(monitoring)]

        if len(baseline) == 0 or len(monitoring) == 0:
            return DriftMetrics(
                feature_name=feature_name,
                test_statistic=np.nan,
                p_value=1.0,
                drift_detected=False,
                drift_score=0.0
            )

        distance = wasserstein_distance(baseline, monitoring)

        return DriftMetrics(
            feature_name=feature_name,
            test_statistic=distance,
            p_value=1.0,  # Wasserstein doesn't have p-value
            drift_detected=distance > self.threshold,
            drift_score=distance,
            metadata={"method": "wasserstein_distance"}
        )


class HellingerDetector(DriftDetector):
    """Detector basado en la distancia de Hellinger."""

    def __init__(self, threshold: float = 0.1, bins: int = 30):
        """
        Inicializa el detector de Hellinger.

        Args:
            threshold: Umbral para la distancia de Hellinger
            bins: Número de bins para histogramas
        """
        super().__init__("Hellinger", threshold)
        self.bins = bins

    def detect(
        self,
        baseline: np.ndarray,
        monitoring: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftMetrics:
        """Calcula la distancia de Hellinger entre dos distribuciones."""
        baseline = baseline[~np.isnan(baseline)]
        monitoring = monitoring[~np.isnan(monitoring)]

        if len(baseline) == 0 or len(monitoring) == 0:
            return DriftMetrics(
                feature_name=feature_name,
                test_statistic=np.nan,
                p_value=1.0,
                drift_detected=False,
                drift_score=0.0
            )

        # Construir histogramas
        min_val = min(baseline.min(), monitoring.min())
        max_val = max(baseline.max(), monitoring.max())

        hist_baseline, _ = np.histogram(baseline, bins=self.bins, range=(min_val, max_val))
        hist_monitoring, _ = np.histogram(monitoring, bins=self.bins, range=(min_val, max_val))

        # Normalize
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_monitoring = hist_monitoring / hist_monitoring.sum()

        # Hellinger distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(hist_baseline) - np.sqrt(hist_monitoring)) ** 2))

        return DriftMetrics(
            feature_name=feature_name,
            test_statistic=hellinger,
            p_value=1.0,
            drift_detected=hellinger > self.threshold,
            drift_score=hellinger,
            metadata={"method": "hellinger_distance", "bins": self.bins}
        )


class DataDriftMonitor:
    """
    Sistema integral de monitoreo de data drift.

    Monitorea tanto drift de datos como degradación de desempeño del modelo.
    """

    def __init__(
        self,
        detectors: Optional[List[DriftDetector]] = None,
        performance_threshold: float = 0.05
    ):
        """
        Inicializa el monitor de drift.

        Args:
            detectors: Lista de detectores a utilizar
            performance_threshold: Umbral absoluto de degradación de desempeño
        """
        self.detectors = detectors or [
            KolmogorovSmirnovDetector(threshold=0.05),
            WassersteinDetector(threshold=0.3),
            HellingerDetector(threshold=0.1)
        ]
        self.performance_threshold = performance_threshold
        self.alerts: List[DriftAlert] = []
        self.drift_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized DataDriftMonitor with {len(self.detectors)} detectors")

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        monitoring_df: pd.DataFrame,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Tuple[List[DriftMetrics], List[DriftAlert]]:
        """
        Detect drift in features.

        Args:
            baseline_df: Baseline validation data
            monitoring_df: Monitoring period data
            numeric_features: List of numeric features to check
            categorical_features: List of categorical features to check

        Returns:
            Tuple of (drift_metrics, alerts)
        """
        drift_results = []
        new_alerts = []

        # Detectar automáticamente features si no se proveen
        if numeric_features is None:
            numeric_features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_features is None:
            categorical_features = baseline_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Revisar features numéricos
        for feature in numeric_features:
            if feature not in baseline_df.columns or feature not in monitoring_df.columns:
                logger.warning(f"La feature {feature} falta en los datos")
                continue

            baseline_data = baseline_df[feature].values
            monitoring_data = monitoring_df[feature].values

            # Ejecutar todos los detectores
            for detector in self.detectors:
                metrics = detector.detect(baseline_data, monitoring_data, feature)
                drift_results.append(metrics)

                # Generate alert if drift detected
                if metrics.drift_detected:
                    alert = self._create_drift_alert(
                        feature=feature,
                        metrics=metrics,
                        detector_name=detector.name,
                        drift_type="statistical"
                    )
                    new_alerts.append(alert)

        # Revisar features categóricos
        for feature in categorical_features:
            if feature not in baseline_df.columns or feature not in monitoring_df.columns:
                logger.warning(f"La feature {feature} falta en los datos")
                continue

            # Prueba chi-cuadrado para variables categóricas
            baseline_counts = baseline_df[feature].value_counts()
            monitoring_counts = monitoring_df[feature].value_counts()

            # Alinear índices
            all_categories = set(baseline_counts.index) | set(monitoring_counts.index)
            baseline_counts = baseline_counts.reindex(all_categories, fill_value=0)
            monitoring_counts = monitoring_counts.reindex(all_categories, fill_value=0)

            # Prueba chi-cuadrado
            chi2, p_value = stats.chisquare(monitoring_counts, baseline_counts)

            metrics = DriftMetrics(
                feature_name=feature,
                test_statistic=chi2,
                p_value=p_value,
                drift_detected=p_value < 0.05,
                drift_score=chi2,
                metadata={"method": "chi_square", "feature_type": "categorical"}
            )
            drift_results.append(metrics)

            if metrics.drift_detected:
                alert = self._create_drift_alert(
                    feature=feature,
                    metrics=metrics,
                    detector_name="Chi-Square",
                    drift_type="statistical"
                )
                new_alerts.append(alert)

        # Detectar features que faltan en el dataset de monitoreo
        missing_in_monitoring = set(baseline_df.columns) - set(monitoring_df.columns)
        for feature in missing_in_monitoring:
            alert = DriftAlert(
                timestamp=datetime.now().isoformat(),
                feature_name=feature,
                drift_type="feature_missing",
                severity="high",
                drift_score=1.0,
                threshold=0.0,
                suggested_action=f"La feature '{feature}' falta en los datos de monitoreo. Verifique el pipeline de datos.",
                details={"missing_in": "monitoring_data"}
            )
            new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "drift_detected": len(new_alerts) > 0,
            "alerts_count": len(new_alerts),
            "features_checked": len(numeric_features) + len(categorical_features)
        })

        return drift_results, new_alerts

    def detect_performance_drift(
        self,
        baseline_metrics: Dict[str, float],
        monitoring_metrics: Dict[str, float],
        metric_name: str = "f1_score"
    ) -> Tuple[bool, DriftAlert]:
        """
        Detect performance degradation.

        Args:
            baseline_metrics: Baseline model metrics
            monitoring_metrics: Current monitoring metrics
            metric_name: Metric to monitor (e.g., 'f1_score', 'accuracy')

        Returns:
            Tuple of (drift_detected, alert)
        """
        if metric_name not in baseline_metrics or metric_name not in monitoring_metrics:
            logger.warning(f"Metric {metric_name} not found in baseline or monitoring metrics")
            return False, None

        baseline_value = baseline_metrics[metric_name]
        monitoring_value = monitoring_metrics[metric_name]
        degradation = baseline_value - monitoring_value

        drift_detected = degradation > self.performance_threshold
        severity = self._calculate_severity(degradation)

        alert = DriftAlert(
            timestamp=datetime.now().isoformat(),
            feature_name=metric_name,
            drift_type="performance",
            severity=severity,
            drift_score=degradation,
            threshold=self.performance_threshold,
            suggested_action=self._get_suggested_action(degradation, metric_name),
            details={
                "baseline_value": baseline_value,
                "monitoring_value": monitoring_value,
                "degradation_pct": (degradation / baseline_value * 100) if baseline_value != 0 else 0
            }
        )

        if drift_detected:
            self.alerts.append(alert)

        return drift_detected, alert

    def detect_seasonal_drift(
        self,
        baseline_df: pd.DataFrame,
        monitoring_df: pd.DataFrame,
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect seasonal or temporal drift patterns.

        Args:
            baseline_df: Baseline data
            monitoring_df: Monitoring data
            date_col: Column with dates for temporal analysis

        Returns:
            Dict with seasonal drift analysis
        """
        results = {
            "seasonal_drift_detected": False,
            "anomalies": [],
            "recommendations": []
        }

        # Check for temporal patterns (if date column provided)
        if date_col and date_col in baseline_df.columns:
            # This is a simplified check - in production, use more sophisticated methods
            baseline_size = len(baseline_df)
            monitoring_size = len(monitoring_df)
            size_ratio = monitoring_size / baseline_size if baseline_size > 0 else 1

            if size_ratio < 0.5:
                results["seasonal_drift_detected"] = True
                results["anomalies"].append({
                    "type": "volume_anomaly",
                    "description": f"Monitoring volume is {size_ratio:.1%} of baseline",
                    "recommendation": "Check if this is expected seasonal variation"
                })

        return results

    def _create_drift_alert(
        self,
        feature: str,
        metrics: DriftMetrics,
        detector_name: str,
        drift_type: str
    ) -> DriftAlert:
        """Create an alert for detected drift."""
        severity = self._calculate_severity(metrics.drift_score)

        suggested_action = self._get_suggested_action_for_feature(
            feature, metrics.drift_score, drift_type
        )

        return DriftAlert(
            timestamp=datetime.now().isoformat(),
            feature_name=feature,
            drift_type=drift_type,
            severity=severity,
            drift_score=metrics.drift_score,
            threshold=metrics.test_statistic,
            suggested_action=suggested_action,
            details={
                "detector": detector_name,
                "p_value": metrics.p_value,
                "baseline": metrics.baseline_distribution,
                "monitoring": metrics.monitoring_distribution
            }
        )

    @staticmethod
    def _calculate_severity(drift_score: float) -> str:
        """Calculate alert severity based on drift score."""
        if drift_score > 1.5:
            return "critical"
        elif drift_score > 1.0:
            return "high"
        elif drift_score > 0.5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _get_suggested_action(degradation: float, metric_name: str) -> str:
        """Get suggested action for performance drift."""
        if degradation > 0.15:
            return (
                f"CRITICAL: {metric_name} degraded by {degradation:.1%}. "
                "Recommend immediate model retraining with recent data."
            )
        elif degradation > 0.10:
            return (
                f"HIGH: {metric_name} degraded by {degradation:.1%}. "
                "Monitor closely and prepare retraining."
            )
        elif degradation > 0.05:
            return (
                f"MEDIUM: {metric_name} degraded by {degradation:.1%}. "
                "Review data quality and feature distributions."
            )
        else:
            return (
                f"LOW: Minor degradation detected ({degradation:.1%}). "
                "Continue monitoring."
            )

    @staticmethod
    def _get_suggested_action_for_feature(
        feature: str,
        drift_score: float,
        drift_type: str
    ) -> str:
        """Get suggested action for feature drift."""
        if drift_type == "feature_missing":
            return f"Check data pipeline - feature '{feature}' is missing"

        if drift_score > 1.5:
            return (
                f"CRITICAL drift in '{feature}'. "
                f"Recommend: (1) Verify data source quality, "
                f"(2) Check for upstream data changes, (3) Consider feature re-engineering"
            )
        elif drift_score > 1.0:
            return (
                f"HIGH drift in '{feature}'. "
                f"Recommend: Review feature pipeline, check for seasonal patterns"
            )
        elif drift_score > 0.5:
            return (
                f"MEDIUM drift in '{feature}'. "
                f"Recommend: Monitor distribution, prepare for model updates"
            )
        else:
            return f"Continue monitoring '{feature}' for further changes"

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift monitoring report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.severity == "critical"),
            "high_alerts": sum(1 for a in self.alerts if a.severity == "high"),
            "medium_alerts": sum(1 for a in self.alerts if a.severity == "medium"),
            "low_alerts": sum(1 for a in self.alerts if a.severity == "low"),
            "alerts": [a.to_dict() for a in self.alerts],
            "history": self.drift_history
        }

    def save_report(self, output_path: Path) -> None:
        """Save monitoring report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        report = self.generate_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self.alerts.clear()
        logger.info("Alerts cleared")
