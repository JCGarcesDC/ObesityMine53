"""
Paquete de monitoreo para seguimiento de modelos y detección de drift en producción.

Módulos incluidos:
- drift_detection: Algoritmos centrales de detección de drift y monitorización
- monitoring_utils: Utilidades para baseline y generación de drift sintético
- visualizations: Herramientas de visualización e informes
"""

from .drift_detection import (
    DataDriftMonitor,
    DriftDetector,
    KolmogorovSmirnovDetector,
    WassersteinDetector,
    HellingerDetector,
    DriftAlert,
    DriftMetrics
)

from .monitoring_utils import (
    BaselineEstablisher,
    SyntheticDriftGenerator,
    PerformanceComparator
)

from .visualizations import DriftVisualizations

__all__ = [
    "DataDriftMonitor",
    "DriftDetector",
    "KolmogorovSmirnovDetector",
    "WassersteinDetector",
    "HellingerDetector",
    "DriftAlert",
    "DriftMetrics",
    "BaselineEstablisher",
    "SyntheticDriftGenerator",
    "PerformanceComparator",
    "DriftVisualizations"
]
