"""
Utilidades de visualización para el monitoreo de data drift.

Provee gráficos para:
- Comparación de distribuciones por feature
- Evolución de métricas de desempeño
- Severidad de alertas de drift
- Comparación baseline vs monitoreo
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DriftVisualizations:
    """Generador de visualizaciones para el análisis de drift y monitoreo."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Inicializa el generador de visualizaciones.

        Args:
            figsize: Tamaño por defecto de la figura
            dpi: DPI para las figuras guardadas
        """
        self.figsize = figsize
        self.dpi = dpi
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.matplotlib_available = True
            logger.info("Matplotlib and seaborn initialized")
        except ImportError:
            self.matplotlib_available = False
            logger.warning("Matplotlib/seaborn not available - visualization disabled")

    def plot_feature_distribution_comparison(
        self,
        baseline_df: pd.DataFrame,
        monitoring_df: pd.DataFrame,
        feature: str,
        output_path: Optional[Path] = None,
        bins: int = 30
    ) -> Optional[str]:
        """
        Plot baseline vs monitoring feature distributions.

        Args:
            baseline_df: Baseline data
            monitoring_df: Monitoring data
            feature: Feature to plot
            output_path: Path to save figure
            bins: Number of histogram bins

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            logger.warning("Visualization not available")
            return None

        try:
            fig, axes = self.plt.subplots(1, 2, figsize=self.figsize)

            # Histograma
            axes[0].hist(baseline_df[feature].dropna(), bins=bins, alpha=0.6, label='Baseline', color='blue')
            axes[0].hist(monitoring_df[feature].dropna(), bins=bins, alpha=0.6, label='Monitoring', color='red')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Frecuencia')
            axes[0].set_title(f'Comparación de distribución: {feature}')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Gráfico KDE
            baseline_df[feature].dropna().plot.kde(ax=axes[1], label='Baseline', linewidth=2)
            monitoring_df[feature].dropna().plot.kde(ax=axes[1], label='Monitoring', linewidth=2)
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Densidad')
            axes[1].set_title(f'Comparación de densidad: {feature}')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved distribution plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}")
            return None
        finally:
            self.plt.close('all')

    def plot_multiple_features(
        self,
        baseline_df: pd.DataFrame,
        monitoring_df: pd.DataFrame,
        features: List[str],
        output_path: Optional[Path] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Optional[str]:
        """
        Plot distributions for multiple features.

        Args:
            baseline_df: Baseline data
            monitoring_df: Monitoring data
            features: List of features to plot
            output_path: Path to save figure
            figsize: Figure size override

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            return None

        try:
            n_features = len(features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols
            figsize = figsize or (4 * n_cols, 3 * n_rows)

            fig, axes = self.plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_features > 1 else [axes]

            for idx, feature in enumerate(features):
                if feature not in baseline_df.columns or feature not in monitoring_df.columns:
                    logger.warning(f"Feature {feature} not found in data")
                    continue

                baseline_data = baseline_df[feature].dropna()
                monitoring_data = monitoring_df[feature].dropna()

                axes[idx].hist(baseline_data, bins=20, alpha=0.6, label='Baseline', color='blue')
                axes[idx].hist(monitoring_data, bins=20, alpha=0.6, label='Monitoring', color='red')
                axes[idx].set_title(f'{feature} Distribution')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)

            # Eliminar subplots sobrantes
            for idx in range(n_features, len(axes)):
                fig.delaxes(axes[idx])

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved multiple features plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating multi-feature plot: {e}")
            return None
        finally:
            self.plt.close('all')

    def plot_performance_metrics(
        self,
        baseline_metrics: Dict[str, float],
        monitoring_metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot performance metrics comparison.

        Args:
            baseline_metrics: Baseline performance metrics
            monitoring_metrics: Monitoring period metrics
            output_path: Path to save figure

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            return None

        try:
            metrics = [k for k in baseline_metrics.keys() if k in monitoring_metrics]

            baseline_vals = [baseline_metrics[m] for m in metrics]
            monitoring_vals = [monitoring_metrics[m] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = self.plt.subplots(figsize=(12, 6))

            bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, monitoring_vals, width, label='Monitoring', color='red', alpha=0.7)

            ax.set_xlabel('Métricas')
            ax.set_ylabel('Puntaje')
            ax.set_title('Comparación de métricas de desempeño')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3, axis='y')

            # Añadir etiquetas con valores en las barras
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved performance metrics plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating performance plot: {e}")
            return None
        finally:
            self.plt.close('all')

    def plot_metrics_history(
        self,
        history: List[Dict[str, Any]],
        metric_name: str = 'f1_weighted',
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot metric evolution over time.

        Args:
            history: List of monitoring records with metrics
            metric_name: Metric to plot
            output_path: Path to save figure

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            return None

        try:
            values = []
            timestamps = []

            for record in history:
                if "degradation" in record and metric_name in record["degradation"]:
                    values.append(record["degradation"][metric_name]["absolute"])
                    timestamps.append(record.get("timestamp", ""))

            if not values:
                logger.warning(f"No data found for metric {metric_name}")
                return None

            fig, ax = self.plt.subplots(figsize=(12, 6))

            ax.plot(range(len(values)), values, marker='o', linewidth=2, markersize=8, color='red')
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline (sin degradación)')
            ax.fill_between(range(len(values)), 0, values, alpha=0.3, color='red')

            ax.set_xlabel('Periodo de monitoreo')
            ax.set_ylabel(f'Degradación de {metric_name}')
            ax.set_title(f'Degradación de {metric_name} en el tiempo')
            ax.grid(alpha=0.3)
            ax.legend()

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved metrics history plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating history plot: {e}")
            return None
        finally:
            self.plt.close('all')

    def plot_feature_statistics_comparison(
        self,
        baseline_stats: Dict[str, Dict[str, float]],
        monitoring_stats: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot comparison of feature statistics.

        Args:
            baseline_stats: Baseline feature statistics
            monitoring_stats: Monitoring feature statistics
            output_path: Path to save figure

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            return None

        try:
            features = list(baseline_stats.keys())
            if not features:
                logger.warning("No features to plot")
                return None

            # Comparar medias y desviaciones estándar
            baseline_means = [baseline_stats[f].get('mean', 0) for f in features]
            monitoring_means = [monitoring_stats.get(f, {}).get('mean', 0) for f in features]
            baseline_stds = [baseline_stats[f].get('std', 0) for f in features]
            monitoring_stds = [monitoring_stats.get(f, {}).get('std', 0) for f in features]

            fig, axes = self.plt.subplots(1, 2, figsize=(14, 5))

            x = np.arange(len(features))
            width = 0.35

            # Means
            axes[0].bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.7)
            axes[0].bar(x + width/2, monitoring_means, width, label='Monitoring', alpha=0.7)
            axes[0].set_ylabel('Media')
            axes[0].set_title('Comparación de medias')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(features, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(alpha=0.3, axis='y')

            # Stds
            axes[1].bar(x - width/2, baseline_stds, width, label='Baseline', alpha=0.7)
            axes[1].bar(x + width/2, monitoring_stds, width, label='Monitoring', alpha=0.7)
            axes[1].set_ylabel('Desviación estándar')
            axes[1].set_title('Comparación de desviaciones estándar')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(features, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(alpha=0.3, axis='y')

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved statistics comparison plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating statistics plot: {e}")
            return None
        finally:
            self.plt.close('all')

    def plot_alert_severity_distribution(
        self,
        alerts_by_severity: Dict[str, int],
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot distribution of alert severities.

        Args:
            alerts_by_severity: Dict with severity counts
            output_path: Path to save figure

        Returns:
            Path to saved figure or None
        """
        if not self.matplotlib_available:
            return None

        try:
            severities = list(alerts_by_severity.keys())
            counts = list(alerts_by_severity.values())

            colors_map = {
                'critical': 'red',
                'high': 'orange',
                'medium': 'yellow',
                'low': 'green'
            }
            colors = [colors_map.get(s, 'blue') for s in severities]

            fig, ax = self.plt.subplots(figsize=(10, 6))

            bars = ax.bar(severities, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            ax.set_ylabel('Count')
            ax.set_title('Alert Severity Distribution')
            ax.grid(alpha=0.3, axis='y')

            # Añadir etiquetas de valor
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

            self.plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved alert severity plot to {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"Error creating alert plot: {e}")
            return None
        finally:
            self.plt.close('all')
