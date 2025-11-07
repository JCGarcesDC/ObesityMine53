"""
Feature engineering package for ObesityMine.

Expose public feature-building utilities here to keep imports stable:
from src.features import build_features
"""

from .feature_engineering import (
    build_features      # si la tienes
)

__all__ = [
    "build_features"
]
