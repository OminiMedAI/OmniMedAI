"""
onem_modeling - Modeling utilities for OmniMedAI feature tables.

This module provides lightweight machine-learning workflows for tabular medical
AI features, including train/test splitting, model fitting, cross-validation,
and prediction export.
"""

from .modeling import ModelingConfig, TabularModeler, train_tabular_model

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    "ModelingConfig",
    "TabularModeler",
    "train_tabular_model",
]
