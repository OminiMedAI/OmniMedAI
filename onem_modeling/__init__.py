"""
onem_modeling - Modeling utilities for OmniMedAI feature tables.

This module provides lightweight machine-learning workflows for tabular medical
AI features, including train/test splitting, model fitting, cross-validation,
and prediction export.
"""

from .modeling import ModelingConfig, TabularModeler, train_tabular_model
from .cross_validation import (
    NestedCVConfig,
    model_param_grid,
    nested_patient_cross_validate,
    xgboost_param_grid,
)
from .validation import (
    assert_no_patient_overlap,
    patient_level_train_test_split,
    summarize_feature_selection_stability,
)
from .feature_selection import (
    FeatureSelectionConfig,
    SequentialRadiomicsSelector,
    repeated_seed_feature_selection,
)
from .explain import shap_feature_summary

__version__ = "1.12.1"
__author__ = "OmniMedAI Team"

__all__ = [
    "ModelingConfig",
    "NestedCVConfig",
    "FeatureSelectionConfig",
    "SequentialRadiomicsSelector",
    "TabularModeler",
    "assert_no_patient_overlap",
    "patient_level_train_test_split",
    "repeated_seed_feature_selection",
    "nested_patient_cross_validate",
    "model_param_grid",
    "summarize_feature_selection_stability",
    "shap_feature_summary",
    "train_tabular_model",
    "xgboost_param_grid",
]
