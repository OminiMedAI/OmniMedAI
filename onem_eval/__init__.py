"""
onem_eval - Model evaluation utilities for OmniMedAI.

This module summarizes model performance for classification and regression
workflows and prepares outputs for reporting and interpretation.
"""

from .evaluation import EvaluationConfig, ModelEvaluator, evaluate_classification
from .advanced import (
    bootstrap_auc_ci,
    calibration_table,
    compare_binary_models,
    decision_curve,
)
from .reporting import WorkflowManifest
from .survival import (
    fit_cox_model,
    kaplan_meier_table,
    number_at_risk_table,
    time_dependent_auc,
    validate_survival_table,
)
from .treatment import (
    estimate_propensity_weights,
    save_waterfall_plot,
    treatment_interaction_logistic,
    validate_treatment_table,
    waterfall_table,
)

__version__ = "1.12.1"
__author__ = "OmniMedAI Team"

__all__ = [
    "EvaluationConfig",
    "ModelEvaluator",
    "WorkflowManifest",
    "bootstrap_auc_ci",
    "calibration_table",
    "compare_binary_models",
    "decision_curve",
    "estimate_propensity_weights",
    "evaluate_classification",
    "fit_cox_model",
    "kaplan_meier_table",
    "number_at_risk_table",
    "save_waterfall_plot",
    "time_dependent_auc",
    "treatment_interaction_logistic",
    "validate_survival_table",
    "validate_treatment_table",
    "waterfall_table",
]
