"""
onem_eval - Model evaluation utilities for OmniMedAI.

This module summarizes model performance for classification and regression
workflows and prepares outputs for reporting and interpretation.
"""

from .evaluation import EvaluationConfig, ModelEvaluator, evaluate_classification

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    "EvaluationConfig",
    "ModelEvaluator",
    "evaluate_classification",
]
