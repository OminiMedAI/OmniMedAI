"""Evaluation helpers for OmniMedAI modeling outputs."""

from typing import Optional

from .config.settings import EvaluationConfig


def _require_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError("scikit-learn is required for evaluation. Install with: pip install scikit-learn") from exc

    return {
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "precision_score": precision_score,
        "r2_score": r2_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def evaluate_classification(
    y_true,
    y_pred,
    y_proba: Optional[object] = None,
    config: Optional[EvaluationConfig] = None,
):
    """Evaluate classification predictions."""
    config = config or EvaluationConfig()
    metrics = _require_sklearn_metrics()

    result = {
        "accuracy": float(metrics["accuracy_score"](y_true, y_pred)),
        "f1": float(metrics["f1_score"](y_true, y_pred, average=config.average, zero_division=0)),
        "recall": float(metrics["recall_score"](y_true, y_pred, average=config.average, zero_division=0)),
        "precision": float(metrics["precision_score"](y_true, y_pred, average=config.average, zero_division=0)),
        "classification_report": metrics["classification_report"](y_true, y_pred, zero_division=0),
    }

    if config.include_confusion_matrix:
        result["confusion_matrix"] = metrics["confusion_matrix"](y_true, y_pred).tolist()

    if config.include_auc and y_proba is not None:
        try:
            if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2:
                auc_input = y_proba[:, 1]
            else:
                auc_input = y_proba
            result["auc"] = float(metrics["roc_auc_score"](y_true, auc_input))
        except Exception as exc:
            result["auc_error"] = str(exc)

    return result


def evaluate_regression(y_true, y_pred):
    """Evaluate regression predictions."""
    metrics = _require_sklearn_metrics()
    return {
        "mae": float(metrics["mean_absolute_error"](y_true, y_pred)),
        "mse": float(metrics["mean_squared_error"](y_true, y_pred)),
        "r2": float(metrics["r2_score"](y_true, y_pred)),
    }


class ModelEvaluator:
    """Convenience evaluator for model outputs."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate(self, y_true, y_pred, y_proba: Optional[object] = None):
        if self.config.task == "classification":
            return evaluate_classification(y_true, y_pred, y_proba, self.config)
        return evaluate_regression(y_true, y_pred)
