# onem_eval

Evaluation module for OmniMedAI modeling results. It summarizes classification
and regression performance for downstream reports.

## Core Capabilities

- Accuracy, F1, recall, precision, and classification report.
- Confusion matrix export.
- Binary ROC-AUC when probabilities are available.
- Regression MAE, MSE, and R2.
- Small API designed to pair with `onem_modeling`.

## Quick Start

```python
from onem_eval import EvaluationConfig, ModelEvaluator

evaluator = ModelEvaluator(EvaluationConfig(
    task="classification",
    average="weighted",
    include_auc=True
))

metrics = evaluator.evaluate(
    y_true=result["y_test"],
    y_pred=result["predictions"],
    y_proba=result["probabilities"]
)

print(metrics["accuracy"])
print(metrics["classification_report"])
```

## Status

This module provides baseline evaluation. Planned extensions include
calibration curves, decision curve analysis, confidence intervals,
external-validation reports, SHAP summaries, and Grad-CAM report hooks.
