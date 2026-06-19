# onem_eval

Evaluation module for OmniMedAI modeling results. It summarizes classification
and regression performance for downstream reports.

## Core Capabilities

- Accuracy, F1, recall, precision, and classification report.
- Confusion matrix export.
- Binary ROC-AUC when probabilities are available.
- Bootstrap confidence intervals for binary ROC-AUC.
- Calibration tables and decision-curve net-benefit tables.
- Regression MAE, MSE, and R2.
- Machine-readable workflow manifests for acquisition, reconstruction,
  preprocessing, model development, validation, code, and data links.
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

```python
from onem_eval import (
    WorkflowManifest,
    bootstrap_auc_ci,
    calibration_table,
    decision_curve,
)

auc_report = bootstrap_auc_ci(y_true, y_score, n_bootstraps=2000)
calibration = calibration_table(y_true, y_score)
net_benefit = decision_curve(y_true, y_score)

manifest = WorkflowManifest(
    study_name="srMRI TSR phenotyping in PDAC",
    cohort_name="external validation",
    task="classification",
)
manifest.save_json("results/workflow_manifest.json")
```

## Status

This module provides baseline and reviewer-facing evaluation utilities. Planned
extensions include survival analysis, external-validation report exporters,
SHAP summaries, and Grad-CAM report hooks.
