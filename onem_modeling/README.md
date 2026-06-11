# onem_modeling

Modeling module for OmniMedAI feature tables. It turns extracted or fused
features into baseline machine-learning models for research workflows.

## Core Capabilities

- Train baseline classifiers or regressors from CSV/DataFrame feature tables.
- Automatically split train/test data.
- Apply median imputation and optional feature scaling.
- Support random forest, SVM, logistic regression, and linear regression.
- Export predictions and probabilities for downstream evaluation.

## Quick Start

```python
from onem_modeling import ModelingConfig, train_tabular_model

result = train_tabular_model(
    csv_path="output/fused_features.csv",
    label_column="label",
    config=ModelingConfig(
        task="classification",
        model_type="random_forest",
        test_size=0.2
    )
)

model = result["model"]
y_test = result["y_test"]
y_pred = result["predictions"]
y_proba = result["probabilities"]
```

## Status

This module is a baseline modeling layer. Future work can add nested
cross-validation, survival models, feature selection pipelines, calibration,
class-imbalance handling, and experiment tracking.
