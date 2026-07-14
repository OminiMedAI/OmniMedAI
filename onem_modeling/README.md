# onem_modeling

Modeling module for OmniMedAI feature tables. It turns extracted or fused
features into baseline machine-learning models for research workflows.

## Core Capabilities

- Train baseline classifiers or regressors from CSV/DataFrame feature tables.
- Automatically split train/test data.
- Apply median imputation and optional feature scaling.
- Support random forest, SVM, logistic regression, and linear regression.
- Support optional XGBoost classification and regression.
- Split datasets at patient level and detect overlap between cohorts.
- Repeat the complete radiomics feature-selection sequence across random seeds.
- Report pairwise Jaccard similarity, per-feature selection frequency, and
  consensus features across runs.
- Compose univariate, correlation, mRMR, and LASSO selection stages.
- Run the complete selection sequence inside nested cross-validation.
- Fit on a development cohort, infer an external cohort, and persist models.
- Generate optional SHAP feature summaries.
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

```python
from onem_modeling import (
    assert_no_patient_overlap,
    patient_level_train_test_split,
    summarize_feature_selection_stability,
)

train_df, test_df, split_summary = patient_level_train_test_split(
    feature_table,
    patient_col="patient_id",
    label_col="label",
    random_state=2026,
)
assert_no_patient_overlap(train_df, test_df)

stability = summarize_feature_selection_stability({
    1: ["feature_a", "feature_b"],
    2: ["feature_a", "feature_b", "feature_c"],
})
```

The complete univariate/correlation/mRMR/LASSO sequence can also be repeated
directly. Ten runs produce 45 pairwise Jaccard comparisons:

```python
from onem_modeling import (
    FeatureSelectionConfig,
    repeated_seed_feature_selection,
)

seed_result = repeated_seed_feature_selection(
    training_features,
    training_labels,
    config=FeatureSelectionConfig(
        univariate_p_threshold=0.05,
        correlation_threshold=0.9,
        mrmr_features=50,
        lasso_cv_folds=10,
        random_state=2026,
    ),
    n_repeats=10,
)

stability = seed_result["stability"]
print(stability["mean_pairwise_jaccard"])
print(stability["feature_selection_rate"])
```

## Nested XGBoost Search

```python
from onem_modeling import (
    NestedCVConfig,
    model_param_grid,
    nested_patient_cross_validate,
    xgboost_param_grid,
)

result = nested_patient_cross_validate(
    feature_table,
    label_column="label",
    patient_column="patient_id",
    config=NestedCVConfig(
        model_type="xgboost",
        outer_folds=5,
        inner_folds=4,
        scoring="roc_auc",
        param_grid=xgboost_param_grid("expanded"),
    ),
)
```

Compact parameter grids are used by default for nested validation. Expanded
grids are available for XGBoost, SVM, logistic regression, random forest,
ExtraTrees, k-nearest neighbors, and Gaussian naive Bayes:

```python
config = NestedCVConfig(
    model_type="svm",
    param_grid=model_param_grid("svm", "expanded"),
)
```

Users can also provide a custom `param_grid` directly; it is passed to
`GridSearchCV` inside the inner patient-level cross-validation loop.

## Status

This module is a baseline modeling layer. Future work can add nested
cross-validation, survival models, feature selection pipelines, calibration,
class-imbalance handling, and experiment tracking.
