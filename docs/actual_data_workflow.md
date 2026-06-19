# Actual-Data Workflow

The reusable APIs accept real study tables and image paths now. Synthetic data
is used only for software tests.

## Installation

```bash
pip install -r requirements-analysis.txt
```

`scikit-survival` may require platform-specific binary packages. It is only
needed for time-dependent ROC.

## Configuration-Driven Execution

Copy `docs/templates/revision_workflow_config.json`, enable the stages for which
data is available, and update the input paths.

```bash
python -m onem_start.revision_analysis path/to/revision_workflow_config.json
```

Each enabled stage writes CSV, JSON, or PNG outputs under the configured output
directory. Stages remain disabled by default so missing study data is never
silently replaced with synthetic data.

## 1. Model Development

Input template: `docs/templates/model_input_schema.csv`

```python
from onem_modeling import NestedCVConfig, nested_patient_cross_validate

result = nested_patient_cross_validate(
    "data/model_input.csv",
    label_column="label",
    patient_column="patient_id",
    config=NestedCVConfig(
        model_type="logistic_regression",
        feature_selection="radiomics_sequence",
        n_features=20,
        outer_folds=5,
        inner_folds=4,
        random_state=2026,
        selection_parameters={
            "univariate_p_threshold": 0.05,
            "correlation_threshold": 0.9,
            "lasso_cv_folds": 5,
        },
    ),
)

result["predictions"].to_csv("results/oof_predictions.csv", index=False)
```

All imputation, scaling, feature selection, and hyperparameter search occur
inside the training folds.

## 2. Classification Evaluation

```python
from onem_eval import bootstrap_auc_ci, calibration_table, decision_curve

predictions = result["predictions"]
auc = bootstrap_auc_ci(predictions["y_true"], predictions["y_score"])
calibration = calibration_table(predictions["y_true"], predictions["y_score"])
dca = decision_curve(predictions["y_true"], predictions["y_score"])
```

## 3. Survival Analysis

Input template: `docs/templates/survival_input_schema.csv`

```python
from onem_eval import fit_cox_model, kaplan_meier_table, number_at_risk_table

cox = fit_cox_model(
    "data/survival.csv",
    duration_column="time_months",
    event_column="event",
    covariates=["tsr_score", "age", "stage"],
    categorical_covariates=["stage"],
)
```

Kaplan-Meier and Cox analysis require `lifelines`. Time-dependent AUC requires
`scikit-survival`.

## 4. Treatment Sensitivity Analysis

Input template: `docs/templates/treatment_input_schema.csv`

```python
from onem_eval import (
    estimate_propensity_weights,
    save_waterfall_plot,
    treatment_interaction_logistic,
)

interaction = treatment_interaction_logistic(
    "data/treatment.csv",
    outcome_column="response",
    treatment_column="treatment",
    biomarker_column="tsr_score",
    covariates=["age", "performance_status", "tumor_burden"],
)

weights = estimate_propensity_weights(
    "data/treatment.csv",
    treatment_column="treatment",
    covariates=["age", "performance_status", "tumor_burden"],
)

save_waterfall_plot(
    "data/treatment.csv",
    "results/treatment_waterfall.png",
    change_column="best_change_percent",
)
```

## 5. Image Reconstruction Comparison

Input template: `docs/templates/image_pair_manifest.csv`

```python
from onem_process import compare_image_pairs

quality = compare_image_pairs(
    "data/image_pairs.csv",
    roi_column="roi_path",
    background_column="background_path",
)
quality.to_csv("results/reconstruction_quality.csv", index=False)
```

## 6. Radiomics Reliability and Harmonization

Input template: `docs/templates/radiomics_repeat_schema.csv`

```python
from onem_radiomics import (
    batch_effect_summary,
    calculate_icc,
    combat_harmonize,
)

icc = calculate_icc(
    repeated_features,
    subject_column="patient_id",
    repeat_column="repeat_id",
)
batch_summary = batch_effect_summary(features, batch_column="center")
harmonized = combat_harmonize(training_features, batch_column="center")
```

ComBat parameters must be estimated without combining held-out external
validation data with training data.

## 7. Pathological TSR Ground Truth

Input template: `docs/templates/pathology_tsr_schema.csv`

```python
from onem_path import aggregate_tsr_ground_truth, tsr_interobserver_icc

ground_truth = aggregate_tsr_ground_truth(
    pathology_measurements,
    cutoff=50.0,
)
agreement = tsr_interobserver_icc(pathology_measurements)
```

The aggregated output reports the median TSR, range across slides/readers,
number of measurements, and the prespecified TSR group.

## 8. Bulk and Single-Cell Omics

Input templates:

- `docs/templates/bulk_expression_schema.csv`
- `docs/templates/omics_sample_metadata_schema.csv`
- `docs/templates/single_cell_metadata_schema.csv`
- `docs/templates/omics_accession_schema.csv`

```python
from onem_omics import (
    meca_fibroblast_summary,
    sequencing_qc_summary,
    validate_accession_manifest,
    validate_expression_metadata,
)

validation = validate_expression_metadata(expression, sample_metadata)
qc = sequencing_qc_summary(sample_metadata)
mecaf = meca_fibroblast_summary(single_cell_metadata)
accessions = validate_accession_manifest(accession_manifest)
```

## Pending Study Inputs

The code is ready to accept actual data, but final manuscript results still
require:

- The real cohort tables and image paths.
- The exact super-resolution reconstruction implementation and parameters.
- Actual treatment regimens and response measurements.
- Bulk RNA-seq and single-cell RNA-seq matrices and metadata.
- Ethics, data-accession, and map-approval identifiers.
