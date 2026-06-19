# Revision Implementation Status

This matrix distinguishes reusable code availability from completion of the
paper's actual analyses.

## Executable with Real Data

| Reviewer requirement | Implementation | Status before real data |
|---|---|---|
| Patient-level splitting and leakage control | `onem_modeling.validation` | Ready |
| Nested CV with fold-local preprocessing and sequential radiomics feature selection | `onem_modeling.cross_validation` | Ready |
| Hyperparameter and selected-feature reporting | Nested CV outputs | Ready |
| Seed/fold feature-selection stability | `summarize_feature_selection_stability` | Ready |
| AUC confidence intervals | `onem_eval.bootstrap_auc_ci` | Ready |
| Calibration and decision curves | `onem_eval.calibration_table`, `decision_curve` | Ready |
| Kaplan-Meier, number at risk, Cox, C-index | `onem_eval.survival` | Ready; requires `lifelines` |
| Time-dependent ROC | `onem_eval.time_dependent_auc` | Ready; requires `scikit-survival` |
| Treatment-biomarker interaction | `onem_eval.treatment_interaction_logistic` | Ready |
| Propensity weighting | `onem_eval.estimate_propensity_weights` | Ready |
| Patient-level waterfall plot | `onem_eval.save_waterfall_plot` | Ready |
| Standard versus reconstructed MRI quality | `onem_process.compare_image_pairs` | Ready |
| Interpolation/deep-model SR reconstruction API | `onem_process.reconstruction` | Ready; study checkpoint still required |
| Stage/size contradiction checks | `onem_process.check_stage_size_consistency` | Ready |
| Radiomics ICC | `onem_radiomics.calculate_icc` | Ready |
| Original/Wavelet/LoG radiomics filters | `RadiomicsConfig.image_types` | Ready |
| Scanner/center batch summaries | `onem_radiomics.batch_effect_summary` | Ready |
| ComBat harmonization | `onem_radiomics.combat_harmonize` | Ready; requires `neuroCombat` |
| Multi-slide/multi-reader pathological TSR | `onem_path.tsr` | Ready |
| TSR interobserver agreement | `onem_path.tsr_interobserver_icc` | Ready |
| Bulk expression/metadata validation | `onem_omics.validate_expression_metadata` | Ready |
| Sequencing QC table | `onem_omics.sequencing_qc_summary` | Ready |
| meCAF proportion comparison | `onem_omics.meca_fibroblast_summary` | Ready |
| Multiple-testing correction | `onem_omics.adjust_pvalues_bh` | Ready |
| Data-accession completeness | `onem_omics.validate_accession_manifest` | Ready |

## Workflow or Template Available

- Machine-readable study workflow manifest.
- Input schemas for model, survival, treatment, image pairs, radiomics repeats,
  pathology TSR, bulk expression, sample metadata, single-cell metadata, and
  data accessions.
- Configuration-driven runner: `python -m onem_start.revision_analysis`.
- Manuscript/editor compliance checklist.
- Data/code availability template.
- Reviewer-response evidence matrix.

## Requires Actual Study Materials

- Final model estimates, confidence intervals, figures, and tables.
- The exact super-resolution implementation, weights, scale factor, and
  reconstruction parameters.
- Corrected clinical stage and tumor-size source records.
- Real chemotherapy/ICI regimens, response values, treatment lines, and
  confounders.
- Bulk RNA-seq and single-cell matrices plus sequencing metadata.
- Public repository accession numbers or controlled-access instructions.
- Ethics approval number and China map approval number.
- Final manuscript pages/lines, figure source files, and official TRIPOD+AI
  checklist.

Synthetic fixtures validate software behavior only and must not be reported as
study evidence.
