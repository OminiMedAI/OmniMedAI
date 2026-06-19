# Reviewer Revision Plan for srMRI TSR PDAC Manuscript

This plan converts the three reviewer reports and associate-editor comments in
`revise/revise1.md` into paper actions and reusable platform actions.

## Highest-Risk Revision Items

1. **Reproducibility and runnable code**
   - Publish a tagged repository release with runnable examples, fixed random
     seeds, parameter files, environment information, and a
     `reproducibility_manifest.json`.
   - Ensure manuscript methods match the released code exactly.
   - Add a supplementary workflow diagram covering MRI acquisition,
     super-resolution reconstruction, preprocessing, segmentation, radiomics,
     feature selection, model training, validation, omics, and reporting.

2. **Data transparency and consistency**
   - Provide code/data URLs for imaging feature tables, bulk RNA-seq,
     single-cell RNA-seq, and any controlled-access clinical data.
   - Audit clinical tables for impossible T-stage and tumor-size combinations.
   - Explain sample overlap, cohort membership, treatment era, missing-data
     handling, endpoint definitions, and follow-up for every cohort.

3. **srMRI and radiomics methods**
   - Name the super-resolution algorithm, implementation, weights, parameters,
     scale factor, input/output spacing, and quantitative image-quality
     comparisons before versus after reconstruction.
   - Report scanner vendor/model, field strength, sequences, resampling,
     normalization, segmentation workflow, inter-reader agreement, PyRadiomics
     parameters, feature stability, and harmonization.
   - Add a direct comparison between standard MRI and srMRI-derived features.

4. **Model-development rigor**
   - Confirm all splits are patient-level, not image-level or ROI-level.
   - Keep preprocessing, feature selection, and model fitting inside
     cross-validation or bootstrap loops.
   - Report hyperparameters, AUC confidence intervals, calibration,
     decision-curve analysis, C-index/time-dependent ROC where relevant,
     multivariable Cox models, and incremental value over clinical/MRI models.
   - Test feature-selection stability across random seeds and report ICC for
     test-retest, inter-reader, or cross-device radiomics robustness.

5. **Therapy-response claims**
   - Tone down immunotherapy conclusions to hypothesis-generating evidence.
   - Describe chemotherapy regimen, line of therapy, performance status, tumor
     burden, subsequent treatment, treatment registration or off-label status,
     and sensitivity analyses.
   - Provide per-patient response values and a waterfall plot for the small
     chemotherapy plus PD-1 subgroup.

6. **Omics transparency**
   - Deposit or link bulk and single-cell data.
   - Report sequencing QC, batch correction, clustering parameters, CAF/T-cell
     markers, pathway-enrichment methods, and multiple-testing correction.
   - Add meCAF proportions in high- and low-TSR groups and avoid causal wording
     unless spatial validation or multiplex IHC is added.

7. **Manuscript formatting and journal compliance**
   - Restructure according to TRIPOD+AI and submit the checklist.
   - Move Materials and Methods after Discussion if required by the journal.
   - Add required declarations, ethics approval number, data availability,
     code repository link, figure legends section, China map approval number,
     and revised figure files meeting journal specifications.

## Reusable APIs Integrated into Existing Modules

The technical requirements are distributed according to existing platform
responsibilities instead of being placed in a manuscript-specific package:

- `onem_modeling`: `patient_level_train_test_split`,
  `assert_no_patient_overlap`, and `summarize_feature_selection_stability`.
- `onem_radiomics`: `calculate_icc` for repeated scans, segmentations, readers,
  or devices.
- `onem_eval`: `bootstrap_auc_ci`, `calibration_table`, `decision_curve`, and
  `WorkflowManifest` for machine-readable study reporting.
- `onem_process`: `check_stage_size_consistency` for clinical table quality
  control.
- `onem_path`: multi-slide/multi-reader TSR aggregation and agreement.
- `onem_omics`: expression/metadata validation, sequencing QC, meCAF
  proportions, FDR correction, and accession validation.

This layout makes the functions reusable by radiomics and multimodal studies
without coupling the public API to one paper or one review cycle.

Implementation status is tracked in
[`revision_implementation_status.md`](./revision_implementation_status.md).

## Recommended Supplementary Files

- `Supplementary Table S1`: Cohort inclusion/exclusion, overlap, stage, CA19-9,
  treatment, missingness, follow-up, and endpoint definitions.
- `Supplementary Table S2`: MRI scanners, sequences, acquisition parameters,
  reconstruction settings, and preprocessing parameters.
- `Supplementary Table S3`: Radiomics features, ICC, harmonization method, and
  selected-feature stability.
- `Supplementary Table S4`: Model hyperparameters, split seeds, calibration,
  AUC confidence intervals, decision-curve results, and external validation.
- `Supplementary Table S5`: Bulk and single-cell sequencing QC, accession
  numbers, annotation markers, enrichment settings, and adjusted P-value rules.
- `Supplementary Figure`: Standard MRI versus srMRI quality and performance
  comparison.
- `Supplementary Figure`: Treatment-response waterfall plot for each patient in
  the chemotherapy plus PD-1 subgroup.
