"""Run reviewer-requested analyses from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(data, path: Path):
    path.write_text(
        json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _section(config: Dict[str, Any], name: str):
    value = config.get(name)
    if not value or value.get("enabled", True) is False:
        return None
    return value


def run_modeling(section, output_dir: Path):
    from onem_modeling import (
        FeatureSelectionConfig,
        NestedCVConfig,
        nested_patient_cross_validate,
        repeated_seed_feature_selection,
        summarize_feature_selection_stability,
    )

    options = section.get("config", {})
    model_config = NestedCVConfig(**options)
    result = nested_patient_cross_validate(
        section["input_csv"],
        label_column=section["label_column"],
        patient_column=section.get("patient_column", "patient_id"),
        feature_columns=section.get("feature_columns"),
        config=model_config,
    )
    result["predictions"].to_csv(output_dir / "model_oof_predictions.csv", index=False)
    stability = summarize_feature_selection_stability(result["selected_features_by_fold"])
    stability["comparison_scope"] = "outer_cross_validation_fold"
    _write_json(result["summary"], output_dir / "model_summary.json")
    _write_json(result["fold_results"], output_dir / "model_fold_results.json")
    _write_json(stability, output_dir / "model_feature_stability.json")

    seed_options = section.get("seed_stability", {})
    if seed_options.get("enabled", False):
        import pandas as pd

        if model_config.feature_selection != "radiomics_sequence":
            raise ValueError(
                "Seed stability requires feature_selection=radiomics_sequence"
            )
        table = pd.read_csv(section["input_csv"])
        patient_column = section.get("patient_column", "patient_id")
        label_column = section["label_column"]
        if patient_column in table.columns and table[patient_column].duplicated().any():
            raise ValueError(
                "Seed stability requires one feature-table row per patient"
            )
        feature_columns = section.get("feature_columns")
        if feature_columns is None:
            excluded = {patient_column, label_column}
            feature_columns = [
                column
                for column in table.columns
                if column not in excluded
                and pd.api.types.is_numeric_dtype(table[column])
            ]
        selector_options = dict(model_config.selection_parameters)
        selector_options.setdefault("task", model_config.task)
        selector_options.setdefault("mrmr_features", model_config.n_features)
        selector_options.setdefault("random_state", model_config.random_state)
        seed_result = repeated_seed_feature_selection(
            table[feature_columns],
            table[label_column],
            config=FeatureSelectionConfig(**selector_options),
            n_repeats=seed_options.get("n_repeats", 10),
            random_states=seed_options.get("random_states"),
        )
        _write_json(
            seed_result,
            output_dir / "model_seed_feature_stability.json",
        )
    return result


def run_classification_evaluation(section, output_dir: Path, model_result=None):
    import pandas as pd

    from onem_eval import bootstrap_auc_ci, calibration_table, decision_curve

    if section.get("predictions_csv"):
        predictions = pd.read_csv(section["predictions_csv"])
    elif model_result is not None:
        predictions = model_result["predictions"]
    else:
        raise ValueError(
            "classification_evaluation requires predictions_csv or an enabled modeling stage"
        )
    true_column = section.get("true_column", "y_true")
    score_column = section.get("score_column", "y_score")
    auc = bootstrap_auc_ci(
        predictions[true_column],
        predictions[score_column],
        n_bootstraps=section.get("n_bootstraps", 2000),
        confidence_level=section.get("confidence_level", 0.95),
        random_state=section.get("random_state", 42),
    )
    calibration = calibration_table(
        predictions[true_column],
        predictions[score_column],
        n_bins=section.get("n_bins", 10),
    )
    dca = decision_curve(predictions[true_column], predictions[score_column])
    _write_json(auc, output_dir / "classification_auc.json")
    calibration.to_csv(output_dir / "classification_calibration.csv", index=False)
    dca.to_csv(output_dir / "classification_decision_curve.csv", index=False)


def run_survival(section, output_dir: Path):
    from onem_eval import (
        fit_cox_model,
        kaplan_meier_table,
        number_at_risk_table,
        time_dependent_auc,
    )

    common = {
        "duration_column": section["duration_column"],
        "event_column": section["event_column"],
        "patient_column": section.get("patient_column", "patient_id"),
    }
    km = kaplan_meier_table(
        section["input_csv"],
        group_column=section.get("group_column"),
        **common,
    )
    km.to_csv(output_dir / "survival_kaplan_meier.csv", index=False)
    if section.get("time_points"):
        risk = number_at_risk_table(
            section["input_csv"],
            time_points=section["time_points"],
            group_column=section.get("group_column"),
            **common,
        )
        risk.to_csv(output_dir / "survival_number_at_risk.csv", index=False)
    if section.get("covariates"):
        cox = fit_cox_model(
            section["input_csv"],
            covariates=section["covariates"],
            categorical_covariates=section.get("categorical_covariates"),
            penalizer=section.get("penalizer", 0.0),
            **common,
        )
        cox["summary"].to_csv(output_dir / "survival_cox_summary.csv", index=False)
        _write_json(
            {
                "concordance_index": cox["concordance_index"],
                "n_complete_cases": cox["n_complete_cases"],
                "model_columns": cox["model_columns"],
            },
            output_dir / "survival_cox_metrics.json",
        )
    if section.get("time_dependent_auc"):
        auc_options = section["time_dependent_auc"]
        result = time_dependent_auc(
            auc_options["train_csv"],
            auc_options["test_csv"],
            duration_column=section["duration_column"],
            event_column=section["event_column"],
            risk_score_column=auc_options["risk_score_column"],
            times=auc_options["times"],
        )
        result["table"].to_csv(output_dir / "survival_time_dependent_auc.csv", index=False)
        _write_json(
            {"mean_auc": result["mean_auc"]},
            output_dir / "survival_time_dependent_auc_summary.json",
        )


def run_treatment(section, output_dir: Path):
    from onem_eval import (
        estimate_propensity_weights,
        save_waterfall_plot,
        treatment_interaction_logistic,
        waterfall_table,
    )

    if section.get("change_column"):
        table = waterfall_table(
            section["input_csv"],
            change_column=section["change_column"],
            patient_column=section.get("patient_column", "patient_id"),
            group_column=section.get("group_column"),
        )
        table.to_csv(output_dir / "treatment_waterfall_data.csv", index=False)
        if section.get("save_plot", True):
            save_waterfall_plot(
                section["input_csv"],
                output_dir / "treatment_waterfall.png",
                change_column=section["change_column"],
                patient_column=section.get("patient_column", "patient_id"),
                group_column=section.get("group_column"),
            )
    if section.get("interaction"):
        options = section["interaction"]
        result = treatment_interaction_logistic(
            section["input_csv"],
            patient_column=section.get("patient_column", "patient_id"),
            outcome_column=options["outcome_column"],
            treatment_column=options["treatment_column"],
            biomarker_column=options["biomarker_column"],
            covariates=options.get("covariates"),
        )
        result["summary"].to_csv(output_dir / "treatment_interaction.csv", index=False)
        _write_json(
            {
                "interaction_term": result["interaction_term"],
                "interaction_odds_ratio": result["interaction_odds_ratio"],
                "interaction_p_value": result["interaction_p_value"],
                "n_complete_cases": result["n_complete_cases"],
            },
            output_dir / "treatment_interaction_summary.json",
        )
    if section.get("propensity"):
        options = section["propensity"]
        result = estimate_propensity_weights(
            section["input_csv"],
            patient_column=section.get("patient_column", "patient_id"),
            treatment_column=options["treatment_column"],
            covariates=options["covariates"],
            estimand=options.get("estimand", "ate"),
        )
        result["weights"].to_csv(output_dir / "treatment_propensity_weights.csv", index=False)


def run_image_quality(section, output_dir: Path):
    from onem_process import compare_image_pairs

    result = compare_image_pairs(
        section["manifest_csv"],
        reference_column=section.get("reference_column", "reference_path"),
        reconstructed_column=section.get(
            "reconstructed_column", "reconstructed_path"
        ),
        patient_column=section.get("patient_column", "patient_id"),
        roi_column=section.get("roi_column"),
        background_column=section.get("background_column"),
    )
    result.to_csv(output_dir / "image_quality.csv", index=False)


def run_radiomics(section, output_dir: Path):
    import pandas as pd

    from onem_radiomics import batch_effect_summary, calculate_icc, combat_harmonize

    table = pd.read_csv(section["input_csv"])
    feature_columns = section.get("feature_columns")
    if section.get("repeat_column"):
        icc = calculate_icc(
            table,
            subject_column=section.get("patient_column", "patient_id"),
            repeat_column=section["repeat_column"],
            feature_columns=feature_columns,
            icc_type=section.get("icc_type", "icc2_1"),
        )
        icc.to_csv(output_dir / "radiomics_icc.csv", index=False)
    if section.get("batch_column"):
        summary = batch_effect_summary(
            table,
            batch_column=section["batch_column"],
            feature_columns=feature_columns,
        )
        summary.to_csv(output_dir / "radiomics_batch_summary.csv", index=False)
    if section.get("combat", {}).get("enabled", False):
        options = section["combat"]
        result = combat_harmonize(
            table,
            batch_column=section["batch_column"],
            feature_columns=feature_columns,
            categorical_covariates=options.get("categorical_covariates"),
            continuous_covariates=options.get("continuous_covariates"),
        )
        result["data"].to_csv(output_dir / "radiomics_harmonized.csv", index=False)


def run_pathology(section, output_dir: Path):
    import pandas as pd

    from onem_path import aggregate_tsr_ground_truth, tsr_interobserver_icc

    table = pd.read_csv(section["input_csv"])
    common = {
        "patient_column": section.get("patient_column", "patient_id"),
        "reader_column": section.get("reader_column", "reader_id"),
        "slide_column": section.get("slide_column", "slide_id"),
        "tsr_column": section.get("tsr_column", "tsr_percent"),
        "specimen_column": section.get("specimen_column", "specimen_type"),
    }
    ground_truth = aggregate_tsr_ground_truth(
        table,
        cutoff=section["cutoff"],
        **common,
    )
    ground_truth.to_csv(output_dir / "pathology_tsr_ground_truth.csv", index=False)
    agreement = tsr_interobserver_icc(
        table,
        patient_column=common["patient_column"],
        reader_column=common["reader_column"],
        tsr_column=common["tsr_column"],
    )
    _write_json(agreement, output_dir / "pathology_tsr_agreement.json")


def run_omics(section, output_dir: Path):
    import pandas as pd

    from onem_omics import (
        meca_fibroblast_summary,
        sequencing_qc_summary,
        validate_accession_manifest,
        validate_expression_metadata,
    )

    if section.get("expression_csv") and section.get("sample_metadata_csv"):
        validation = validate_expression_metadata(
            section["expression_csv"],
            section["sample_metadata_csv"],
            sample_column=section.get("sample_column", "sample_id"),
            gene_column=section.get("gene_column", "gene_id"),
        )
        _write_json(validation, output_dir / "omics_expression_validation.json")
    if section.get("sample_metadata_csv") and section.get("qc_columns"):
        columns = section["qc_columns"]
        qc = sequencing_qc_summary(
            section["sample_metadata_csv"],
            sample_column=section.get("sample_column", "sample_id"),
            total_reads_column=columns["total_reads"],
            mapped_reads_column=columns["mapped_reads"],
            detected_genes_column=columns.get("detected_genes"),
        )
        qc.to_csv(output_dir / "omics_sequencing_qc.csv", index=False)
    if section.get("cell_metadata_csv"):
        result = meca_fibroblast_summary(
            section["cell_metadata_csv"],
            meca_label=section.get("mecaf_label", "meCAF"),
            sample_column=section.get("sample_column", "sample_id"),
            group_column=section.get("group_column", "tsr_group"),
            cell_type_column=section.get("cell_type_column", "cell_type"),
        )
        result["sample_proportions"].to_csv(
            output_dir / "omics_mecaf_sample_proportions.csv", index=False
        )
        result["group_summary"].to_csv(
            output_dir / "omics_mecaf_group_summary.csv", index=False
        )
        result["group_tests"].to_csv(
            output_dir / "omics_mecaf_group_tests.csv", index=False
        )
    if section.get("accession_manifest_csv"):
        result = validate_accession_manifest(section["accession_manifest_csv"])
        incomplete_path = output_dir / "omics_incomplete_accessions.csv"
        result["incomplete_rows"].to_csv(incomplete_path, index=False)
        _write_json(
            {
                "valid": result["valid"],
                "missing_dataset_types": result["missing_dataset_types"],
                "incomplete_rows_csv": str(incomplete_path),
            },
            output_dir / "omics_accession_validation.json",
        )


def run_workflow(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config.get("output_dir", "results/revision_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_result = None
    completed = []

    runners = [
        ("modeling", run_modeling),
        ("classification_evaluation", run_classification_evaluation),
        ("survival", run_survival),
        ("treatment", run_treatment),
        ("image_quality", run_image_quality),
        ("radiomics", run_radiomics),
        ("pathology", run_pathology),
        ("omics", run_omics),
    ]
    for name, runner in runners:
        section = _section(config, name)
        if section is None:
            continue
        if name == "modeling":
            model_result = runner(section, output_dir)
        elif name == "classification_evaluation":
            runner(section, output_dir, model_result=model_result)
        else:
            runner(section, output_dir)
        completed.append(name)

    _write_json(
        {
            "config_path": str(config_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "completed_stages": completed,
        },
        output_dir / "workflow_run_summary.json",
    )
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Run actual-data reviewer analyses from JSON configuration."
    )
    parser.add_argument("config", help="Path to revision workflow JSON configuration")
    args = parser.parse_args()
    completed = run_workflow(args.config)
    print("Completed stages: " + (", ".join(completed) if completed else "none"))


if __name__ == "__main__":
    main()
