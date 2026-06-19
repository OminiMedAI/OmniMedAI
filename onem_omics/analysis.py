"""Actual-data-ready bulk and single-cell omics table utilities."""

from typing import List, Optional


def _require_dependencies():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("numpy and pandas are required for omics analysis") from exc
    return np, pd


def _load_table(data):
    _, pd = _require_dependencies()
    return pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()


def validate_expression_metadata(
    expression,
    metadata,
    sample_column: str = "sample_id",
    gene_column: str = "gene_id",
):
    """Validate a genes-by-samples expression matrix against sample metadata."""
    _, pd = _require_dependencies()
    expression_table = _load_table(expression)
    metadata_table = _load_table(metadata)
    if sample_column not in metadata_table.columns:
        raise ValueError(f"Metadata missing sample column: {sample_column}")
    if metadata_table[sample_column].duplicated().any():
        raise ValueError("Metadata must contain one row per sample")
    if gene_column not in expression_table.columns:
        raise ValueError(f"Expression matrix missing gene column: {gene_column}")
    if expression_table[gene_column].duplicated().any():
        raise ValueError("Expression matrix contains duplicate gene identifiers")

    expression_samples = [column for column in expression_table.columns if column != gene_column]
    metadata_samples = set(metadata_table[sample_column].astype(str))
    missing_metadata = sorted(set(expression_samples).difference(metadata_samples))
    missing_expression = sorted(metadata_samples.difference(expression_samples))
    non_numeric = [
        column
        for column in expression_samples
        if not pd.api.types.is_numeric_dtype(expression_table[column])
    ]
    return {
        "valid": not missing_metadata and not missing_expression and not non_numeric,
        "n_genes": int(len(expression_table)),
        "n_expression_samples": int(len(expression_samples)),
        "n_metadata_samples": int(len(metadata_table)),
        "samples_missing_metadata": missing_metadata,
        "samples_missing_expression": missing_expression,
        "non_numeric_expression_columns": non_numeric,
    }


def sequencing_qc_summary(
    metadata,
    sample_column: str = "sample_id",
    total_reads_column: str = "total_reads",
    mapped_reads_column: str = "mapped_reads",
    detected_genes_column: Optional[str] = "detected_genes",
):
    """Create per-sample sequencing QC metrics from metadata."""
    np, pd = _require_dependencies()
    table = _load_table(metadata)
    required = {sample_column, total_reads_column, mapped_reads_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing sequencing QC columns: {missing}")
    result = table[[sample_column, total_reads_column, mapped_reads_column]].copy()
    total = pd.to_numeric(result[total_reads_column], errors="coerce")
    mapped = pd.to_numeric(result[mapped_reads_column], errors="coerce")
    result["mapping_rate"] = np.where(total > 0, mapped / total, np.nan)
    if detected_genes_column and detected_genes_column in table.columns:
        result[detected_genes_column] = pd.to_numeric(
            table[detected_genes_column], errors="coerce"
        )
    result["qc_issue"] = ""
    result.loc[total <= 0, "qc_issue"] = "non-positive total reads"
    result.loc[mapped > total, "qc_issue"] = "mapped reads exceed total reads"
    return result


def adjust_pvalues_bh(p_values):
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    np, _ = _require_dependencies()
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def compare_cell_type_proportions(
    cell_metadata,
    sample_column: str = "sample_id",
    group_column: str = "tsr_group",
    cell_type_column: str = "cell_type",
    target_cell_type: Optional[str] = None,
):
    """Calculate per-sample cell-type proportions and compare study groups."""
    np, pd = _require_dependencies()
    table = _load_table(cell_metadata)
    required = {sample_column, group_column, cell_type_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing single-cell metadata columns: {missing}")
    group_counts = table.groupby(sample_column)[group_column].nunique()
    if (group_counts > 1).any():
        raise ValueError("Each sample must belong to one study group")

    counts = (
        table.groupby([sample_column, group_column, cell_type_column])
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = (
        table.groupby([sample_column, group_column])
        .size()
        .rename("total_cells")
        .reset_index()
    )
    proportions = counts.merge(totals, on=[sample_column, group_column], how="left")
    proportions["proportion"] = proportions["n_cells"] / proportions["total_cells"]
    if target_cell_type is not None:
        proportions = proportions[
            proportions[cell_type_column].astype(str) == str(target_cell_type)
        ].copy()

    summary = proportions.groupby([group_column, cell_type_column]).agg(
        n_samples=(sample_column, "nunique"),
        mean_proportion=("proportion", "mean"),
        median_proportion=("proportion", "median"),
        std_proportion=("proportion", "std"),
    ).reset_index()

    tests = []
    try:
        from scipy.stats import mannwhitneyu

        for cell_type, subset in proportions.groupby(cell_type_column):
            groups = list(subset.groupby(group_column))
            if len(groups) != 2:
                continue
            left_name, left = groups[0]
            right_name, right = groups[1]
            statistic, p_value = mannwhitneyu(
                left["proportion"],
                right["proportion"],
                alternative="two-sided",
            )
            tests.append(
                {
                    "cell_type": cell_type,
                    "group_1": left_name,
                    "group_2": right_name,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    except ImportError:
        pass
    tests_table = pd.DataFrame(tests)
    if len(tests_table):
        tests_table["adjusted_p_value"] = adjust_pvalues_bh(tests_table["p_value"])
    return {
        "sample_proportions": proportions,
        "group_summary": summary,
        "group_tests": tests_table,
    }


def meca_fibroblast_summary(
    cell_metadata,
    meca_label: str = "meCAF",
    sample_column: str = "sample_id",
    group_column: str = "tsr_group",
    cell_type_column: str = "cell_type",
):
    """Convenience wrapper for reviewer-requested meCAF comparisons."""
    return compare_cell_type_proportions(
        cell_metadata,
        sample_column=sample_column,
        group_column=group_column,
        cell_type_column=cell_type_column,
        target_cell_type=meca_label,
    )


def validate_accession_manifest(
    manifest,
    required_dataset_types: Optional[List[str]] = None,
):
    """Validate public or controlled-access repository records."""
    table = _load_table(manifest)
    required_columns = {"dataset_type", "repository", "accession", "url", "status"}
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"Accession manifest missing columns: {missing}")
    required_dataset_types = required_dataset_types or ["bulk_rnaseq", "scrnaseq"]
    present = set(table["dataset_type"].astype(str))
    missing_types = sorted(set(required_dataset_types).difference(present))
    incomplete = table[
        table[["repository", "accession", "url", "status"]]
        .replace("", float("nan"))
        .isna()
        .any(axis=1)
    ]
    return {
        "valid": not missing_types and incomplete.empty,
        "missing_dataset_types": missing_types,
        "incomplete_rows": incomplete,
    }


def validate_marker_config(marker_config, cell_type_column: str = "cell_type"):
    """Validate a cell-type-to-marker configuration table."""
    table = _load_table(marker_config)
    required = {cell_type_column, "positive_markers"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Marker configuration missing columns: {missing}")
    issues = []
    for index, row in table.iterrows():
        markers = [
            marker.strip()
            for marker in str(row["positive_markers"]).split(",")
            if marker.strip()
        ]
        if not markers:
            issues.append(
                {"row": int(index), "cell_type": row[cell_type_column], "issue": "no markers"}
            )
    return {"valid": not issues, "issues": issues, "table": table}


def validate_enrichment_results(
    results,
    term_column: str = "term",
    p_value_column: str = "p_value",
    adjusted_p_column: Optional[str] = None,
):
    """Validate GO/KEGG/GSEA-style results and add BH correction if needed."""
    _, pd = _require_dependencies()
    table = _load_table(results)
    required = {term_column, p_value_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Enrichment results missing columns: {missing}")
    p_values = pd.to_numeric(table[p_value_column], errors="coerce")
    if p_values.isna().any() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("Enrichment p-values must be numeric and between 0 and 1")
    output = table.copy()
    if adjusted_p_column and adjusted_p_column in output.columns:
        adjusted = pd.to_numeric(output[adjusted_p_column], errors="coerce")
        if adjusted.isna().any() or ((adjusted < 0) | (adjusted > 1)).any():
            raise ValueError("Adjusted p-values must be between 0 and 1")
    else:
        adjusted_p_column = adjusted_p_column or "adjusted_p_value"
        output[adjusted_p_column] = adjust_pvalues_bh(p_values)
    return output.sort_values(adjusted_p_column).reset_index(drop=True)
