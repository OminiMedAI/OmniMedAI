"""Omics quality control and cell-composition analysis utilities."""

from .analysis import (
    adjust_pvalues_bh,
    compare_cell_type_proportions,
    meca_fibroblast_summary,
    sequencing_qc_summary,
    validate_accession_manifest,
    validate_enrichment_results,
    validate_expression_metadata,
    validate_marker_config,
)

__version__ = "1.12.1"
__author__ = "OmniMedAI Team"

__all__ = [
    "adjust_pvalues_bh",
    "compare_cell_type_proportions",
    "meca_fibroblast_summary",
    "sequencing_qc_summary",
    "validate_accession_manifest",
    "validate_enrichment_results",
    "validate_expression_metadata",
    "validate_marker_config",
]
