"""Pathology image extraction and tumor-stroma ratio utilities."""

from .config.settings import PathologyConfig, PRESET_CONFIGS
from .tsr import (
    aggregate_tsr_ground_truth,
    compare_specimen_tsr,
    tsr_group_agreement,
    tsr_interobserver_icc,
    validate_tsr_table,
)

__version__ = "1.12.1"
__author__ = "OmniMedAI Team"

__all__ = [
    'CellProfilerExtractor',
    'TITANExtractor', 
    'PathologyConfig',
    'PRESET_CONFIGS',
    'aggregate_tsr_ground_truth',
    'compare_specimen_tsr',
    'tsr_group_agreement',
    'tsr_interobserver_icc',
    'validate_tsr_table',
]


def __getattr__(name):
    if name == "CellProfilerExtractor":
        from .extractors.cellprofiler_extractor import CellProfilerExtractor

        return CellProfilerExtractor
    if name == "TITANExtractor":
        from .extractors.titan_extractor import TITANExtractor

        return TITANExtractor
    raise AttributeError(name)
