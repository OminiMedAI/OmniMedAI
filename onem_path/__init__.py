"""
onem_path - Pathology Image Feature Extraction Module

This module provides comprehensive pathology image feature extraction capabilities,
including traditional pathology radiomics using CellProfiler and deep transfer 
learning feature extraction using TITAN model.
"""

from .extractors.cellprofiler_extractor import CellProfilerExtractor
from .extractors.titan_extractor import TITANExtractor
from .config.settings import PathologyConfig, PRESET_CONFIGS

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    'CellProfilerExtractor',
    'TITANExtractor', 
    'PathologyConfig',
    'PRESET_CONFIGS'
]