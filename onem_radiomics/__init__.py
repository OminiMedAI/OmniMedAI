"""
onem_radiomics - Medical Imaging Radiomics Feature Extraction Module

This module provides comprehensive radiomics feature extraction capabilities for medical images.
It supports extraction of various radiomics features from NIfTI format images and masks,
with configurable parameters and batch processing capabilities.
"""

from .extractors.radiomics_extractor import RadiomicsExtractor
from .config.settings import RadiomicsConfig, PRESET_CONFIGS
from .utils.radiomics_utils import calculate_icc
from .harmonization import batch_effect_summary, combat_harmonize

__version__ = "1.12.1"
__author__ = "OmniMedAI Team"

__all__ = [
    'RadiomicsExtractor',
    'RadiomicsConfig',
    'PRESET_CONFIGS',
    'batch_effect_summary',
    'calculate_icc',
    'combat_harmonize'
]
