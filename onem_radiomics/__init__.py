"""
onem_radiomics - Medical Imaging Radiomics Feature Extraction Module

This module provides comprehensive radiomics feature extraction capabilities for medical images.
It supports extraction of various radiomics features from NIfTI format images and masks,
with configurable parameters and batch processing capabilities.
"""

from .extractors.radiomics_extractor import RadiomicsExtractor
from .config.settings import RadiomicsConfig, PRESET_CONFIGS

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    'RadiomicsExtractor',
    'RadiomicsConfig',
    'PRESET_CONFIGS'
]