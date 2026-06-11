"""
onem_fusion - Multimodal feature fusion utilities for OmniMedAI.

This module aligns and combines radiomics, pathomics, clinical, genomic,
habitat, and deep-learning feature tables into modeling-ready datasets.
"""

from .fusion import FusionConfig, FeatureFusion, align_feature_tables, concatenate_features

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    "FusionConfig",
    "FeatureFusion",
    "align_feature_tables",
    "concatenate_features",
]
