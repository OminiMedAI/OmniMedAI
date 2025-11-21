"""
onem_segment - Medical Image Automatic ROI Segmentation Module

This module provides automatic ROI (Region of Interest) segmentation capabilities for medical images.
It automatically determines whether to use 2D or 3D segmentation models based on the z-axis characteristics
of the NIfTI images, and saves the segmented ROI as NIfTI files.
"""

from .segmenters.roi_segmenter import ROISegmenter
from .config.settings import SegmentationConfig, PRESET_CONFIGS
from .utils.image_analyzer import ImageDimensionAnalyzer

__version__ = "1.0.0"
__author__ = "OmniMedAI Team"

__all__ = [
    'ROISegmenter',
    'SegmentationConfig',
    'PRESET_CONFIGS',
    'ImageDimensionAnalyzer'
]