"""
Utility functions for pathology feature extraction
"""

from .file_utils import (
    get_pathology_files, create_output_structure, validate_pathology_image,
    calculate_file_hash, get_image_file_info, create_file_manifest,
    save_batch_report, clean_temp_files, create_progress_callback,
    format_processing_time
)
from .image_utils import (
    preprocess_pathology_image, normalize_staining, resize_pathology_image,
    enhance_image_contrast, load_pil_image, load_skimage_image,
    load_cv2_image, convert_color_space, extract_nuclei_features,
    _reinhard_normalization, _macenko_normalization, _vahadane_normalization
)
from .cellprofiler_utils import (
    create_cellprofiler_pipeline, run_cellprofiler_pipeline,
    parse_cellprofiler_output, convert_cellprofiler_to_standard_format,
    create_cellprofiler_batch_script
)
from .titan_utils import (
    load_titan_model, extract_titan_features, preprocess_for_titan,
    create_titan_transforms, extract_layer_features,
    benchmark_titan_extraction, get_available_backbones,
    validate_titan_model, save_titan_model
)

__all__ = [
    # File utilities
    'get_pathology_files',
    'create_output_structure',
    'validate_pathology_image',
    'calculate_file_hash',
    'get_image_file_info',
    'create_file_manifest',
    'save_batch_report',
    'clean_temp_files',
    'create_progress_callback',
    'format_processing_time',
    
    # Image utilities
    'preprocess_pathology_image',
    'normalize_staining',
    'resize_pathology_image',
    'enhance_image_contrast',
    'load_pil_image',
    'load_skimage_image',
    'load_cv2_image',
    'convert_color_space',
    'extract_nuclei_features',
    '_reinhard_normalization',
    '_macenko_normalization',
    '_vahadane_normalization',
    
    # CellProfiler utilities
    'create_cellprofiler_pipeline',
    'run_cellprofiler_pipeline',
    'parse_cellprofiler_output',
    'convert_cellprofiler_to_standard_format',
    'create_cellprofiler_batch_script',
    
    # TITAN utilities
    'load_titan_model',
    'extract_titan_features',
    'preprocess_for_titan',
    'create_titan_transforms',
    'extract_layer_features',
    'benchmark_titan_extraction',
    'get_available_backbones',
    'validate_titan_model',
    'save_titan_model'
]