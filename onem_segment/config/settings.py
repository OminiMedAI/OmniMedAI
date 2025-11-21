"""
Configuration settings for ROI segmentation
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SegmentationConfig:
    """
    Configuration class for ROI segmentation.
    
    Contains all parameters needed to configure the segmentation process,
    including model selection, preprocessing, and post-processing options.
    """
    
    # Model selection parameters
    model_type: str = 'auto'  # 'auto', '2d', '3d'
    quality_level: str = 'default'  # 'default', 'high_quality'
    model_dir: Optional[str] = None
    device: str = 'auto'  # 'cpu', 'cuda', 'auto'
    
    # Image analysis parameters
    min_3d_slices: int = 30
    max_slice_spacing: float = 5.0
    min_content_variation: float = 0.1
    slice_sampling_ratio: float = 0.1
    
    # Preprocessing parameters
    normalization: str = 'z_score'  # 'z_score', 'min_max', 'none'
    resize_input: bool = True
    input_size_2d: Tuple[int, int] = (256, 256)
    input_size_3d: Tuple[int, int, int] = (64, 64, 64)
    interpolation_method: str = 'bilinear'  # 'nearest', 'bilinear', 'trilinear'
    
    # Post-processing parameters
    min_roi_volume: int = 10  # Minimum volume in voxels
    remove_small_components: bool = True
    morphological_operations: bool = True
    morph_iterations: int = 2
    connectivity: int = 1  # 1 for 6-connectivity, 2 for 26-connectivity
    
    # Inference parameters
    return_probabilities: bool = False
    probability_threshold: float = 0.5
    batch_inference: bool = False
    batch_size: int = 4
    
    # Output parameters
    output_format: str = 'nii.gz'  # 'nii.gz', 'nii'
    save_probabilities: bool = False
    save_preprocessing: bool = False
    generate_statistics: bool = True
    
    # Quality control parameters
    validate_inputs: bool = True
    check_model_compatibility: bool = True
    log_intermediate_steps: bool = False
    
    # Performance parameters
    n_jobs: int = 1
    memory_limit_gb: Optional[float] = None
    use_mixed_precision: bool = False
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters and set defaults."""
        # Validate model type
        valid_model_types = ['auto', '2d', '3d']
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. "
                           f"Valid options: {valid_model_types}")
        
        # Validate quality level
        valid_quality_levels = ['default', 'high_quality']
        if self.quality_level not in valid_quality_levels:
            raise ValueError(f"Invalid quality_level: {self.quality_level}. "
                           f"Valid options: {valid_quality_levels}")
        
        # Validate device
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. "
                           f"Valid options: {valid_devices}")
        
        # Validate normalization
        valid_normalizations = ['z_score', 'min_max', 'none']
        if self.normalization not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.normalization}. "
                           f"Valid options: {valid_normalizations}")
        
        # Validate input sizes
        if len(self.input_size_2d) != 2 or any(s <= 0 for s in self.input_size_2d):
            raise ValueError("input_size_2d must be a tuple of 2 positive integers")
        
        if len(self.input_size_3d) != 3 or any(s <= 0 for s in self.input_size_3d):
            raise ValueError("input_size_3d must be a tuple of 3 positive integers")
        
        # Validate probability threshold
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be between 0 and 1")
        
        # Validate output format
        valid_formats = ['nii.gz', 'nii']
        if self.output_format not in valid_formats:
            raise ValueError(f"Invalid output_format: {self.output_format}. "
                           f"Valid options: {valid_formats}")
        
        # Validate numerical parameters
        if self.min_3d_slices <= 0:
            raise ValueError("min_3d_slices must be positive")
        
        if self.max_slice_spacing <= 0:
            raise ValueError("max_slice_spacing must be positive")
        
        if self.min_roi_volume < 0:
            raise ValueError("min_roi_volume must be non-negative")
        
        if self.morph_iterations < 0:
            raise ValueError("morph_iterations must be non-negative")
        
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        self.validate()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'quality_level': self.quality_level,
            'model_dir': self.model_dir,
            'device': self.device,
            'min_3d_slices': self.min_3d_slices,
            'max_slice_spacing': self.max_slice_spacing,
            'min_content_variation': self.min_content_variation,
            'slice_sampling_ratio': self.slice_sampling_ratio,
            'normalization': self.normalization,
            'resize_input': self.resize_input,
            'input_size_2d': self.input_size_2d,
            'input_size_3d': self.input_size_3d,
            'interpolation_method': self.interpolation_method,
            'min_roi_volume': self.min_roi_volume,
            'remove_small_components': self.remove_small_components,
            'morphological_operations': self.morphological_operations,
            'morph_iterations': self.morph_iterations,
            'connectivity': self.connectivity,
            'return_probabilities': self.return_probabilities,
            'probability_threshold': self.probability_threshold,
            'batch_inference': self.batch_inference,
            'batch_size': self.batch_size,
            'output_format': self.output_format,
            'save_probabilities': self.save_probabilities,
            'save_preprocessing': self.save_preprocessing,
            'generate_statistics': self.generate_statistics,
            'validate_inputs': self.validate_inputs,
            'check_model_compatibility': self.check_model_compatibility,
            'log_intermediate_steps': self.log_intermediate_steps,
            'n_jobs': self.n_jobs,
            'memory_limit_gb': self.memory_limit_gb,
            'use_mixed_precision': self.use_mixed_precision,
            'custom_settings': self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SegmentationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, file_path: str):
        """Save configuration to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'SegmentationConfig':
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations for different use cases
PRESET_CONFIGS = {
    'default': SegmentationConfig(
        model_type='auto',
        quality_level='default',
        normalization='z_score',
        morphological_operations=True,
        generate_statistics=True
    ),
    
    'fast': SegmentationConfig(
        model_type='auto',
        quality_level='default',
        input_size_2d=(128, 128),
        input_size_3d=(32, 32, 32),
        remove_small_components=False,
        morphological_operations=False,
        generate_statistics=False,
        n_jobs=4
    ),
    
    'high_quality': SegmentationConfig(
        model_type='auto',
        quality_level='high_quality',
        input_size_2d=(512, 512),
        input_size_3d=(128, 128, 128),
        morphological_operations=True,
        morph_iterations=3,
        return_probabilities=True,
        save_probabilities=True,
        use_mixed_precision=True
    ),
    
    '2d_only': SegmentationConfig(
        model_type='2d',
        quality_level='default',
        normalization='z_score',
        resize_input=True,
        input_size_2d=(256, 256)
    ),
    
    '3d_only': SegmentationConfig(
        model_type='3d',
        quality_level='default',
        normalization='z_score',
        resize_input=True,
        input_size_3d=(64, 64, 64),
        min_3d_slices=20  # Lower threshold for 3D-only
    ),
    
    'ct_organ': SegmentationConfig(
        model_type='auto',
        quality_level='high_quality',
        normalization='z_score',
        min_3d_slices=25,
        max_slice_spacing=3.0,
        min_roi_volume=50,
        morphological_operations=True,
        morph_iterations=3,
        return_probabilities=True
    ),
    
    'mri_brain': SegmentationConfig(
        model_type='auto',
        quality_level='high_quality',
        normalization='z_score',
        input_size_2d=(256, 256),
        input_size_3d=(96, 96, 96),
        min_3d_slices=30,
        min_roi_volume=20,
        morphological_operations=True,
        connectivity=2  # 26-connectivity for brain structures
    ),
    
    'pet_tumor': SegmentationConfig(
        model_type='auto',
        quality_level='default',
        normalization='min_max',
        min_3d_slices=15,
        max_slice_spacing=6.0,
        min_content_variation=0.05,
        min_roi_volume=8,
        return_probabilities=True,
        probability_threshold=0.3
    ),
    
    'research': SegmentationConfig(
        model_type='auto',
        quality_level='high_quality',
        normalization='z_score',
        input_size_2d=(512, 512),
        input_size_3d=(256, 256, 256),
        morphological_operations=True,
        morph_iterations=4,
        return_probabilities=True,
        save_probabilities=True,
        save_preprocessing=True,
        generate_statistics=True,
        log_intermediate_steps=True,
        use_mixed_precision=True
    ),
    
    'production': SegmentationConfig(
        model_type='auto',
        quality_level='default',
        normalization='z_score',
        validate_inputs=True,
        check_model_compatibility=True,
        generate_statistics=True,
        n_jobs=2,
        memory_limit_gb=8.0
    ),
    
    'memory_efficient': SegmentationConfig(
        model_type='2d',  # Force 2D for memory efficiency
        quality_level='default',
        input_size_2d=(128, 128),
        batch_inference=True,
        batch_size=2,
        memory_limit_gb=4.0,
        use_mixed_precision=True
    )
}


def get_available_presets() -> List[str]:
    """Get list of available preset configurations."""
    return list(PRESET_CONFIGS.keys())


def create_custom_config(
    model_type: str = 'auto',
    quality_level: str = 'default',
    input_size: Optional[Tuple] = None,
    device: str = 'auto',
    **kwargs
) -> SegmentationConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        model_type: Model type selection strategy
        quality_level: Quality level for processing
        input_size: Input size for images (2D or 3D)
        device: Device for computation
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom SegmentationConfig object
    """
    config = SegmentationConfig()
    
    if model_type is not None:
        config.model_type = model_type
    if quality_level is not None:
        config.quality_level = quality_level
    if device is not None:
        config.device = device
    
    # Set input size based on model type
    if input_size is not None:
        if len(input_size) == 2:
            config.input_size_2d = input_size
        elif len(input_size) == 3:
            config.input_size_3d = input_size
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config