"""
Configuration settings for radiomics feature extraction
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RadiomicsConfig:
    """
    Configuration class for radiomics feature extraction.
    
    Contains all parameters needed to configure the radiomics extraction process,
    including image preprocessing, feature selection, and extraction parameters.
    """
    
    # Image preprocessing parameters
    resampled_pixel_spacing: Optional[Tuple[float, float, float]] = None
    interpolator: str = 'sitkBSpline'
    bin_width: int = 25
    normalize: bool = False
    normalize_scale: int = 100
    
    # Feature selection
    feature_types: List[str] = field(default_factory=lambda: [
        'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'
    ])
    
    # Weighting parameters for voxel-based features
    weight_center: Optional[Tuple[float, float, float]] = None
    weight_radius: Optional[float] = None
    
    # Processing parameters
    n_jobs: int = 1
    verbose: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters and set defaults."""
        # Validate interpolator
        valid_interpolators = [
            'sitkNearestNeighbor', 'sitkLinear', 'sitkBSpline', 
            'sitkGaussian', 'sitkLanczosWindowedSinc'
        ]
        if self.interpolator not in valid_interpolators:
            raise ValueError(f"Invalid interpolator: {self.interpolator}. "
                           f"Valid options: {valid_interpolators}")
        
        # Validate feature types
        valid_features = [
            'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape'
        ]
        for feature_type in self.feature_types:
            if feature_type not in valid_features:
                raise ValueError(f"Invalid feature type: {feature_type}. "
                               f"Valid options: {valid_features}")
        
        # Validate bin width
        if self.bin_width <= 0:
            raise ValueError("bin_width must be positive")
        
        # Validate resampled pixel spacing
        if self.resampled_pixel_spacing is not None:
            if len(self.resampled_pixel_spacing) != 3:
                raise ValueError("resampled_pixel_spacing must be a 3-element tuple")
            if any(spacing <= 0 for spacing in self.resampled_pixel_spacing):
                raise ValueError("All pixel spacing values must be positive")
        
        # Validate weighting parameters
        if self.weight_center and not self.weight_radius:
            raise ValueError("weight_radius must be specified if weight_center is provided")
        if self.weight_radius and not self.weight_center:
            raise ValueError("weight_center must be specified if weight_radius is provided")
        
        if self.weight_center and len(self.weight_center) != 3:
            raise ValueError("weight_center must be a 3-element tuple")
        
        # Validate processing parameters
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
            'resampled_pixel_spacing': self.resampled_pixel_spacing,
            'interpolator': self.interpolator,
            'bin_width': self.bin_width,
            'normalize': self.normalize,
            'normalize_scale': self.normalize_scale,
            'feature_types': self.feature_types,
            'weight_center': self.weight_center,
            'weight_radius': self.weight_radius,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'custom_settings': self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RadiomicsConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, file_path: str):
        """Save configuration to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'RadiomicsConfig':
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations for different use cases
PRESET_CONFIGS = {
    'standard': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm'],
        bin_width=25,
        verbose=True
    ),
    
    'comprehensive': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape'],
        bin_width=16,
        normalize=True,
        verbose=True
    ),
    
    'ct_lung': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm'],
        bin_width=25,
        resampled_pixel_spacing=(1.0, 1.0, 1.0),
        interpolator='sitkBSpline',
        custom_settings={
            'distances': [1, 2, 3],
            'force2D': False,
            'force2Ddimension': 0
        }
    ),
    
    'mri_brain': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm'],
        bin_width=16,
        resampled_pixel_spacing=(1.0, 1.0, 1.0),
        interpolator='sitkBSpline',
        normalize=True,
        normalize_scale=100
    ),
    
    'pet_tumor': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'],
        bin_width=32,
        resampled_pixel_spacing=(4.0, 4.0, 4.0),
        interpolator='sitkLinear',
        custom_settings={
            'gldm_a': 2
        }
    ),
    
    'fast_processing': RadiomicsConfig(
        feature_types=['firstorder', 'glcm'],
        bin_width=50,
        resampled_pixel_spacing=(2.0, 2.0, 2.0),
        n_jobs=4,
        verbose=False
    ),
    
    'high_quality': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape'],
        bin_width=8,
        resampled_pixel_spacing=(0.5, 0.5, 0.5),
        interpolator='sitkBSpline',
        normalize=True,
        normalize_scale=1000,
        custom_settings={
            'distances': [1, 2, 3, 4, 5],
            'force2D': False
        }
    ),
    
    'texture_focused': RadiomicsConfig(
        feature_types=['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'],
        bin_width=16,
        custom_settings={
            'distances': [1, 2, 3, 4],
            'symmetricalGLCM': True,
            'weightingNorm': 'euclidean'
        }
    ),
    
    'shape_focused': RadiomicsConfig(
        feature_types=['shape', 'firstorder'],
        bin_width=25,
        resampled_pixel_spacing=(1.0, 1.0, 1.0)
    ),
    
    'research': RadiomicsConfig(
        feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape'],
        bin_width=12,
        resampled_pixel_spacing=(1.0, 1.0, 1.0),
        interpolator='sitkBSpline',
        normalize=True,
        custom_settings={
            'distances': [1, 2, 3, 4, 5, 6],
            'symmetricalGLCM': True,
            'symmetricalGLRLM': True,
            'symmetricalGLSZM': True,
            'gldm_a': 2
        }
    )
}


def get_available_presets() -> List[str]:
    """Get list of available preset configurations."""
    return list(PRESET_CONFIGS.keys())


def create_custom_config(
    feature_types: Optional[List[str]] = None,
    bin_width: Optional[int] = None,
    resampled_pixel_spacing: Optional[Tuple[float, float, float]] = None,
    **kwargs
) -> RadiomicsConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        feature_types: List of feature types to extract
        bin_width: Histogram bin width
        resampled_pixel_spacing: Target pixel spacing for resampling
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom RadiomicsConfig object
    """
    config = RadiomicsConfig()
    
    if feature_types is not None:
        config.feature_types = feature_types
    if bin_width is not None:
        config.bin_width = bin_width
    if resampled_pixel_spacing is not None:
        config.resampled_pixel_spacing = resampled_pixel_spacing
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config