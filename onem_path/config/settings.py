"""
Configuration settings for pathology feature extraction
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathologyConfig:
    """
    Configuration class for pathology feature extraction.
    
    Contains all parameters needed to configure both traditional pathology
    radiomics (CellProfiler) and deep transfer learning (TITAN) features.
    """
    
    # Common parameters
    output_format: str = 'json'  # 'json', 'csv'
    device: str = 'auto'  # 'cpu', 'cuda', 'auto'
    n_jobs: int = 4
    verbose: bool = True
    
    # Image preprocessing parameters
    resize_images: bool = True
    image_size: Tuple[int, int] = (224, 224)
    normalize_images: bool = True
    color_conversion: str = 'rgb'  # 'rgb', 'hsv', 'lab'
    
    # CellProfiler parameters
    extract_cellprofiler_features: bool = True
    cellprofiler_modules: List[str] = field(default_factory=lambda: [
        'morphological', 'texture', 'intensity', 'color', 'nuclei'
    ])
    cellprofiler_binary_threshold: str = 'otsu'  # 'otsu', 'adaptive', 'manual'
    cellprofiler_manual_threshold: float = 0.5
    min_object_size: int = 50
    max_object_size: Optional[int] = None
    
    # TITAN model parameters
    extract_titan_features: bool = True
    titan_backbone: str = 'resnet50'  # 'resnet50', 'efficientnet_b0', etc.
    titan_pretrained: bool = True
    titan_feature_dim: int = 1024
    titan_use_attention: bool = True
    titan_checkpoint_path: Optional[str] = None
    titan_extract_layer_features: bool = False
    titan_target_layers: List[str] = field(default_factory=lambda: [
        'layer1', 'layer2', 'layer3', 'layer4'
    ])
    
    # Feature selection parameters
    feature_selection: Optional[str] = None  # 'variance', 'correlation', 'pca'
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    n_pca_components: Optional[int] = None
    
    # Quality control parameters
    validate_inputs: bool = True
    check_image_quality: bool = True
    min_image_size: Tuple[int, int] = (64, 64)
    max_image_size: Tuple[int, int] = (4096, 4096)
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'
    ])
    
    # Output parameters
    save_intermediate_results: bool = False
    save_feature_images: bool = False
    generate_reports: bool = True
    compress_output: bool = False
    
    # Performance parameters
    batch_size_titan: int = 32
    num_workers_titan: int = 4
    memory_limit_gb: Optional[float] = None
    use_mixed_precision: bool = False
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters and set defaults."""
        # Validate output format
        valid_formats = ['json', 'csv']
        if self.output_format not in valid_formats:
            raise ValueError(f"Invalid output_format: {self.output_format}. "
                           f"Valid options: {valid_formats}")
        
        # Validate device
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. "
                           f"Valid options: {valid_devices}")
        
        # Validate color conversion
        valid_conversions = ['rgb', 'hsv', 'lab']
        if self.color_conversion not in valid_conversions:
            raise ValueError(f"Invalid color_conversion: {self.color_conversion}. "
                           f"Valid options: {valid_conversions}")
        
        # Validate CellProfiler modules
        valid_modules = ['morphological', 'texture', 'intensity', 'color', 'nuclei',
                        'cells', 'cytoplasm', 'membrane']
        for module in self.cellprofiler_modules:
            if module not in valid_modules:
                raise ValueError(f"Invalid CellProfiler module: {module}. "
                               f"Valid options: {valid_modules}")
        
        # Validate image size
        if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
            raise ValueError("image_size must be a tuple of 2 positive integers")
        
        # Validate thresholds
        if not 0 <= self.cellprofiler_manual_threshold <= 1:
            raise ValueError("cellprofiler_manual_threshold must be between 0 and 1")
        
        if not 0 <= self.variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")
        
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        
        # Validate numeric parameters
        if self.min_object_size <= 0:
            raise ValueError("min_object_size must be positive")
        
        if self.max_object_size is not None and self.max_object_size <= self.min_object_size:
            raise ValueError("max_object_size must be greater than min_object_size")
        
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        
        # Validate batch sizes
        if self.batch_size_titan < 1:
            raise ValueError("batch_size_titan must be at least 1")
        
        if self.num_workers_titan < 0:
            raise ValueError("num_workers_titan must be non-negative")
    
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
            'output_format': self.output_format,
            'device': self.device,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'resize_images': self.resize_images,
            'image_size': self.image_size,
            'normalize_images': self.normalize_images,
            'color_conversion': self.color_conversion,
            'extract_cellprofiler_features': self.extract_cellprofiler_features,
            'cellprofiler_modules': self.cellprofiler_modules,
            'cellprofiler_binary_threshold': self.cellprofiler_binary_threshold,
            'cellprofiler_manual_threshold': self.cellprofiler_manual_threshold,
            'min_object_size': self.min_object_size,
            'max_object_size': self.max_object_size,
            'extract_titan_features': self.extract_titan_features,
            'titan_backbone': self.titan_backbone,
            'titan_pretrained': self.titan_pretrained,
            'titan_feature_dim': self.titan_feature_dim,
            'titan_use_attention': self.titan_use_attention,
            'titan_checkpoint_path': self.titan_checkpoint_path,
            'titan_extract_layer_features': self.titan_extract_layer_features,
            'titan_target_layers': self.titan_target_layers,
            'feature_selection': self.feature_selection,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'n_pca_components': self.n_pca_components,
            'validate_inputs': self.validate_inputs,
            'check_image_quality': self.check_image_quality,
            'min_image_size': self.min_image_size,
            'max_image_size': self.max_image_size,
            'supported_formats': self.supported_formats,
            'save_intermediate_results': self.save_intermediate_results,
            'save_feature_images': self.save_feature_images,
            'generate_reports': self.generate_reports,
            'compress_output': self.compress_output,
            'batch_size_titan': self.batch_size_titan,
            'num_workers_titan': self.num_workers_titan,
            'memory_limit_gb': self.memory_limit_gb,
            'use_mixed_precision': self.use_mixed_precision,
            'custom_settings': self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PathologyConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, file_path: str):
        """Save configuration to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'PathologyConfig':
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations for different use cases
PRESET_CONFIGS = {
    'default': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=True,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'nuclei'],
        titan_backbone='resnet50',
        titan_feature_dim=1024,
        output_format='json'
    ),
    
    'cellprofiler_only': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=False,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'color', 'nuclei', 'cells'],
        output_format='json'
    ),
    
    'titan_only': PathologyConfig(
        extract_cellprofiler_features=False,
        extract_titan_features=True,
        titan_backbone='resnet50',
        titan_feature_dim=1024,
        titan_use_attention=True,
        output_format='json'
    ),
    
    'high_quality': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=True,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'color', 'nuclei', 'cells', 'cytoplasm'],
        titan_backbone='resnet101',
        titan_feature_dim=2048,
        titan_use_attention=True,
        image_size=(512, 512),
        output_format='json',
        generate_reports=True
    ),
    
    'fast_processing': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=False,
        cellprofiler_modules=['morphological', 'intensity'],
        image_size=(224, 224),
        batch_size_titan=64,
        n_jobs=8,
        save_intermediate_results=False,
        output_format='csv'
    ),
    
    'research_grade': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=True,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'color', 'nuclei', 'cells', 'cytoplasm', 'membrane'],
        titan_backbone='resnet152',
        titan_feature_dim=2048,
        titan_use_attention=True,
        titan_extract_layer_features=True,
        image_size=(512, 512),
        output_format='json',
        save_intermediate_results=True,
        save_feature_images=True,
        generate_reports=True
    ),
    
    'production': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=True,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'nuclei'],
        titan_backbone='resnet50',
        titan_feature_dim=1024,
        titan_use_attention=False,  # Faster inference
        image_size=(224, 224),
        validate_inputs=True,
        check_image_quality=True,
        output_format='csv',
        generate_reports=True
    ),
    
    'memory_efficient': PathologyConfig(
        extract_cellprofiler_features=False,
        extract_titan_features=True,
        titan_backbone='efficientnet_b0',  # Smaller backbone
        titan_feature_dim=512,
        titan_use_attention=False,
        image_size=(256, 256),
        batch_size_titan=16,
        use_mixed_precision=True,
        memory_limit_gb=4.0,
        output_format='csv'
    ),
    
    'gpu_accelerated': PathologyConfig(
        extract_cellprofiler_features=False,
        extract_titan_features=True,
        titan_backbone='resnet50',
        titan_feature_dim=1024,
        titan_use_attention=True,
        image_size=(224, 224),
        batch_size_titan=128,
        num_workers_titan=8,
        use_mixed_precision=True,
        device='cuda',
        output_format='json'
    ),
    
    'comprehensive': PathologyConfig(
        extract_cellprofiler_features=True,
        extract_titan_features=True,
        cellprofiler_modules=['morphological', 'texture', 'intensity', 'color', 'nuclei', 'cells', 'cytoplasm', 'membrane'],
        titan_backbone='resnet101',
        titan_feature_dim=2048,
        titan_use_attention=True,
        titan_extract_layer_features=True,
        titan_target_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        image_size=(512, 512),
        output_format='json',
        save_intermediate_results=True,
        save_feature_images=True,
        generate_reports=True
    )
}


def get_available_presets() -> List[str]:
    """Get list of available preset configurations."""
    return list(PRESET_CONFIGS.keys())


def get_available_backbones() -> List[str]:
    """Get list of available TITAN backbone architectures."""
    return [
        # ResNet variants
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        # EfficientNet variants
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
        'efficientnet_b6', 'efficientnet_b7',
        # DenseNet variants
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        # VGG variants
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
        # MobileNet variants (if timm available)
        'mobilenetv3_small_100', 'mobilenetv3_medium_100'
    ]


def get_available_cellprofiler_modules() -> List[str]:
    """Get list of available CellProfiler feature modules."""
    return [
        'morphological',     # Morphological features
        'texture',          # Texture features (GLCM, LBP, Gabor)
        'intensity',        # Intensity statistics
        'color',           # Color-based features
        'nuclei',          # Nuclei-specific features
        'cells',           # Cell-specific features
        'cytoplasm',       # Cytoplasm features
        'membrane'         # Membrane features
    ]


def create_custom_config(
    extract_cellprofiler: bool = True,
    extract_titan: bool = True,
    cellprofiler_modules: Optional[List[str]] = None,
    titan_backbone: str = 'resnet50',
    titan_feature_dim: int = 1024,
    image_size: Optional[Tuple[int, int]] = None,
    device: str = 'auto',
    **kwargs
) -> PathologyConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        extract_cellprofiler: Whether to extract CellProfiler features
        extract_titan: Whether to extract TITAN features
        cellprofiler_modules: List of CellProfiler modules to extract
        titan_backbone: TITAN backbone architecture
        titan_feature_dim: Dimension of TITAN features
        image_size: Image resize size
        device: Computation device
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom PathologyConfig object
    """
    config = PathologyConfig()
    
    # Set main extraction flags
    config.extract_cellprofiler_features = extract_cellprofiler
    config.extract_titan_features = extract_titan
    
    # Set CellProfiler parameters
    if cellprofiler_modules is not None:
        config.cellprofiler_modules = cellprofiler_modules
    
    # Set TITAN parameters
    config.titan_backbone = titan_backbone
    config.titan_feature_dim = titan_feature_dim
    
    # Set image size
    if image_size is not None:
        config.image_size = image_size
    
    # Set device
    if device is not None:
        config.device = device
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config