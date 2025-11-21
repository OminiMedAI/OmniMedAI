# onem_path

Pathology Image Feature Extraction Module - Implements CellProfiler-based pathology radiomics feature extraction and TITAN model deep transfer learning feature extraction for pathology image directories.

## Features

### 1. Traditional Pathology Radiomics Feature Extraction
- **Morphological Features**: Area, perimeter, circularity, shape descriptors, etc.
- **Texture Features**: GLCM, LBP, Gabor filter, and other texture features
- **Intensity Features**: Statistical features, percentiles, skewness, kurtosis, etc.
- **Color Features**: RGB, HSV, Lab color space features
- **Nuclei Features**: Nuclei-specific morphological features
- **Cell Features**: Complete cell morphological features
- **Cytoplasm Features**: Cytoplasm region features
- **Membrane Features**: Membrane features

### 2. Deep Transfer Learning Feature Extraction
- **TITAN Model**: Transfer learning model specifically designed for pathology images
- **Multiple Backbones**: ResNet, EfficientNet, DenseNet, VGG, etc.
- **Attention Mechanisms**: Multi-head attention for enhanced feature representation
- **Hierarchical Features**: Support for extracting features from different model layers
- **Pretrained Weights**: Support for ImageNet pretrained weights
- **Checkpoint Loading**: Support for loading trained model checkpoints

### 3. Image Preprocessing
- **Stain Normalization**: Reinhard, Macenko, Vahadane stain normalization
- **Contrast Enhancement**: CLAHE, histogram equalization, gamma correction
- **Noise Reduction**: Gaussian filtering, bilateral filtering, median filtering
- **Size Standardization**: Resampling with multiple interpolation methods
- **Color Space Conversion**: RGB, HSV, Lab color space conversion

### 4. Configuration Management
- **Preset Configurations**: Preset parameters for different use cases
- **Custom Configuration**: Flexible parameter customization and validation
- **Modular Extraction**: Selective extraction of specific feature types
- **Quality Control**: Input validation and image quality checking
- **Performance Optimization**: Batch processing, parallel computation, memory management

## Installation

```bash
# Core dependencies
pip install numpy pandas

# Image processing
pip install Pillow scikit-image

# Deep learning framework
pip install torch torchvision

# Visualization (optional)
pip install matplotlib seaborn

# Traditional feature extraction (optional, for full CellProfiler functionality)
pip install cellprofiler

# OpenCV (optional, for additional image processing)
pip install opencv-python

# Development and testing
pip install unittest
```

## Quick Start

### 1. Traditional Pathology Radiomics Feature Extraction

```python
from onem_path import CellProfilerExtractor

# Create extractor
extractor = CellProfilerExtractor()

# Extract features from single image
result = extractor.extract_features(
    image_path="pathology/patient001.tif",
    output_path="features/patient001_features.json",
    modules=['morphological', 'texture', 'intensity', 'color', 'nuclei']
)

print(f"Successfully extracted {len(result['features'])} traditional features")
print("Module results:", result['module_results'])
```

### 2. Batch Traditional Feature Extraction

```python
# Batch extraction
df = extractor.extract_batch_features(
    image_dir="data/pathology",
    output_dir="output/cellprofiler_features",
    file_pattern="*.tif *.tiff",
    modules=['morphological', 'texture', 'intensity', 'nuclei'],
    n_jobs=4
)

print(f"Successfully processed {len(df)} images")
print(f"Feature columns: {len([col for col in df.columns if col.startswith('morphological_') or col.startswith('texture_')])}")
```

### 3. TITAN Deep Feature Extraction

```python
from onem_path import TITANExtractor, PathologyConfig, PRESET_CONFIGS

# Use preset configuration
config = PRESET_CONFIGS['high_quality']
extractor = TITANExtractor(config)

# Extract deep features from single image
result = extractor.extract_features(
    image_path="pathology/patient001.tif",
    output_path="features/patient001_titan_features.json"
)

print(f"Successfully extracted {len(result['features'])} deep features")
print("Model information:", result['model_info'])
```

### 4. TITAN Batch Deep Feature Extraction

```python
# GPU accelerated batch extraction
gpu_config = PRESET_CONFIGS['gpu_accelerated']
extractor = TITANExtractor(gpu_config)

df = extractor.extract_batch_features(
    image_dir="data/pathology",
    output_dir="output/titan_features",
    batch_size=32,
    num_workers=4
)

print(f"GPU batch processing completed, feature dimension: {len([col for col in df.columns if col.startswith('titan_feature_')])}")
```

### 5. Combined Feature Extraction

```python
# Simultaneous traditional and deep feature extraction
cellprofiler_extractor = CellProfilerExtractor()
titan_extractor = TITANExtractor()

# Create output structure
from onem_path.utils.file_utils import create_output_structure
output_structure = create_output_structure("output/combined_features")

image_files = get_pathology_files("data/pathology")[:5]  # Process first 5 images
combined_results = []

for image_path in image_files:
    # Extract traditional features
    cp_result = cellprofiler_extractor.extract_features(
        image_path=image_path,
        modules=['morphological', 'texture', 'nuclei']
    )
    
    # Extract deep features
    titan_result = titan_extractor.extract_features(image_path=image_path)
    
    # Combine results
    combined = {
        'image_path': image_path,
        'cellprofiler_features': cp_result['features'],
        'titan_features': titan_result['features'],
        'total_features': len(cp_result['features']) + len(titan_result['features'])
    }
    combined_results.append(combined)

print(f"Combined feature extraction completed: {len(combined_results)} images")
```

### 6. Image Preprocessing

```python
from onem_path.utils.image_utils import preprocess_pathology_image

# Stain normalization preprocessing
preprocessed = preprocess_pathology_image(
    image_path="pathology/h&e_stained.tif",
    config={
        'normalize_staining': True,
        'staining_method': 'reinhard',
        'enhance_contrast': True,
        'contrast_method': 'clahe',
        'target_size': (512, 512),
        'color_conversion': 'hsv'
    }
)
```

### 7. Model Performance Benchmarking

```python
from onem_path.utils.titan_utils import benchmark_titan_extraction, get_available_backbones

# Get available backbones
backbones = get_available_backbones()[:3]  # Test first 3
print(f"Available backbones: {backbones}")

# Benchmark testing
for backbone in backbones:
    from onem_path.models.titan_model import create_titan_model
    
    model = create_titan_model(backbone_name=backbone)
    benchmark_result = benchmark_titan_extraction(
        model=model,
        num_iterations=100
    )
    
    print(f"{backbone}:")
    print(f"  FPS: {benchmark_result['fps']:.1f}")
    print(f"  Memory usage: {benchmark_result['memory_used_gb']:.2f} GB")
    print(f"  Feature dimension: {benchmark_result['feature_dim']}")
```

## Directory Structure

```
onem_path/
├── __init__.py                          # Main entry file
├── extractors/                           # Feature extractors
│   ├── __init__.py
│   ├── cellprofiler_extractor.py     # CellProfiler feature extractor
│   └── titan_extractor.py             # TITAN deep feature extractor
├── models/                               # Model management
│   ├── __init__.py
│   └── titan_model.py                 # TITAN model implementation
├── config/                              # Configuration management
│   ├── __init__.py
│   └── settings.py                     # Configuration classes and presets
├── utils/                               # Utility functions
│   ├── __init__.py
│   ├── file_utils.py                   # File operation utilities
│   ├── image_utils.py                  # Image processing utilities
│   ├── cellprofiler_utils.py            # CellProfiler utilities
│   └── titan_utils.py                  # TITAN model utilities
├── example_usage.py                      # Usage examples
├── test_basic.py                        # Basic tests
└── README.md                           # Documentation
```

## API Documentation

### CellProfilerExtractor

```python
class CellProfilerExtractor:
    def __init__(self, config=None)
    
    def extract_features(self, image_path: str, output_path: str = None,
                        modules: List[str] = None) -> Dict
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             file_pattern: str = "*.jpg *.png *.tif *.tiff",
                             modules: List[str] = None, n_jobs: int = 1) -> pd.DataFrame
    
    def get_module_results(self, result: Dict) -> Dict
```

### TITANExtractor

```python
class TITANExtractor:
    def __init__(self, config=None)
    
    def extract_features(self, image_path: str, output_path: str = None) -> Dict
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame
    
    def extract_layer_features(self, image_path: str, layers: List[str] = None) -> Dict
    
    def benchmark_model(self, input_shape: Tuple = (3, 224, 224),
                       num_iterations: int = 100) -> Dict
```

### TITANModel

```python
class TITANModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 1024,
                 use_attention: bool = True)
    
    def extract_features(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor
    
    def get_model_info(self) -> Dict
```

### PathologyConfig

```python
class PathologyConfig:
    # Image preprocessing parameters
    resize_images: bool = True
    image_size: Tuple[int, int] = (224, 224)
    normalize_images: bool = True
    color_conversion: str = 'rgb'
    
    # CellProfiler parameters
    extract_cellprofiler_features: bool = True
    cellprofiler_modules: List[str] = ['morphological', 'texture', 'intensity', 'nuclei']
    min_object_size: int = 50
    
    # TITAN parameters
    extract_titan_features: bool = True
    titan_backbone: str = 'resnet50'
    titan_feature_dim: int = 1024
    titan_use_attention: bool = True
    titan_pretrained: bool = True
    
    # Performance parameters
    device: str = 'auto'
    batch_size_titan: int = 32
    use_mixed_precision: bool = False
```

## Preset Configurations

### Available Presets

```python
from onem_path import PRESET_CONFIGS

# Default configuration - balanced traditional+deep features
config = PRESET_CONFIGS['default']

# Traditional features only
config = PRESET_CONFIGS['cellprofiler_only']

# Deep features only
config = PRESET_CONFIGS['titan_only']

# High quality configuration
config = PRESET_CONFIGS['high_quality']

# Fast processing configuration
config = PRESET_CONFIGS['fast_processing']

# Research grade configuration
config = PRESET_CONFIGS['research_grade']

# GPU accelerated configuration
config = PRESET_CONFIGS['gpu_accelerated']

# Memory efficient configuration
config = PRESET_CONFIGS['memory_efficient']

# Comprehensive configuration
config = PRESET_CONFIGS['comprehensive']
```

### Preset Configuration Comparison

| Configuration | Traditional | Deep | Backbone | Feature Dim | Use Case |
|-------------|------------|--------|------------|----------|
| default | ✅ | ✅ | resnet50 | 1024 | Daily use |
| cellprofiler_only | ✅ | ❌ | - | - | Traditional analysis |
| titan_only | ❌ | ✅ | resnet50 | 1024 | Deep learning |
| high_quality | ✅ | ✅ | resnet101 | 2048 | High quality analysis |
| fast_processing | ✅ | ❌ | - | - | Fast processing |
| research_grade | ✅ | ✅ | resnet152 | 2048 | Research purposes |
| gpu_accelerated | ❌ | ✅ | resnet50 | 1024 | GPU batch processing |
| memory_efficient | ❌ | ✅ | efficientnet_b0 | 512 | Memory limited |
| comprehensive | ✅ | ✅ | resnet101 | 2048 | Comprehensive analysis |

## Supported Image Formats

### Input Formats
- **TIFF/TIFF**: Lossless format commonly used in pathology
- **PNG**: Supports transparency channel
- **JPEG/JPG**: Lossy compression format
- **BMP**: Bitmap format
- **SVS**: Aperio scanner format (requires additional libraries)

### Output Formats
- **JSON**: Structured data with metadata
- **CSV**: Tabular format for data analysis
- **HDF5**: Large-scale data storage (requires h5py)

## Feature Types Details

### Morphological Features
```python
# Cell nuclei morphological features example
morphological_features = {
    'num_objects': 125,                    # Number of objects
    'avg_object_area': 156.3,              # Average area
    'max_object_area': 892.1,              # Maximum area
    'avg_circularity': 0.73,               # Average circularity
    'avg_solidity': 0.85,                 # Average solidity
    'avg_eccentricity': 0.42,             # Average eccentricity
    'binary_fill_ratio': 0.15               # Fill ratio
}
```

### Texture Features
```python
# GLCM texture features example
texture_features = {
    'glcm_contrast_mean': 0.234,           # Contrast mean
    'glcm_correlation_mean': 0.567,         # Correlation mean
    'glcm_homogeneity_mean': 0.891,         # Homogeneity mean
    'glcm_energy_mean': 0.145,              # Energy mean
    'lbp_energy': 0.067,                    # LBP energy
    'lbp_entropy': 2.345,                    # LBP entropy
    'gabor_mean': 0.128,                     # Gabor filter response mean
}
```

### Deep Features
```python
# TITAN deep features example (1024-dim)
deep_features = {
    'titan_feature_0': 0.234,               # First feature value
    'titan_feature_1': -0.567,              # Second feature value
    # ... 1024 feature values total
    'titan_feature_1023': 0.891,            # Last feature value
}
```

## Performance Optimization

### 1. GPU Acceleration
```python
# GPU configuration
gpu_config = {
    'device': 'cuda',
    'batch_size_titan': 64,
    'num_workers_titan': 8,
    'use_mixed_precision': True
}

extractor = TITANExtractor(gpu_config)
```

### 2. Memory Optimization
```python
# Memory efficient configuration
memory_config = {
    'titan_backbone': 'efficientnet_b0',    # Lightweight backbone
    'titan_feature_dim': 512,              # Reduce feature dimension
    'batch_size_titan': 16,                  # Small batch size
    'memory_limit_gb': 4.0                   # Memory limit
}
```

### 3. Batch Processing Optimization
```python
# Parallel processing
df = extractor.extract_batch_features(
    image_dir="data/pathology",
    n_jobs=8,                              # Use 8 processes
    batch_size_titan=64                      # TITAN batch size
)
```

## Testing

Run basic tests:

```bash
cd onem_path
python test_basic.py
```

Run examples:

```bash
python example_usage.py
```

## Frequently Asked Questions

### Q: How to choose the appropriate feature extraction method?
A: Choose based on research purpose:
- **Traditional methods**: Suitable for research requiring interpretability
- **Deep learning methods**: Suitable for research requiring high accuracy
- **Combined methods**: Combine both advantages for comprehensive information

### Q: How to handle different staining methods?
A: Use stain normalization:
```python
config = {
    'normalize_staining': True,
    'staining_method': 'reinhard'  # H&E staining
}
preprocessed = preprocess_pathology_image(image_path, config)
```

### Q: How to optimize large-scale data processing?
A: Use the following strategies:
- GPU acceleration: Use `gpu_accelerated` preset
- Memory management: Use `memory_efficient` preset
- Batch processing: Adjust `batch_size_titan` and `n_jobs`
- Feature selection: Reduce number of features

### Q: How to customize feature extraction?
A: Customize through configuration parameters:
```python
# Custom CellProfiler modules
modules = ['morphological', 'texture', 'intensity', 'nuclei', 'cells']

# Custom TITAN model
config = {
    'titan_backbone': 'resnet101',    # Use ResNet-101
    'titan_feature_dim': 2048,     # Increase feature dimension
    'titan_use_attention': True       # Enable attention mechanism
}
```

### Q: How to handle different sized pathology images?
A: Automatic size adjustment:
```python
config = {
    'resize_images': True,
    'image_size': (512, 512),     # Uniform size
    'normalize_intensity': True       # Intensity normalization
}
```

## Acknowledgements

For the professional version, please contact:
- **WeChat**: AcePwn
- **Email**: acezqy@gmail.com

## License

This project is licensed under the MIT License.

## Contributing

Issues and Pull Requests are welcome to improve this module.

## Changelog

### v1.0.0
- Initial release
- Implemented CellProfiler-based traditional pathology radiomics feature extraction
- Implemented TITAN deep transfer learning feature extraction model
- Support for multiple backbones and attention mechanisms
- Implemented complete image preprocessing pipeline
- Implemented flexible configuration management system
- Provided multiple preset configuration templates
- Implemented stain normalization and image enhancement
- Support for batch processing and parallel computation
- Provided complete examples and tests