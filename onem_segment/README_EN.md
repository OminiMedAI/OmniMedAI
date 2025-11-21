# onem_segment

Medical Image Automatic ROI Segmentation Module - Automatically segments ROI from NIfTI files in the images directory, selecting 2D or 3D models based on z-axis characteristics, and saves segmented ROI as NIfTI files to specified directory.

## Features

### 1. Intelligent Model Selection
- **Automatic Dimension Analysis**: Automatically determines 2D/3D processing based on z-axis characteristics (slice count, spacing, content variation)
- **Multi-dimensional Evaluation**: Comprehensive consideration of slice count, layer spacing, content variation factors
- **Confidence Assessment**: Provides confidence scores for selection recommendations
- **Batch Analysis**: Supports batch image analysis and unified recommendations

### 2. Multi-Model Support
- **2D Models**: 2D U-Net, 2D Attention U-Net, etc.
- **3D Models**: 3D U-Net, 3D V-Net, etc.
- **Quality Levels**: Default and high-quality processing modes
- **Model Management**: Unified model loading and inference interface

### 3. Flexible Preprocessing
- **Multiple Normalizations**: Z-score, Min-Max, Robust normalization
- **Image Resampling**: Support for different spacing resampling
- **Size Adjustment**: Automatic adjustment to model input dimensions
- **Contrast Enhancement**: Histogram equalization, CLAHE, etc.

### 4. Intelligent Post-processing
- **Connected Component Analysis**: Remove small connected components
- **Morphological Operations**: Opening/closing operations, hole filling
- **Boundary Optimization**: ROI boundary smoothing
- **Quality Control**: Automatic quality assessment and reporting

### 5. Configuration Management
- **Preset Configurations**: Preset parameters for different modalities and requirements
- **Custom Configuration**: Flexible parameter customization and validation
- **Configuration Saving**: Save and load configurations
- **Batch Configuration**: Unified configuration for batch processing

## Installation

```bash
# Core dependencies
pip install numpy pandas

# Medical image processing
pip install nibabel SimpleITK

# Deep learning framework
pip install torch torchvision

# Image processing
pip install scikit-image scipy

# Visualization (optional)
pip install matplotlib seaborn

# Development and testing
pip install unittest
```

## Quick Start

### 1. Basic ROI Segmentation

```python
from onem_segment import ROISegmenter

# Create segmenter with default configuration
segmenter = ROISegmenter()

# Segment a single image
result = segmenter.segment_image(
    image_path="images/patient001.nii.gz",
    output_path=None,  # Automatically generated
    model_type='auto',  # Automatic 2D/3D selection
    quality='default'
)

print(f"Segmentation completed: {result['output_path']}")
print(f"Model used: {result['model_used']}")
print(f"Processing type: {result['model_type']}")

# Get segmentation statistics
stats = segmenter.get_segmentation_statistics(result)
print(f"ROI volume: {stats['roi_volume_voxels']} voxels")
print(f"ROI percentage: {stats['roi_percentage']:.2f}%")
```

### 2. Batch Segmentation

```python
# Batch segmentation
results = segmenter.segment_batch(
    image_dir="data/images",
    output_dir="output/segmentations",
    file_pattern="*.nii.gz",
    model_type='auto',
    quality='high_quality'
)

successful = [r for r in results if 'error' not in r]
print(f"Successfully segmented: {len(successful)} images")

# Analyze model usage
model_usage = {}
for result in successful:
    model = result.get('model_used', 'unknown')
    model_usage[model] = model_usage.get(model, 0) + 1

print("Model usage statistics:")
for model, count in model_usage.items():
    print(f"  {model}: {count} images")
```

### 3. Image Dimension Analysis

```python
from onem_segment import ImageDimensionAnalyzer

# Create image analyzer
analyzer = ImageDimensionAnalyzer(
    min_3d_slices=30,
    max_slice_spacing=5.0,
    min_content_variation=0.1
)

# Analyze single image
analysis = analyzer.analyze_image("images/patient001.nii.gz")

print(f"Image shape: {analysis['shape']}")
print(f"Slice count: {analysis['n_slices']}")
print(f"Slice spacing: {analysis['slice_spacing']:.2f} mm")
print(f"Recommended processing: {'3D' if analysis['is_3d'] else '2D'}")

print("Content analysis:")
content = analysis['content_analysis']
print(f"  Mean intensity: {content['mean']:.2f}")
print(f"  Standard deviation: {content['std']:.2f}")
print(f"  Skewness: {content['skewness']:.2f}")

print("Quality metrics:")
quality = analysis['quality_metrics']
print(f"  Signal-to-noise ratio: {quality['snr']:.2f}")
print(f"  Contrast-to-noise ratio: {quality['cnr']:.2f}")
```

### 4. Using Preset Configurations

```python
from onem_segment import PRESET_CONFIGS

# Use high-quality preset
config = PRESET_CONFIGS['high_quality']
segmenter = ROISegmenter(config)

print(f"Using high-quality preset:")
print(f"  Quality level: {config.quality_level}")
print(f"  Normalization: {config.normalization}")
print(f"  Morphological operations: {config.morphological_operations}")
print(f"  Return probabilities: {config.return_probabilities}")
```

### 5. Custom Configuration

```python
from onem_segment import SegmentationConfig, create_custom_config

# Create custom configuration
custom_config = create_custom_config(
    model_type='auto',
    quality_level='high_quality',
    input_size=(512, 512),  # 2D model input size
    normalization='z_score',
    min_roi_volume=20,
    morphological_operations=True,
    return_probabilities=True,
    generate_statistics=True
)

segmenter = ROISegmenter(custom_config)

# Save configuration
custom_config.save("custom_segmentation_config.json")
```

## Directory Structure

```
onem_segment/
├── __init__.py                      # Main entry file
├── segmenters/                       # Segmentation modules
│   ├── __init__.py
│   └── roi_segmenter.py             # ROI segmenter
├── models/                          # Model management
│   ├── __init__.py
│   └── model_manager.py             # Model manager
├── config/                          # Configuration management
│   ├── __init__.py
│   └── settings.py                 # Configuration classes and presets
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── image_analyzer.py           # Image dimension analysis
│   ├── file_utils.py               # File operation utilities
│   └── preprocessing.py           # Image preprocessing
├── example_usage.py                  # Usage examples
├── test_basic.py                    # Basic tests
└── README.md                       # Documentation
```

## API Documentation

### ROISegmenter

```python
class ROISegmenter:
    def __init__(self, config=None)
    
    def segment_image(self, image_path: str, output_path: str = None,
                    model_type: str = 'auto', quality: str = 'default',
                    return_probabilities: bool = False) -> Dict
    
    def segment_batch(self, image_dir: str, output_dir: str = None,
                    file_pattern: str = "*.nii.gz", model_type: str = 'auto',
                    quality: str = 'default', n_jobs: int = 1) -> List[Dict]
    
    def get_segmentation_statistics(self, result: Dict) -> Dict
```

### ImageDimensionAnalyzer

```python
class ImageDimensionAnalyzer:
    def __init__(self, min_3d_slices: int = 30, max_slice_spacing: float = 5.0,
                 min_content_variation: float = 0.1)
    
    def analyze_image(self, image_path: str) -> Dict
    
    def batch_analyze(self, image_paths: list) -> Dict
```

### SegmentationConfig

```python
class SegmentationConfig:
    # Model selection parameters
    model_type: str = 'auto'          # 'auto', '2d', '3d'
    quality_level: str = 'default'     # 'default', 'high_quality'
    device: str = 'auto'             # 'cpu', 'cuda', 'auto'
    
    # Image analysis parameters
    min_3d_slices: int = 30
    max_slice_spacing: float = 5.0
    min_content_variation: float = 0.1
    
    # Preprocessing parameters
    normalization: str = 'z_score'
    input_size_2d: Tuple[int, int] = (256, 256)
    input_size_3d: Tuple[int, int, int] = (64, 64, 64)
    
    # Post-processing parameters
    min_roi_volume: int = 10
    morphological_operations: bool = True
    morph_iterations: int = 2
```

## Preset Configurations

### Available Presets

```python
from onem_segment import PRESET_CONFIGS

# Default configuration
config = PRESET_CONFIGS['default']

# Fast processing
config = PRESET_CONFIGS['fast']

# High quality processing
config = PRESET_CONFIGS['high_quality']

# 2D only
config = PRESET_CONFIGS['2d_only']

# 3D only
config = PRESET_CONFIGS['3d_only']

# CT organ segmentation
config = PRESET_CONFIGS['ct_organ']

# MRI brain segmentation
config = PRESET_CONFIGS['mri_brain']

# PET tumor segmentation
config = PRESET_CONFIGS['pet_tumor']

# Research grade configuration
config = PRESET_CONFIGS['research']

# Production configuration
config = PRESET_CONFIGS['production']

# Memory efficient configuration
config = PRESET_CONFIGS['memory_efficient']
```

### Preset Configuration Comparison

| Configuration | Model Selection | Quality Level | Normalization | Input Size | Use Case |
|---------------|-----------------|---------------|--------------|-------------|----------|
| default | auto | default | z_score | 256×256 / 64³ | Daily use |
| fast | auto | default | z_score | 128×128 / 32³ | Fast processing |
| high_quality | auto | high_quality | z_score | 512×512 / 128³ | High quality output |
| 2d_only | 2d | default | z_score | 256×256 | 2D processing only |
| 3d_only | 3d | default | z_score | 64×64×64 | 3D processing only |
| ct_organ | auto | high_quality | z_score | 256×256 / 64³ | CT organ segmentation |
| mri_brain | auto | high_quality | z_score | 256×256 / 96³ | MRI brain segmentation |
| pet_tumor | auto | default | min_max | 256×256 / 64³ | PET tumor segmentation |
| research | auto | high_quality | z_score | 512×512 / 256³ | Research use |
| production | auto | default | z_score | 256×256 / 64³ | Production environment |

## 2D/3D Model Selection Logic

### Selection Criteria

The system automatically selects 2D or 3D models based on the following criteria:

1. **Slice Count**: 
   - Less than `min_3d_slices` (default 30) → Select 2D
   - Equal to or greater than `min_3d_slices` → Candidate for 3D

2. **Slice Spacing**:
   - Greater than `max_slice_spacing` (default 5.0mm) → Select 2D
   - Less than or equal to `max_slice_spacing` → Candidate for 3D

3. **Content Variation**:
   - Calculate content variation between adjacent slices
   - Variation less than `min_content_variation` → Select 2D
   - Variation greater than or equal to `min_content_variation` → Candidate for 3D

4. **Comprehensive Decision**:
   - Need 2 or 3 conditions met to select 3D processing
   - Otherwise select 2D processing

### Example Scenarios

```python
# Typical CT scan (64 slices, 1.25mm spacing, high variation)
analysis = {
    'n_slices': 64,           # ✅ Meets slice count
    'slice_spacing': 1.25,     # ✅ Meets spacing
    'content_variation': 0.3,   # ✅ Meets content variation
    'is_3d': True,            # Recommend 3D model
    'confidence': 'high'
}

# Thin slice scan (15 slices, 0.5mm spacing, low variation)
analysis = {
    'n_slices': 15,           # ❌ Doesn't meet slice count
    'slice_spacing': 0.5,     # ✅ Meets spacing
    'content_variation': 0.05,  # ❌ Doesn't meet content variation
    'is_3d': False,           # Recommend 2D model
    'confidence': 'medium'
}
```

## Output Format

### Segmentation Result Files

Segmentation results are saved in NIfTI format, including:

- **Binary Mask**: Binary segmentation of ROI region (0=background, 1=foreground)
- **Probability Map**: (Optional) ROI probability values (0.0-1.0)
- **Original Header**: Maintains spatial information from input image

### File Naming Rules

```bash
# Automatic output naming
patient001.nii.gz → patient001_3d_seg.nii.gz
patient002.nii.gz → patient002_2d_seg.nii.gz

# Probability maps
patient001.nii.gz → patient001_3d_prob.nii.gz
```

### Statistical Reports

Detailed segmentation statistical reports are generated:

```json
{
  "roi_volume_voxels": 1250,
  "roi_volume_mm3": 1562.5,
  "roi_percentage": 2.45,
  "roi_shape": [512, 512, 64],
  "roi_slices": 23,
  "roi_centroid": [256.3, 247.8, 31.2],
  "bounding_box": [[200, 300], [180, 320], [15, 45]]
}
```

## Performance Optimization

### 1. Memory Optimization

```python
# Use memory efficient configuration
config = PRESET_CONFIGS['memory_efficient']
segmenter = ROISegmenter(config)

# Or custom memory configuration
config = create_custom_config(
    model_type='2d',              # Force 2D
    input_size=(128, 128),        # Reduce input size
    batch_inference=True,          # Batch inference
    batch_size=2,                 # Small batch size
    memory_limit_gb=4.0,          # Memory limit
    use_mixed_precision=True        # Mixed precision
)
```

### 2. Processing Speed Optimization

```python
# Fast processing configuration
config = PRESET_CONFIGS['fast']
segmenter = ROISegmenter(config)

# Disable post-processing for speed
config.morphological_operations = False
config.remove_small_components = False
config.generate_statistics = False
```

### 3. GPU Acceleration

```python
# Specify GPU device
config.device = 'cuda'

# Use mixed precision
config.use_mixed_precision = True

# Batch inference
config.batch_inference = True
config.batch_size = 4
```

## Testing

Run basic tests:

```bash
cd onem_segment
python test_basic.py
```

Run examples:

```bash
python example_usage.py
```

## Frequently Asked Questions

### Q: How to adjust 2D/3D selection thresholds?
A: Modify ImageDimensionAnalyzer parameters:
```python
analyzer = ImageDimensionAnalyzer(
    min_3d_slices=25,      # Lower slice count requirement
    max_slice_spacing=6.0,   # Relax spacing requirement
    min_content_variation=0.05  # Lower variation requirement
)
```

### Q: How to handle different modality images?
A: Use corresponding preset configurations:
```python
# CT images
config = PRESET_CONFIGS['ct_organ']

# MRI images
config = PRESET_CONFIGS['mri_brain']

# PET images
config = PRESET_CONFIGS['pet_tumor']
```

### Q: How to improve segmentation quality?
A: Use high-quality configuration:
```python
config = PRESET_CONFIGS['high_quality']
# Or research grade configuration
config = PRESET_CONFIGS['research']
```

### Q: How to handle large memory images?
A: Use memory optimization strategies:
```python
config = PRESET_CONFIGS['memory_efficient']
# Or force 2D processing
config.model_type = '2d'
config.input_size_2d = (128, 128)
```

### Q: How to batch process different sized images?
A: Use automatic size adjustment:
```python
config.resize_input = True  # Enable automatic resampling
config.interpolation_method = 'bilinear'  # Choose interpolation method
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
- Implemented automatic 2D/3D model selection
- Implemented multi-model support and unified interface
- Implemented flexible image preprocessing and post-processing
- Implemented complete configuration management system
- Provided multiple preset configuration templates
- Implemented image dimension analysis and batch processing
- Provided complete examples and tests