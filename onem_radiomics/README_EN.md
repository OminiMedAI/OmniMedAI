# onem_radiomics

Medical Imaging Radiomics Feature Extraction Module - Extract comprehensive radiomics features from NIfTI format images and masks, and save them to CSV files.

## Features

### 1. Comprehensive Feature Extraction
- **First-order Statistics**: Mean, variance, skewness, kurtosis, and other descriptive statistics
- **Texture Features**: GLCM, GLRLM, GLSZM, GLTDM, NGTDM texture features
- **Shape Features**: Volume, surface area, compactness, and other shape descriptors
- **Flexible Configuration**: Selectable feature types and parameter settings

### 2. Image Preprocessing
- **Resampling**: Support for custom pixel spacing resampling
- **Interpolation Methods**: Multiple interpolation methods (nearest neighbor, linear, B-spline, etc.)
- **Normalization**: Optional intensity value normalization
- **Binning**: Configurable histogram bin width

### 3. Batch Processing Capabilities
- **Parallel Processing**: Support for multi-threaded parallel feature extraction
- **Batch Matching**: Automatic matching of image and mask files
- **Progress Tracking**: Detailed processing progress and logging
- **Error Handling**: Robust error handling and recovery mechanisms

### 4. Configuration Management
- **Preset Configurations**: Preset configurations for different imaging modalities (CT, MRI, PET, etc.)
- **Custom Configuration**: Flexible parameter configuration and validation
- **Configuration Saving**: Save and load configuration parameters
- **Parameter Validation**: Automatic parameter validity checking

## Installation

```bash
# Core dependencies
pip install numpy pandas

# Medical image processing
pip install nibabel SimpleITK

# Radiomics feature extraction
pip install pyradiomics

# Visualization and analysis (optional)
pip install matplotlib seaborn

# Development and testing
pip install unittest
```

## Quick Start

### 1. Basic Feature Extraction

```python
from onem_radiomics import RadiomicsExtractor, RadiomicsConfig

# Create extractor with default configuration
extractor = RadiomicsExtractor()

# Extract features for a single image
result = extractor.extract_features(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    patient_id="patient001"
)

print(f"Successfully extracted {len(result['features'])} features")
print("Sample features:")
for feature_name, value in list(result['features'].items())[:5]:
    print(f"  {feature_name}: {value:.4f}")
```

### 2. Batch Feature Extraction

```python
# Batch extraction and save to CSV
df = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks", 
    output_csv_path="output/radiomics_features.csv",
    file_pattern="*.nii.gz",
    n_jobs=4  # Use 4 parallel processes
)

print(f"Successfully processed data for {len(df)} patients")
print(f"Number of extracted features: {len(df.columns) - 3}")  # Subtract ID and path columns
```

### 3. Using Preset Configurations

```python
from onem_radiomics import PRESET_CONFIGS

# Use CT lung cancer preset configuration
config = PRESET_CONFIGS['ct_lung']
extractor = RadiomicsExtractor(config)

print(f"Feature types: {config.feature_types}")
print(f"Bin width: {config.bin_width}")
print(f"Resampled pixel spacing: {config.resampled_pixel_spacing}")
```

### 4. Custom Configuration

```python
from onem_radiomics import RadiomicsConfig, create_custom_config

# Create custom configuration
custom_config = create_custom_config(
    feature_types=['firstorder', 'glcm', 'glrlm'],
    bin_width=16,
    resampled_pixel_spacing=(1.5, 1.5, 3.0),
    interpolator='sitkLinear',
    normalize=True,
    n_jobs=8
)

extractor = RadiomicsExtractor(custom_config)

# Save configuration
custom_config.save("custom_config.json")

# Load configuration from file
loaded_config = RadiomicsConfig.load("custom_config.json")
```

### 5. Feature Analysis and Reporting

```python
from onem_radiomics.utils.radiomics_utils import (
    find_constant_features, 
    find_highly_correlated_features,
    create_feature_selection_report
)

# Find constant features (zero variance)
constant_features = find_constant_features(df)
print(f"Number of constant features: {len(constant_features)}")

# Find highly correlated feature pairs
high_corr_pairs = find_highly_correlated_features(df, threshold=0.95)
print(f"Number of highly correlated feature pairs: {len(high_corr_pairs)}")

# Generate feature selection report
report = create_feature_selection_report(df)
with open("feature_analysis_report.txt", "w") as f:
    f.write(report)
```

## Directory Structure

```
onem_radiomics/
├── __init__.py                    # Main entry file
├── extractors/                     # Feature extractors
│   ├── __init__.py
│   └── radiomics_extractor.py     # Radiomics feature extractor
├── config/                         # Configuration management
│   ├── __init__.py
│   └── settings.py                # Configuration classes and presets
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── file_utils.py              # File operation utilities
│   └── radiomics_utils.py        # Radiomics utilities
├── example_usage.py                # Usage examples
├── test_basic.py                   # Basic tests
└── README.md                      # Documentation
```

## API Documentation

### RadiomicsExtractor

```python
class RadiomicsExtractor:
    def __init__(self, config: Optional[RadiomicsConfig] = None)
    
    def extract_features(self, image_path: str, mask_path: str, 
                        patient_id: Optional[str] = None) -> Dict
    
    def batch_extract_features(self, images_dir: str, masks_dir: str,
                            output_csv_path: str, file_pattern: str = "*.nii.gz",
                            n_jobs: int = 1) -> pd.DataFrame
    
    def get_feature_descriptions(self) -> Dict[str, str]
    
    def validate_extraction_setup(self, test_image_path: str, 
                                 test_mask_path: str) -> Dict
```

### RadiomicsConfig

```python
class RadiomicsConfig:
    # Image preprocessing parameters
    resampled_pixel_spacing: Optional[Tuple[float, float, float]]
    interpolator: str = 'sitkBSpline'
    bin_width: int = 25
    normalize: bool = False
    normalize_scale: int = 100
    
    # Feature selection
    feature_types: List[str] = [
        'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'
    ]
    
    # Processing parameters
    n_jobs: int = 1
    verbose: bool = True
    
    def validate(self)
    def update(self, **kwargs)
    def save(self, file_path: str)
    @classmethod
    def load(cls, file_path: str)
```

## Preset Configurations

### Available Presets

```python
from onem_radiomics import PRESET_CONFIGS

# Standard configuration
config = PRESET_CONFIGS['standard']

# Comprehensive configuration (all feature types)
config = PRESET_CONFIGS['comprehensive']

# CT lung cancer
config = PRESET_CONFIGS['ct_lung']

# MRI brain tumor
config = PRESET_CONFIGS['mri_brain']

# PET tumor
config = PRESET_CONFIGS['pet_tumor']

# Fast processing (fewer features)
config = PRESET_CONFIGS['fast_processing']

# High quality processing (fine parameters)
config = PRESET_CONFIGS['high_quality']

# Texture focused
config = PRESET_CONFIGS['texture_focused']

# Shape focused
config = PRESET_CONFIGS['shape_focused']

# Research grade configuration
config = PRESET_CONFIGS['research']
```

### Preset Configuration Comparison

| Configuration | Feature Types | Bin Width | Resampling | Normalization | Use Case |
|---------------|--------------|-----------|------------|---------------|----------|
| standard | 4 basic | 25 | - | No | Daily use |
| comprehensive | 7 all | 16 | - | Yes | Research analysis |
| ct_lung | 5 texture | 25 | (1,1,1) | No | CT lung cancer |
| mri_brain | 4 basic | 16 | (1,1,1) | Yes | MRI brain tumor |
| pet_tumor | 6 texture | 32 | (4,4,4) | No | PET tumor |
| fast_processing | 2 basic | 50 | (2,2,2) | No | Fast processing |
| high_quality | 7 all | 8 | (0.5,0.5,0.5) | Yes | High quality |

## File Naming Conventions

### Image-Mask Matching Rules

The module supports multiple image and mask file naming conventions:

1. **Suffix Matching**: `patient001.nii.gz` ↔ `patient001_mask.nii.gz`
2. **Segmentation Suffix**: `patient001.nii.gz` ↔ `patient001_seg.nii.gz`
3. **Same Name**: `patient001.nii.gz` ↔ `patient001.nii.gz` (different directories)
4. **Modality Suffix**: `patient001_CT.nii.gz` ↔ `patient001_mask.nii.gz`
5. **Common Suffixes**: `patient001.nii.gz` ↔ `patient001_{mask,seg,roi,label}.nii.gz`

### Directory Structure Example

```
data/
├── images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── patient003.nii.gz
└── masks/
    ├── patient001_mask.nii.gz
    ├── patient002_mask.nii.gz
    └── patient003_mask.nii.gz
```

## Output Format

### CSV File Structure

The generated CSV file contains the following columns:

- **patient_id**: Patient identifier
- **image_path**: Image file path
- **mask_path**: Mask file path
- **Feature columns**: All extracted radiomics features
- **Metadata columns**: Metadata information prefixed with `meta_`

### Example CSV Output

```csv
patient_id,image_path,mask_path,Original_Firstorder_Mean,Original_GLCM_Correlation,...,meta_feature_types
patient001,/path/to/img1.nii.gz,/path/to/mask1.nii.gz,45.23,0.78,...,"['firstorder','glcm']"
patient002,/path/to/img2.nii.gz,/path/to/mask2.nii.gz,38.91,0.65,...,"['firstorder','glcm']"
```

### Feature Summary File

A `_summary.txt` file is also generated, containing:

- Total number of patients processed
- Total number of features extracted
- Feature types used
- Extraction parameter settings

## Performance Optimization

### 1. Parallel Processing

```python
# Use multi-processing to speed up processing
extractor = RadiomicsExtractor(config)
df = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks",
    output_csv_path="output/features.csv",
    n_jobs=8  # Use 8 CPU cores
)
```

### 2. Memory Optimization

```python
# Reduce feature types to lower memory usage
config = RadiomicsConfig(feature_types=['firstorder'])

# Use larger bin width to reduce feature count
config.bin_width = 50

# Lower resampling resolution
config.resampled_pixel_spacing = (2.0, 2.0, 2.0)
```

### 3. Processing Time Optimization

```python
# Fast processing preset
config = PRESET_CONFIGS['fast_processing']

# Or custom fast configuration
fast_config = create_custom_config(
    feature_types=['firstorder'],  # Only first-order features
    bin_width=100,                 # Large bin width
    resampled_pixel_spacing=(3, 3, 3),  # Low resolution
    n_jobs=8                      # High parallelism
)
```

## Testing

Run basic tests:

```bash
cd onem_radiomics
python test_basic.py
```

Run examples:

```bash
python example_usage.py
```

## Frequently Asked Questions

### Q: How to choose appropriate bin width?
A: Bin width selection depends on:
- **Data dynamic range**: Use larger values for wide ranges (25-50)
- **Noise level**: Use larger values for noisy data to reduce noise impact
- **Application needs**: Use smaller values for fine texture analysis (8-16)

### Q: Which feature types are most commonly used?
A: Depending on the application:
- **Basic research**: `firstorder` + `glcm` + `glrlm`
- **Comprehensive analysis**: Add `glszm` + `gldm`
- **Morphological studies**: Add `shape` features
- **Advanced texture**: Add `ngtdm`

### Q: How to handle missing data?
A: The module will:
- Automatically skip image-mask pairs that cannot be loaded
- Convert NaN/Inf values to None
- Record missing data statistics in the report

### Q: How to reduce feature redundancy?
A: You can use the following strategies:
```python
# Find constant features
constant_features = find_constant_features(df)

# Find highly correlated features
high_corr_pairs = find_highly_correlated_features(df, threshold=0.95)

# Generate detailed feature analysis report
report = create_feature_selection_report(df)
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
- Implemented comprehensive radiomics feature extraction
- Support for multiple feature types and preprocessing options
- Implemented batch processing and parallel computation
- Added complete configuration management system
- Provided multiple preset configuration templates
- Implemented file utilities and feature analysis tools
- Provided complete examples and tests