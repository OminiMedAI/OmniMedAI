# onem_process

Medical image processing module featuring DICOM to NIfTI format conversion, batch format conversion, and 3D ROI extraction.

## Features

### 1. Format Conversion
- **DICOM to NIfTI Conversion**: Support for both single DICOM files and DICOM series conversion
- **Batch Format Conversion**: Automatically identify and convert multiple formats in images/masks directories
- **Smart File Matching**: Automatically match images with corresponding mask files

### 2. 3D ROI Processing
- **ROI Extraction**: Extract 3D regions of interest based on masks
- **Batch Processing**: Support for batch ROI extraction
- **ROI Statistics**: Calculate volume, centroid, intensity statistics, and more
- **Bounding Box Expansion**: Configurable ROI boundary expansion

### 3. Image Processing
- **Resampling**: Adjust image resolution
- **Normalization**: Z-score, Min-Max, percentile normalization
- **Crop/Pad**: Center cropping and boundary padding
- **Window/Level**: CT image window/level adjustment

### 4. Configuration Management
- **Flexible Configuration**: Support for processing and conversion parameter configuration
- **Configuration Import/Export**: Save and load configuration files
- **Preset Templates**: Provide commonly used configuration templates

## Installation

```bash
# Core dependencies
pip install numpy nibabel pydicom

# Image processing dependencies
pip install opencv-python scipy

# Advanced functionality dependencies (optional)
pip install SimpleITK

# Development and testing dependencies
pip install unittest
```

## Quick Start

### 1. Single DICOM File Conversion

```python
from onem_process.converters import DicomToNiftiConverter

# Create converter
converter = DicomToNiftiConverter(output_dir="output/nifti")

# Convert single file
nifti_file = converter.convert_single_dicom("data/patient001.dcm")
print(f"Conversion completed: {nifti_file}")
```

### 2. DICOM Series Conversion

```python
# Convert DICOM series
nifti_file = converter.convert_dicom_series("data/dicom_series/")
print(f"Series conversion completed: {nifti_file}")
```

### 3. Batch Dataset Conversion

```python
from onem_process.converters import BatchConverter

# Create batch converter
converter = BatchConverter(
    base_dir="data/dataset",
    output_base_dir="output/converted"
)

# Convert entire dataset
converted_images, converted_masks = converter.convert_dataset(
    images_dir="images",
    masks_dir="masks"
)

# Get conversion summary
summary = converter.get_conversion_summary(converted_images, converted_masks)
print(f"Converted {summary['total_images']} image files")
print(f"Converted {summary['total_masks']} mask files")
```

### 4. ROI Extraction

```python
from onem_process.processors import ROIProcessor

# Create ROI processor
roi_processor = ROIProcessor(padding=(10, 10, 10))

# Extract single ROI
roi_image, roi_mask = roi_processor.extract_roi_from_mask(
    image_path="output/converted/images_nifti/patient001.nii.gz",
    mask_path="output/converted/masks_nifti/patient001_mask.nii.gz",
    output_dir="output/rois",
    roi_name="patient001_roi"
)

print(f"ROI image: {roi_image}")
print(f"ROI mask: {roi_mask}")

# Get ROI statistics
stats = roi_processor.get_roi_statistics(roi_mask)
print(f"Volume: {stats['actual_volume_mm3']:.2f} mm³")
print(f"Centroid: {stats['centroid']}")
```

### 5. Batch ROI Extraction

```python
# Batch extract ROI
results = roi_processor.batch_extract_rois(
    image_dir="output/converted/images_nifti",
    mask_dir="output/converted/masks_nifti", 
    output_dir="output/batch_rois",
    file_pattern="*.nii.gz"
)

print(f"Successfully processed {len(results)} files")
```

### 6. Image Processing

```python
from onem_process.processors import ImageProcessor

# Create image processor
processor = ImageProcessor()

# Resampling
resampled = processor.resample_image(
    "input.nii.gz",
    target_spacing=(1.0, 1.0, 1.0),
    output_path="resampled.nii.gz"
)

# Normalization
normalized = processor.normalize_image(
    "resampled.nii.gz",
    method='z_score',
    output_path="normalized.nii.gz"
)

# Center cropping
cropped = processor.crop_image(
    "normalized.nii.gz",
    crop_size=(128, 128, 128),
    output_path="cropped.nii.gz"
)
```

### 7. Configuration Management

```python
from onem_process.config import ConfigManager, ProcessingConfig

# Create configuration manager
config_manager = ConfigManager("config")

# Get configuration
processing_config = config_manager.get_processing_config()
print(f"Current normalization method: {processing_config.normalize_method}")

# Update configuration
config_manager.update_processing_config(
    normalize_method='percentile',
    resample_spacing=(0.5, 0.5, 0.5),
    roi_padding=(20, 20, 20)
)

# Export configuration
config_manager.export_configs("output/configs")
```

## Directory Structure

```
onem_process/
├── __init__.py              # Main entry point
├── converters/               # Format converters
│   ├── __init__.py
│   ├── dicom_to_nifti.py   # DICOM converter
│   └── batch_converter.py   # Batch converter
├── processors/               # Image processors
│   ├── __init__.py
│   ├── roi_processor.py     # ROI processor
│   └── image_processor.py   # Image processor
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── file_utils.py        # File utilities
│   └── medical_utils.py     # Medical utilities
├── config/                   # Configuration module
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── example_usage.py         # Usage examples
├── test_basic.py           # Basic tests
└── README.md               # Documentation
```

## API Documentation

### DicomToNiftiConverter

```python
class DicomToNiftiConverter:
    def __init__(self, output_dir: Optional[str] = None)
    def convert_single_dicom(self, dicom_path: str, output_path: Optional[str] = None) -> str
    def convert_dicom_series(self, dicom_dir: str, output_path: Optional[str] = None) -> str
    def batch_convert(self, input_paths: List[str], output_dir: Optional[str] = None) -> List[str]
```

### BatchConverter

```python
class BatchConverter:
    def __init__(self, base_dir: str, output_base_dir: Optional[str] = None)
    def convert_dataset(self, images_dir: str = "images", masks_dir: str = "masks", 
                       skip_existing: bool = True) -> Tuple[Dict, Dict]
    def get_conversion_summary(self, converted_images: Dict, converted_masks: Dict) -> Dict
```

### ROIProcessor

```python
class ROIProcessor:
    def __init__(self, padding: Tuple[int, int, int] = (10, 10, 10))
    def extract_roi_from_mask(self, image_path: str, mask_path: str, 
                             output_dir: str, roi_name: Optional[str] = None) -> Tuple[str, str]
    def batch_extract_rois(self, image_dir: str, mask_dir: str, 
                          output_dir: str, file_pattern: str = "*.nii.gz") -> List[Dict]
    def get_roi_statistics(self, roi_path: str) -> Dict
```

### ImageProcessor

```python
class ImageProcessor:
    def resample_image(self, image_path: str, target_spacing: Tuple[float, float, float],
                      output_path: str, interpolation: str = 'linear') -> str
    def normalize_image(self, image_path: str, method: str = 'z_score',
                       output_path: Optional[str] = None) -> str
    def crop_center(self, image_path: str, crop_size: Tuple[int, int, int],
                   output_path: str) -> str
    def pad_image(self, image_path: str, target_shape: Tuple[int, int, int],
                  output_path: str, pad_value: Union[float, str] = 'min') -> str
```

## Configuration Options

### ProcessingConfig

- `normalize_method`: Normalization method ('z_score', 'min_max', 'percentile')
- `resample_spacing`: Resampling spacing (z, y, x)
- `roi_padding`: ROI boundary expansion (z, y, x)
- `window_center/window_width`: Window/level settings

### ConversionConfig

- `dicom_output_dir`: DICOM conversion output directory
- `batch_skip_existing`: Whether to skip existing files
- `image_mask_patterns`: Image mask matching patterns
- `file_extensions`: Supported file extensions

## Preset Configuration Templates

```python
from onem_process.config.settings import CT_CHEST_CONFIG, MRI_BRAIN_CONFIG

# Use preset configuration
config_manager.update_processing_config(
    normalize_method=CT_CHEST_CONFIG.normalize_method,
    window_center=CT_CHEST_CONFIG.window_center,
    window_width=CT_CHEST_CONFIG.window_width
)
```

## Error Handling

The module includes comprehensive error handling mechanisms:

- **Dependency Check**: Automatically detect if required dependencies are installed
- **File Validation**: Validate input file formats and integrity
- **Exception Catching**: Provide detailed error messages and logging
- **Optional Imports**: Gracefully degrade when some dependencies are missing

## Performance Optimization

- **Batch Processing**: Support parallel processing of large numbers of files
- **Memory Optimization**: Use streaming processing for large files
- **Caching Mechanism**: Avoid reprocessing the same files
- **Progress Feedback**: Provide processing progress information

## Testing

Run tests:

```bash
cd onem_process
python test_basic.py
```

Run examples:

```bash
python example_usage.py
```

## License

This project is licensed under the MIT License.

## Contributing

Welcome to submit Issues and Pull Requests to improve this module.

## Changelog

### v1.0.0
- Initial release
- Implemented DICOM to NIfTI conversion functionality
- Implemented 3D ROI extraction functionality
- Added batch processing and configuration management features