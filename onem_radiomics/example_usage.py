"""
Example usage of onem_radiomics module

This script demonstrates how to use the onem_radiomics module for extracting
radiomics features from medical images and masks.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from onem_radiomics import RadiomicsExtractor, RadiomicsConfig, PRESET_CONFIGS
from onem_radiomics.utils.file_utils import get_file_info, create_output_structure
from onem_radiomics.utils.radiomics_utils import create_feature_selection_report


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('radiomics_extraction.log')
        ]
    )


def example_basic_extraction():
    """Example 1: Basic radiomics feature extraction."""
    print("\n" + "="*60)
    print("Example 1: Basic Radiomics Feature Extraction")
    print("="*60)
    
    # Create extractor with default configuration
    extractor = RadiomicsExtractor()
    
    # Example paths (replace with your actual paths)
    image_path = "data/images/patient001.nii.gz"
    mask_path = "data/masks/patient001_mask.nii.gz"
    
    # Check if example files exist
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print("Example files not found. Please update paths or use your own data.")
        return
    
    try:
        # Extract features for a single patient
        result = extractor.extract_features(image_path, mask_path, "patient001")
        
        print(f"Successfully extracted {len(result['features'])} features")
        print(f"Patient ID: {result['patient_id']}")
        print("Sample features:")
        
        # Show first 5 features
        for i, (feature_name, value) in enumerate(result['features'].items()):
            if i >= 5:
                break
            print(f"  {feature_name}: {value:.4f}")
        
        print(f"Metadata: {result['metadata']}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")


def example_batch_extraction():
    """Example 2: Batch extraction with custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Batch Extraction with Custom Configuration")
    print("="*60)
    
    # Use preset configuration for CT lung cancer
    config = PRESET_CONFIGS['ct_lung']
    print(f"Using preset configuration: ct_lung")
    print(f"Feature types: {config.feature_types}")
    print(f"Bin width: {config.bin_width}")
    print(f"Resampled pixel spacing: {config.resampled_pixel_spacing}")
    
    # Create extractor with custom config
    extractor = RadiomicsExtractor(config)
    
    # Example directories (replace with your actual directories)
    images_dir = "data/images"
    masks_dir = "data/masks"
    output_csv = "output/radiomics_features.csv"
    
    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("Example directories not found. Please update paths or use your own data.")
        return
    
    try:
        # Create output directory structure
        output_structure = create_output_structure("output")
        print(f"Created output structure: {output_structure}")
        
        # Batch extract features
        df = extractor.batch_extract_features(
            images_dir=images_dir,
            masks_dir=masks_dir,
            output_csv_path=output_csv,
            file_pattern="*.nii.gz",
            n_jobs=2  # Use 2 parallel jobs
        )
        
        print(f"Successfully extracted features for {len(df)} patients")
        print(f"Total features extracted: {len(df.columns) - 3}")  # Subtract ID and path columns
        print(f"Results saved to: {output_csv}")
        
        # Show summary statistics
        print("\nFeature Statistics:")
        numeric_columns = df.select_dtypes(include=['number']).columns
        print(f"Numeric feature columns: {len(numeric_columns)}")
        
        if len(numeric_columns) > 0:
            print(f"Sample feature ranges:")
            for col in numeric_columns[:3]:  # Show first 3
                print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
        
        # Create feature selection report
        report = create_feature_selection_report(df)
        report_path = "output/feature_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Feature analysis report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during batch extraction: {e}")


def example_custom_configuration():
    """Example 3: Custom configuration and validation."""
    print("\n" + "="*60)
    print("Example 3: Custom Configuration and Validation")
    print("="*60)
    
    from onem_radiomics.config.settings import create_custom_config
    
    # Create custom configuration
    custom_config = create_custom_config(
        feature_types=['firstorder', 'glcm', 'glrlm'],
        bin_width=16,
        resampled_pixel_spacing=(1.5, 1.5, 3.0),
        interpolator='sitkLinear',
        normalize=True,
        n_jobs=4
    )
    
    print("Custom configuration:")
    print(f"  Feature types: {custom_config.feature_types}")
    print(f"  Bin width: {custom_config.bin_width}")
    print(f"  Resampled pixel spacing: {custom_config.resampled_pixel_spacing}")
    print(f"  Normalization: {custom_config.normalize}")
    print(f"  Parallel jobs: {custom_config.n_jobs}")
    
    # Create extractor with custom config
    extractor = RadiomicsExtractor(custom_config)
    
    # Get feature descriptions
    feature_descriptions = extractor.get_feature_descriptions()
    print(f"\nAvailable feature types: {len(feature_descriptions)}")
    print("Sample feature descriptions:")
    for i, (feature, description) in enumerate(feature_descriptions.items()):
        if i >= 5:
            break
        print(f"  {feature}: {description}")
    
    # Save configuration
    config_path = "output/custom_config.json"
    custom_config.save(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Test extraction setup (requires sample data)
    image_path = "data/images/patient001.nii.gz"
    mask_path = "data/masks/patient001_mask.nii.gz"
    
    if os.path.exists(image_path) and os.path.exists(mask_path):
        try:
            validation_result = extractor.validate_extraction_setup(image_path, mask_path)
            print(f"\nValidation results:")
            print(f"  Valid: {validation_result['valid']}")
            print(f"  Errors: {len(validation_result['errors'])}")
            print(f"  Warnings: {len(validation_result['warnings'])}")
            
            if validation_result['test_features']:
                features = validation_result['test_features']['features']
                print(f"  Test features extracted: {len(features)}")
                
        except Exception as e:
            print(f"Validation failed: {e}")
    else:
        print("Validation skipped - test files not found")


def example_file_utilities():
    """Example 4: File utilities and information."""
    print("\n" + "="*60)
    print("Example 4: File Utilities and Information")
    print("="*60)
    
    from onem_radiomics.utils.file_utils import (
        get_matching_files, get_file_info, validate_file_paths
    )
    
    # Example directories
    images_dir = "data/images"
    masks_dir = "data/masks"
    
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        # Find matching image-mask pairs
        matching_pairs = get_matching_files(images_dir, masks_dir, "*.nii.gz")
        print(f"Found {len(matching_pairs)} matching image-mask pairs")
        
        # Show first few pairs
        for i, (image_path, mask_path) in enumerate(matching_pairs[:3]):
            print(f"  {i+1}. {Path(image_path).name} ↔ {Path(mask_path).name}")
        
        # Get file information
        if matching_pairs:
            sample_image = matching_pairs[0][0]
            try:
                info = get_file_info(sample_image)
                print(f"\nFile information for {Path(sample_image).name}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Voxel size: {info['voxel_size_mm']}")
                print(f"  Data range: [{info['data_min']:.3f}, {info['data_max']:.3f}]")
                print(f"  Non-zero voxels: {info['non_zero_voxels']:,}")
                print(f"  File size: {info['file_size_mb']:.2f} MB")
                
            except Exception as e:
                print(f"Error getting file info: {e}")
    else:
        print("Example directories not found")
    
    # Test file validation
    test_files = [
        "data/images/patient001.nii.gz",
        "data/images/nonexistent.nii.gz"
    ]
    existing, missing = validate_file_paths(test_files)
    print(f"\nFile validation:")
    print(f"  Existing files: {len(existing)}")
    print(f"  Missing files: {len(missing)}")


def main():
    """Main function to run all examples."""
    print("onem_radiomics Usage Examples")
    print("==============================")
    
    # Setup logging
    setup_logging()
    
    # Run examples
    try:
        example_basic_extraction()
        example_batch_extraction()
        example_custom_configuration()
        example_file_utilities()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        logging.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    main()