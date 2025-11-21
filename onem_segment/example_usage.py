"""
Example usage of onem_segment module

This script demonstrates how to use the onem_segment module for automatic
ROI segmentation of medical images with 2D/3D model selection.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from onem_segment import ROISegmenter, SegmentationConfig, PRESET_CONFIGS
from onem_segment.utils.image_analyzer import ImageDimensionAnalyzer
from onem_segment.utils.file_utils import validate_nifti_file, get_nifti_files


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('segmentation.log')
        ]
    )


def example_basic_segmentation():
    """Example 1: Basic ROI segmentation."""
    print("\n" + "="*60)
    print("Example 1: Basic ROI Segmentation")
    print("="*60)
    
    # Create segmenter with default configuration
    segmenter = ROISegmenter()
    
    # Example image path (replace with your actual path)
    image_path = "data/images/patient001.nii.gz"
    
    # Check if example file exists
    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        print("Please update the path or use your own data.")
        return
    
    try:
        # Perform segmentation
        result = segmenter.segment_image(
            image_path=image_path,
            output_path=None,  # Will be generated automatically
            model_type='auto',  # Automatic 2D/3D selection
            quality='default',
            return_probabilities=False
        )
        
        print(f"Successfully segmented {Path(image_path).name}")
        print(f"Input path: {result['input_path']}")
        print(f"Output path: {result['output_path']}")
        print(f"Model used: {result['model_used']}")
        print(f"Processing type: {result['model_type']}")
        
        # Get segmentation statistics
        stats = segmenter.get_segmentation_statistics(result)
        print(f"ROI volume: {stats['roi_volume_voxels']} voxels")
        print(f"ROI percentage: {stats['roi_percentage']:.2f}%")
        print(f"ROI slices: {stats['roi_slices']}")
        
        # Display image analysis results
        analysis = result['analysis']
        print(f"Image shape: {analysis['shape']}")
        print(f"Slice spacing: {analysis['slice_spacing']:.2f} mm")
        print(f"Recommended processing: {'3D' if analysis['is_3d'] else '2D'}")
        
    except Exception as e:
        print(f"Error during segmentation: {e}")


def example_batch_segmentation():
    """Example 2: Batch segmentation with custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Batch Segmentation with Custom Configuration")
    print("="*60)
    
    # Use high-quality preset configuration
    config = PRESET_CONFIGS['high_quality']
    print(f"Using high-quality preset configuration")
    print(f"Model type: {config.model_type}")
    print(f"Quality level: {config.quality_level}")
    print(f"Normalization: {config.normalization}")
    print(f"Return probabilities: {config.return_probabilities}")
    
    # Create segmenter with custom config
    segmenter = ROISegmenter(config)
    
    # Example directories (replace with your actual directories)
    images_dir = "data/images"
    output_dir = "output/segmentations"
    
    # Check if directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        print("Please update the path or use your own data.")
        return
    
    try:
        # Batch segment
        results = segmenter.segment_batch(
            image_dir=images_dir,
            output_dir=output_dir,
            file_pattern="*.nii.gz",
            model_type='auto',
            quality='high_quality',
            n_jobs=1  # Sequential processing
        )
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"Batch segmentation completed:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        # Show model usage statistics
        model_usage = {}
        for result in successful:
            model = result.get('model_used', 'unknown')
            model_usage[model] = model_usage.get(model, 0) + 1
        
        print(f"Model usage:")
        for model, count in model_usage.items():
            print(f"  {model}: {count} images")
        
        # Show sample results
        if successful:
            sample_result = successful[0]
            stats = segmenter.get_segmentation_statistics(sample_result)
            print(f"\nSample result for {Path(sample_result['input_path']).name}:")
            print(f"  ROI volume: {stats['roi_volume_voxels']} voxels")
            print(f"  Model used: {sample_result['model_used']}")
            print(f"  Processing type: {sample_result['model_type']}")
        
    except Exception as e:
        print(f"Error during batch segmentation: {e}")


def example_image_analysis():
    """Example 3: Image dimension analysis."""
    print("\n" + "="*60)
    print("Example 3: Image Dimension Analysis")
    print("="*60)
    
    # Create image analyzer with custom parameters
    analyzer = ImageDimensionAnalyzer(
        min_3d_slices=25,
        max_slice_spacing=4.0,
        min_content_variation=0.15
    )
    
    # Example image paths
    image_paths = [
        "data/images/patient001.nii.gz",
        "data/images/patient002.nii.gz"
    ]
    
    # Filter existing files
    existing_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("No example images found.")
        print("Please update the paths or use your own data.")
        return
    
    try:
        # Analyze single image
        print("Single image analysis:")
        image_path = existing_paths[0]
        analysis = analyzer.analyze_image(image_path)
        
        print(f"Image: {Path(image_path).name}")
        print(f"  Shape: {analysis['shape']}")
        print(f"  Slices: {analysis['n_slices']}")
        print(f"  Slice spacing: {analysis['slice_spacing']:.2f} mm")
        print(f"  Recommended: {'3D' if analysis['is_3d'] else '2D'}")
        print(f"  Confidence: {analysis['recommendations']['confidence']}")
        
        print("\nContent analysis:")
        content = analysis['content_analysis']
        print(f"  Mean intensity: {content.get('mean', 'N/A'):.2f}")
        print(f"  Std deviation: {content.get('std', 'N/A'):.2f}")
        print(f"  Skewness: {content.get('skewness', 'N/A'):.2f}")
        print(f"  Dynamic range: {content.get('range', 'N/A'):.2f}")
        
        print("\nQuality metrics:")
        quality = analysis['quality_metrics']
        print(f"  SNR: {quality.get('snr', 'N/A'):.2f}")
        print(f"  CNR: {quality.get('cnr', 'N/A'):.2f}")
        print(f"  Sparsity: {quality.get('sparsity', 'N/A'):.3f}")
        
        # Batch analysis
        if len(existing_paths) > 1:
            print("\nBatch analysis:")
            batch_result = analyzer.batch_analyze(existing_paths)
            
            summary = batch_result['summary']
            print(f"  Total images: {summary['total_images']}")
            print(f"  Recommend 3D: {summary['recommend_3d']}")
            print(f"  Recommend 2D: {summary['recommend_2d']}")
            
            batch_recs = batch_result['batch_recommendations']
            print(f"  Primary approach: {batch_recs.get('primary_approach', 'unknown')}")
            print(f"  Confidence: {batch_recs.get('confidence', 0):.2f}")
            print(f"  Mixed processing: {batch_recs.get('mixed_processing', False)}")
        
    except Exception as e:
        print(f"Error during image analysis: {e}")


def example_custom_configuration():
    """Example 4: Custom configuration and model management."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    from onem_segment.config.settings import create_custom_config
    
    # Create custom configuration
    custom_config = create_custom_config(
        model_type='auto',
        quality_level='high_quality',
        input_size=(512, 512),  # For 2D models
        device='auto',
        normalization='z_score',
        min_roi_volume=20,
        morphological_operations=True,
        return_probabilities=True,
        generate_statistics=True
    )
    
    print("Custom configuration:")
    print(f"  Model type: {custom_config.model_type}")
    print(f"  Quality level: {custom_config.quality_level}")
    print(f"  Input size (2D): {custom_config.input_size_2d}")
    print(f"  Input size (3D): {custom_config.input_size_3d}")
    print(f"  Normalization: {custom_config.normalization}")
    print(f"  Min ROI volume: {custom_config.min_roi_volume}")
    print(f"  Return probabilities: {custom_config.return_probabilities}")
    
    # Save configuration
    config_path = "output/custom_segmentation_config.json"
    custom_config.save(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Load configuration
    loaded_config = custom_config.__class__.load(config_path)
    print(f"Configuration loaded successfully")
    print(f"Loaded config matches original: {custom_config.to_dict() == loaded_config.to_dict()}")
    
    # Demonstrate available presets
    from onem_segment.config.settings import get_available_presets
    available_presets = get_available_presets()
    print(f"\nAvailable presets: {available_presets}")
    
    print("\nPreset configurations:")
    for preset_name in available_presets[:5]:  # Show first 5
        preset = PRESET_CONFIGS[preset_name]
        print(f"  {preset_name}:")
        print(f"    Model type: {preset.model_type}")
        print(f"    Quality: {preset.quality_level}")
        print(f"    Min 3D slices: {preset.min_3d_slices}")


def example_file_utilities():
    """Example 5: File utility functions."""
    print("\n" + "="*60)
    print("Example 5: File Utilities")
    print("="*60)
    
    from onem_segment.utils.file_utils import (
        validate_nifti_file, get_nifti_files, create_file_manifest
    )
    
    # Example directory
    images_dir = "data/images"
    
    if os.path.exists(images_dir):
        # Get NIfTI files
        nifti_files = get_nifti_files(images_dir, recursive=False)
        print(f"Found {len(nifti_files)} NIfTI files in {images_dir}")
        
        # Validate first few files
        print("\nFile validation:")
        for file_path in nifti_files[:3]:  # Show first 3
            validation = validate_nifti_file(file_path)
            
            if validation['valid']:
                info = validation['file_info']
                print(f"  {Path(file_path).name}:")
                print(f"    Shape: {info['shape']}")
                print(f"    Data type: {info['data_type']}")
                print(f"    Size: {info['file_size_mb']:.2f} MB")
                print(f"    Intensity range: [{info['min_value']:.2f}, {info['max_value']:.2f}]")
                
                if validation.get('warnings'):
                    print(f"    Warnings: {len(validation['warnings'])}")
            else:
                print(f"  {Path(file_path).name}: {validation['error']}")
        
        # Create file manifest
        manifest = create_file_manifest(images_dir)
        print(f"\nDirectory manifest:")
        print(f"  Total files: {manifest['total_files']}")
        print(f"  Total size: {manifest['total_size_mb']:.2f} MB")
        print(f"  File types: {manifest['file_types']}")
        
    else:
        print(f"Example directory not found: {images_dir}")


def main():
    """Main function to run all examples."""
    print("onem_segment Usage Examples")
    print("===========================")
    
    # Setup logging
    setup_logging()
    
    # Run examples
    try:
        example_basic_segmentation()
        example_batch_segmentation()
        example_image_analysis()
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