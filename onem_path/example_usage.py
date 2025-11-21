"""
Example usage of onem_path module

This script demonstrates how to use onem_path module for pathology image
feature extraction using CellProfiler and TITAN deep transfer learning.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from onem_path import CellProfilerExtractor, TITANExtractor, PathologyConfig, PRESET_CONFIGS
from onem_path.utils.file_utils import get_pathology_files, create_output_structure
from onem_path.utils.image_utils import preprocess_pathology_image
from onem_path.utils.titan_utils import get_available_backbones


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pathology_extraction.log')
        ]
    )


def example_cellprofiler_extraction():
    """Example 1: CellProfiler-based pathology radiomics feature extraction."""
    print("\n" + "="*60)
    print("Example 1: CellProfiler Pathology Radiomics Extraction")
    print("="*60)
    
    # Create CellProfiler extractor with default configuration
    extractor = CellProfilerExtractor()
    
    # Example image path (replace with your actual path)
    image_path = "data/pathology/patient001.tif"
    
    # Check if example file exists
    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        print("Please update path or use your own data.")
        return
    
    try:
        # Extract features with specific modules
        modules = ['morphological', 'texture', 'intensity', 'color', 'nuclei']
        
        result = extractor.extract_features(
            image_path=image_path,
            output_path="output/patient001_cellprofiler_features.json",
            modules=modules
        )
        
        print(f"Successfully extracted features from {Path(image_path).name}")
        print(f"Module results: {result['module_results']}")
        print(f"Total features: {len(result['features'])}")
        
        # Show sample features
        print("\nSample features:")
        for i, (name, value) in enumerate(list(result['features'].items())[:10]):
            print(f"  {name}: {value:.4f}")
        
        # Show image metadata
        metadata = result['image_metadata']
        print(f"\nImage metadata:")
        print(f"  Shape: {metadata['image_shape']}")
        print(f"  File size: {metadata['file_size_bytes']} bytes")
        print(f"  Aspect ratio: {metadata['aspect_ratio']:.3f}")
        
    except Exception as e:
        print(f"Error during CellProfiler extraction: {e}")


def example_cellprofiler_batch():
    """Example 2: Batch CellProfiler extraction."""
    print("\n" + "="*60)
    print("Example 2: Batch CellProfiler Feature Extraction")
    print("="*60)
    
    # Create extractor
    extractor = CellProfilerExtractor()
    
    # Example directories (replace with your actual directories)
    images_dir = "data/pathology"
    output_dir = "output/cellprofiler_features"
    
    # Check if directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        print("Please update path or use your own data.")
        return
    
    try:
        # Batch extract features
        results = extractor.extract_batch_features(
            image_dir=images_dir,
            output_dir=output_dir,
            file_pattern="*.tif *.tiff *.png *.jpg",
            modules=['morphological', 'texture', 'intensity', 'nuclei'],
            n_jobs=4
        )
        
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        print(f"Batch processing completed:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        if successful > 0:
            # Analyze feature statistics
            feature_columns = [col for col in results.columns if col.startswith('morphological_') or 
                           col.startswith('texture_') or col.startswith('intensity_') or
                           col.startswith('nuclei_')]
            
            print(f"\nFeature analysis:")
            print(f"  Total feature types: {len(feature_columns)}")
            print(f"  Sample feature columns: {feature_columns[:5]}")
            
    except Exception as e:
        print(f"Error during batch CellProfiler extraction: {e}")


def example_titan_extraction():
    """Example 3: TITAN deep transfer learning feature extraction."""
    print("\n" + "="*60)
    print("Example 3: TITAN Deep Transfer Learning Feature Extraction")
    print("="*60)
    
    # Create TITAN extractor with specific configuration
    config = {
        'titan_backbone': 'resnet50',
        'titan_feature_dim': 1024,
        'titan_use_attention': True,
        'resize_images': True,
        'image_size': (224, 224),
        'normalize_images': True
    }
    
    extractor = TITANExtractor(config)
    
    # Example image path
    image_path = "data/pathology/patient001.tif"
    
    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        return
    
    try:
        # Extract deep features
        result = extractor.extract_features(
            image_path=image_path,
            output_path="output/patient001_titan_features.json"
        )
        
        print(f"Successfully extracted TITAN features from {Path(image_path).name}")
        print(f"Features extracted: {len(result['features'])}")
        print(f"Feature dimension: {result['extraction_metadata']['feature_dim']}")
        print(f"Extraction time: {result['extraction_metadata']['extraction_time']:.3f}s")
        
        # Show model information
        model_info = result['model_info']
        print(f"\nModel information:")
        print(f"  Backbone: {model_info['backbone_name']}")
        print(f"  Feature dimension: {model_info['feature_dim']}")
        print(f"  Use attention: {model_info['use_attention']}")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        
        # Show sample features
        print("\nSample TITAN features:")
        for i, value in enumerate(result['features'][:10]):
            print(f"  titan_feature_{i}: {value:.6f}")
        
    except Exception as e:
        print(f"Error during TITAN extraction: {e}")


def example_titan_batch():
    """Example 4: Batch TITAN feature extraction."""
    print("\n" + "="*60)
    print("Example 4: Batch TITAN Feature Extraction")
    print("="*60)
    
    # Create TITAN extractor with GPU acceleration
    config = {
        'titan_backbone': 'efficientnet_b0',  # More efficient backbone
        'titan_feature_dim': 512,
        'titan_use_attention': False,  # Faster inference
        'device': 'cuda',  # Use GPU if available
        'batch_size_titan': 32,
        'num_workers_titan': 4
    }
    
    extractor = TITANExtractor(config)
    
    # Example directories
    images_dir = "data/pathology"
    output_dir = "output/titan_features"
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    try:
        # Batch extract features
        results = extractor.extract_batch_features(
            image_dir=images_dir,
            output_dir=output_dir,
            batch_size=32,
            num_workers=4
        )
        
        successful = len(results[results['extraction_success'] == True])
        
        print(f"Batch TITAN extraction completed:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {successful}")
        
        if successful > 0:
            # Analyze feature statistics
            feature_cols = [col for col in results.columns if col.startswith('titan_feature_')]
            
            print(f"\nTITAN feature analysis:")
            print(f"  Feature dimension: {len(feature_cols)}")
            print(f"  Sample feature range: titan_feature_0")
            if 'titan_feature_0' in results.columns:
                col_data = results['titan_feature_0']
                print(f"    Min: {col_data.min():.6f}")
                print(f"    Max: {col_data.max():.6f}")
                print(f"    Mean: {col_data.mean():.6f}")
        
    except Exception as e:
        print(f"Error during batch TITAN extraction: {e}")


def example_combined_extraction():
    """Example 5: Combined CellProfiler + TITAN extraction."""
    print("\n" + "="*60)
    print("Example 5: Combined Feature Extraction")
    print("="*60)
    
    # Create output structure
    output_structure = create_output_structure("output/combined_features")
    
    # Example directories
    images_dir = "data/pathology"
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    try:
        # Get image files
        image_files = get_pathology_files(images_dir)[:3]  # Process first 3 for demo
        
        print(f"Processing {len(image_files)} images with combined extraction...")
        
        # Extract CellProfiler features
        cp_extractor = CellProfilerExtractor()
        cp_results = []
        
        for image_path in image_files:
            try:
                cp_result = cp_extractor.extract_features(
                    image_path=image_path,
                    modules=['morphological', 'texture', 'nuclei']
                )
                cp_results.append(cp_result)
            except Exception as e:
                print(f"CellProfiler failed for {image_path}: {e}")
        
        # Extract TITAN features
        titan_config = {'titan_backbone': 'resnet50', 'device': 'cuda'}
        titan_extractor = TITANExtractor(titan_config)
        titan_results = []
        
        for image_path in image_files:
            try:
                titan_result = titan_extractor.extract_features(image_path=image_path)
                titan_results.append(titan_result)
            except Exception as e:
                print(f"TITAN failed for {image_path}: {e}")
        
        # Combine results
        combined_results = []
        for i, image_path in enumerate(image_files):
            combined = {
                'image_path': image_path,
                'image_name': Path(image_path).name
            }
            
            # Add CellProfiler features
            if i < len(cp_results):
                cp_result = cp_results[i]
                combined.update({
                    'cp_success': True,
                    'cp_features': cp_result['features'],
                    'cp_total_features': len(cp_result['features'])
                })
            else:
                combined.update({'cp_success': False, 'cp_features': {}, 'cp_total_features': 0})
            
            # Add TITAN features
            if i < len(titan_results):
                titan_result = titan_results[i]
                combined.update({
                    'titan_success': True,
                    'titan_features': titan_result['features'],
                    'titan_feature_dim': titan_result['extraction_metadata']['feature_dim']
                })
            else:
                combined.update({'titan_success': False, 'titan_features': {}, 'titan_feature_dim': 0})
            
            combined_results.append(combined)
        
        # Save combined results
        import pandas as pd
        import json
        
        # Save detailed JSON
        with open(output_structure['features'] + '/combined_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for result in combined_results:
            row = {
                'image_name': result['image_name'],
                'cp_success': result.get('cp_success', False),
                'titan_success': result.get('titan_success', False),
                'cp_feature_count': result.get('cp_total_features', 0),
                'titan_feature_count': result.get('titan_feature_dim', 0),
                'total_features': result.get('cp_total_features', 0) + result.get('titan_feature_dim', 0)
            }
            
            # Add sample features
            if result.get('cp_features'):
                cp_sample = list(result['cp_features'].items())[0] if result['cp_features'] else ('none', 0)
                row[f'cp_sample_{cp_sample[0]}'] = cp_sample[1]
            
            if result.get('titan_features'):
                row['titan_sample_0'] = result['titan_features'][0]
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = output_structure['features'] + '/combined_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"Combined extraction completed:")
        print(f"  Images processed: {len(combined_results)}")
        print(f"  CellProfiler successful: {sum(1 for r in combined_results if r.get('cp_success', False))}")
        print(f"  TITAN successful: {sum(1 for r in combined_results if r.get('titan_success', False))}")
        print(f"  Results saved to: {output_structure['features']}")
        
    except Exception as e:
        print(f"Error during combined extraction: {e}")


def example_model_benchmark():
    """Example 6: TITAN model benchmarking."""
    print("\n" + "="*60)
    print("Example 6: TITAN Model Benchmarking")
    print("="*60)
    
    try:
        # Show available backbones
        backbones = get_available_backbones()
        print(f"Available TITAN backbones: {len(backbones)}")
        print("Sample backbones:", backbones[:5])
        
        # Benchmark different backbones
        test_backbones = ['resnet18', 'resnet50', 'efficientnet_b0']
        
        from onem_path.utils.titan_utils import load_titan_model, benchmark_titan_extraction
        
        benchmark_results = []
        
        for backbone in test_backbones:
            print(f"\nBenchmarking {backbone}...")
            
            # Load model
            model = load_titan_model(
                backbone_name=backbone,
                device='auto'
            )
            
            # Benchmark
            result = benchmark_titan_extraction(
                model=model,
                num_iterations=50
            )
            
            result['backbone'] = backbone
            benchmark_results.append(result)
        
        # Display comparison
        print("\nBenchmark Results Summary:")
        print("Backbone      | FPS   | Time (ms) | Memory (MB) | Params")
        print("-" * 60)
        
        for result in benchmark_results:
            fps = result['fps']
            time_ms = result['avg_time_per_inference'] * 1000
            memory_mb = result['memory_used_gb'] * 1024
            params = result['model_info']['total_parameters']
            
            print(f"{result['backbone']:13s} | {fps:5.1f} | {time_ms:8.2f} | {memory_mb:8.1f} | {params:,}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")


def example_preset_configurations():
    """Example 7: Using preset configurations."""
    print("\n" + "="*60)
    print("Example 7: Preset Configurations")
    print("="*60)
    
    # Show available presets
    available_presets = list(PRESET_CONFIGS.keys())
    print(f"Available presets: {available_presets}")
    
    # Demonstrate different presets
    preset_examples = [
        ('default', 'Default balanced approach'),
        ('cellprofiler_only', 'Traditional pathology features only'),
        ('titan_only', 'Deep learning features only'),
        ('high_quality', 'High quality extraction'),
        ('fast_processing', 'Fast processing for large datasets'),
        ('research_grade', 'Comprehensive research-grade extraction')
    ]
    
    for preset_name, description in preset_examples:
        print(f"\n{preset_name.upper()} - {description}")
        
        try:
            config = PRESET_CONFIGS[preset_name]
            
            # Create extractors based on preset
            if config.extract_cellprofiler_features:
                cp_extractor = CellProfilerExtractor()
                print(f"  CellProfiler modules: {config.cellprofiler_modules}")
            
            if config.extract_titan_features:
                titan_extractor = TITANExtractor(config)
                print(f"  TITAN backbone: {config.titan_backbone}")
                print(f"  TITAN feature dim: {config.titan_feature_dim}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main function to run all examples."""
    print("onem_path Usage Examples")
    print("=========================")
    
    # Setup logging
    setup_logging()
    
    # Run examples
    try:
        example_cellprofiler_extraction()
        example_cellprofiler_batch()
        example_titan_extraction()
        example_titan_batch()
        example_combined_extraction()
        example_model_benchmark()
        example_preset_configurations()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        logging.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    main()