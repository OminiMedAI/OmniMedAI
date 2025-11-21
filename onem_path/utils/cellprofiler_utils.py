"""
CellProfiler utility functions for pathology feature extraction
"""

import os
import logging
import subprocess
import tempfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Try to import required libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def create_cellprofiler_pipeline(modules: List[str] = None,
                            parameters: Dict = None) -> str:
    """
    Create a CellProfiler pipeline configuration.
    
    Args:
        modules: List of feature modules to include
        parameters: Dictionary of parameters for modules
        
    Returns:
        Path to created pipeline file
    """
    if modules is None:
        modules = ['morphological', 'texture', 'intensity']
    
    if parameters is None:
        parameters = {}
    
    # Create temporary file for pipeline
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cppipe', delete=False)
    
    try:
        # Create basic pipeline structure
        pipeline = {
            'version': 6,
            'integers': [],
            'floats': [],
            'namespaces': [],
            'modules': []
        }
        
        # Add metadata
        pipeline['metadata'] = {
            'name': 'onem_path_feature_extraction',
            'description': 'Automatic feature extraction for pathology images',
            'version': '1.0'
        }
        
        # Image module
        image_module = _create_image_module(parameters.get('image', {}))
        pipeline['modules'].append(image_module)
        
        # Feature extraction modules based on requested modules
        module_creators = {
            'morphological': _create_morphology_module,
            'texture': _create_texture_module,
            'intensity': _create_intensity_module,
            'color': _create_color_module,
            'nuclei': _create_nuclei_module,
            'cells': _create_cells_module,
            'cytoplasm': _create_cytoplasm_module,
            'membrane': _create_membrane_module
        }
        
        for module_name in modules:
            if module_name in module_creators:
                module_config = parameters.get(module_name, {})
                module = module_creators[module_name](module_config)
                pipeline['modules'].append(module)
        
        # Convert to JSON and save
        with open(temp_file.name, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        return temp_file.name
        
    except Exception as e:
        logging.error(f"Failed to create CellProfiler pipeline: {e}")
        return None


def _create_image_module(params: Dict) -> Dict:
    """Create Image module for CellProfiler."""
    return {
        'module_number': 1,
        'module_name': 'Images',
        'settings': {
            'FilterImages': {
                'Filter': params.get('filter', 'All files'),
                'Directory': params.get('directory', '.'),
                'FileTypes': params.get('file_types', [('.tif', '.tiff', '.png', '.jpg', '.jpeg')]),
                'CheckImages': params.get('check_images', True),
                'NameSingle': params.get('name_single', 'Image'),
                'NameMulti': params.get('name_multi', 'Image'),
                'MaxImageSize': params.get('max_size', 2048),
                'LoadImagesAsColorImage': params.get('load_as_color', True),
                'MatchTextColor': params.get('match_text_color', False)
            }
        }
    }


def _create_morphology_module(params: Dict) -> Dict:
    """Create Morphology module."""
    return {
        'module_number': 2,
        'module_name': 'MeasureObjectIntensity',
        'settings': {
            'CalculateIntensity': params.get('calculate_intensity', True),
            'CalculateAdvanced': params.get('calculate_advanced', True),
            'RelateToObject': params.get('relate_to_object', 1),
            'ObjectName': params.get('object_name', 'Nuclei'),
            'MeasureArea': params.get('measure_area', True),
            'MeasurePerimeter': params.get('measure_perimeter', True),
            'MeasureFormFactor': params.get('measure_form_factor', True),
            'MeasureSolidity': params.get('measure_solidity', True),
            'MeasureExtent': params.get('measure_extent', True),
            'MeasureEuler': params.get('measure_euler', True),
            'MeasureCenterOfMass': params.get('measure_center_of_mass', True),
            'MeasureMaxIntensity': params.get('measure_max_intensity', True),
            'MeasureMeanIntensity': params.get('measure_mean_intensity', True),
            'MeasureMinIntensity': params.get('measure_min_intensity', True),
            'MeasureIntegratedIntensity': params.get('measure_integrated_intensity', True)
        }
    }


def _create_texture_module(params: Dict) -> Dict:
    """Create Texture module."""
    return {
        'module_number': 3,
        'module_name': 'MeasureTexture',
        'settings': {
            'ObjectName': params.get('object_name', 'Nuclei'),
            'RelateToObject': params.get('relate_to_object', 1),
            'CalculateHaralick': params.get('calculate_haralick', True),
            'CalculateGabor': params.get('calculate_gabor', True),
            'CalculateZernike': params.get('calculate_zernike', True),
            'Angles': params.get('angles', [0, 45, 90, 135]),
            'GaborFrequencies': params.get('gabor_frequencies', [0.1, 0.3, 0.5]),
            'ZernikeDegree': params.get('zernike_degree', 12)
        }
    }


def _create_intensity_module(params: Dict) -> Dict:
    """Create Intensity module."""
    return {
        'module_number': 4,
        'module_name': 'MeasureImageIntensity',
        'settings': {
            'CalculateTexture': params.get('calculate_texture', False),
            'CalculateColocalization': params.get('calculate_colocalization', False),
            'CalculateSaturation': params.get('calculate_saturation', True),
            'CalculateSummaries': params.get('calculate_summaries', True),
            'MeasureGranularity': params.get('measure_granularity', 10),
            'CalculateTextureMoments': params.get('calculate_texture_moments', True)
        }
    }


def _create_color_module(params: Dict) -> Dict:
    """Create Color module."""
    return {
        'module_number': 5,
        'module_name': 'MeasureObjectSizeShape',
        'settings': {
            'ObjectName': params.get('object_name', 'Nuclei'),
            'CalculateArea': params.get('calculate_area', True),
            'CalculatePerimeter': params.get('calculate_perimeter', True),
            'CalculateFormFactor': params.get('calculate_form_factor', True),
            'CalculateSolidity': params.get('measure_solidity', True),
            'CalculateExtent': params.get('measure_extent', True),
            'CalculateEuler': params.get('measure_euler', True),
            'CalculateCenterOfMass': params.get('measure_center_of_mass', True),
            'CalculateFeretDiameter': params.get('measure_feret_diameter', True),
            'CalculateMajorMinorAxisLength': params.get('measure_axes', True),
            'CalculateEccentricity': params.get('measure_eccentricity', True)
        }
    }


def _create_nuclei_module(params: Dict) -> Dict:
    """Create Nuclei-specific module."""
    return {
        'module_number': 6,
        'module_name': 'IdentifyPrimaryObjects',
        'settings': {
            'SelectInputImage': params.get('select_input', 1),
            'ObjectName': params.get('object_name', 'Nuclei'),
            'SelectInputType': params.get('input_type', 'Both'),
            'SelectMethod': params.get('method', 'Adaptive'),
            'GlobalThresholdingMethod': params.get('global_threshold', 'Minimum Cross Entropy'),
            'MinMax': params.get('min_max', [0, 1]),
            'TypicalObjectSize': params.get('typical_size', [64, 64]),
            'FillHoles': params.get('fill_holes', True),
            'RegularizationFactor': params.get('regularization_factor', 0.1),
            'LowerOutlierFraction': params.get('lower_outlier', 0.05),
            'UpperOutlierFraction': params.get('upper_outlier', 0.05),
            'LogTransform': params.get('log_transform', False)
        }
    }


def _create_cells_module(params: Dict) -> Dict:
    """Create Cells module."""
    return _create_nuclei_module({**params, 'object_name': 'Cells'})


def _create_cytoplasm_module(params: Dict) -> Dict:
    """Create Cytoplasm module."""
    return {
        'module_number': 7,
        'module_name': 'IdentifySecondaryObjects',
        'settings': {
            'SelectInputImage': params.get('select_input', 1),
            'ObjectName': params.get('object_name', 'Cytoplasm'),
            'SelectInputType': params.get('input_type', 'Objects'),
            'SelectObjects': params.get('select_objects', 'Nuclei'),
            'SelectDiameter': params.get('diameter', 30),
            'Method': params.get('method', 'Propagation'),
            'MaximumDistance': params.get('max_distance', 50),
            'RegularizationFactor': params.get('regularization_factor', 0.1)
        }
    }


def _create_membrane_module(params: Dict) -> Dict:
    """Create Membrane module."""
    return {
        'module_number': 8,
        'module_name': 'MeasureObjectNeighbors',
        'settings': {
            'ObjectName': params.get('object_name', 'Cells'),
            'SelectAdjacentType': params.get('adjacent_type', 'Touching'),
            'SelectExpansionSize': params.get('expansion_size', 1),
            'SelectDiameter': params.get('diameter', 30)
        }
    }


def run_cellprofiler_pipeline(pipeline_path: str,
                         input_dir: str,
                         output_dir: str,
                         image_pattern: str = "*.tif",
                         num_workers: int = 1) -> Dict:
    """
    Run CellProfiler pipeline on images.
    
    Args:
        pipeline_path: Path to CellProfiler pipeline file
        input_dir: Input directory with images
        output_dir: Output directory
        image_pattern: Image file pattern
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Check if CellProfiler is available
        cellprofiler_path = _find_cellprofiler_executable()
        
        if not cellprofiler_path:
            raise RuntimeError("CellProfiler executable not found")
        
        # Prepare command
        cmd = [
            cellprofiler_path,
            '-c', str(num_workers),
            '-r', pipeline_path,
            '-i', input_dir,
            '-o', output_dir,
            '-f', image_pattern
        ]
        
        logging.info(f"Running CellProfiler with command: {' '.join(cmd)}")
        
        # Execute CellProfiler
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        result = {
            'success': return_code == 0,
            'return_code': return_code,
            'stdout': stdout,
            'stderr': stderr,
            'pipeline_path': pipeline_path,
            'input_dir': input_dir,
            'output_dir': output_dir
        }
        
        if result['success']:
            logging.info("CellProfiler execution completed successfully")
        else:
            logging.error(f"CellProfiler execution failed with code {return_code}")
            logging.error(f"Error output: {stderr}")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to run CellProfiler pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def _find_cellprofiler_executable() -> Optional[str]:
    """Find CellProfiler executable."""
    # Try different possible paths
    possible_paths = [
        'cellprofiler',  # In PATH
        '/Applications/CellProfiler.app/Contents/MacOS/cellprofiler',  # macOS
        'C:\\Program Files\\CellProfiler\\cellprofiler.exe',  # Windows
        'C:\\Program Files (x86)\\CellProfiler\\cellprofiler.exe'  # Windows 32-bit
    ]
    
    # Check if executable exists
    import shutil
    
    for path in possible_paths:
        if shutil.which(path):
            return path
    
    return None


def parse_cellprofiler_output(output_dir: str) -> Dict:
    """
    Parse CellProfiler output files.
    
    Args:
        output_dir: Directory containing CellProfiler output files
        
    Returns:
        Dictionary with parsed results
    """
    try:
        output_path = Path(output_dir)
        
        # Look for CSV output files
        csv_files = list(output_path.glob("*.csv"))
        
        results = {
            'output_dir': str(output_path),
            'output_files': [str(f) for f in csv_files],
            'feature_data': {},
            'metadata': {}
        }
        
        # Parse main features file
        main_features_file = output_path / "MyExpt_Object.csv"
        if main_features_file.exists() and PANDAS_AVAILABLE:
            df = pd.read_csv(str(main_features_file))
            results['feature_data'] = df.to_dict('records')
            results['feature_columns'] = list(df.columns)
            results['num_objects'] = len(df)
        
        # Parse image metadata file
        metadata_file = output_path / "MyExpt_Image.csv"
        if metadata_file.exists() and PANDAS_AVAILABLE:
            df = pd.read_csv(str(metadata_file))
            results['metadata'] = df.to_dict('records')
            results['num_images'] = len(df)
        
        # Parse other CSV files if they exist
        for csv_file in csv_files:
            if csv_file.name != "MyExpt_Object.csv" and csv_file.name != "MyExpt_Image.csv":
                try:
                    df = pd.read_csv(str(csv_file))
                    results[csv_file.stem] = df.to_dict('records')
                except Exception as e:
                    logging.warning(f"Failed to parse {csv_file.name}: {e}")
        
        return results
        
    except Exception as e:
        logging.error(f"Failed to parse CellProfiler output: {e}")
        return {
            'error': str(e),
            'output_dir': output_dir
        }


def convert_cellprofiler_to_standard_format(cellprofiler_results: Dict) -> Dict:
    """
    Convert CellProfiler output to standard format.
    
    Args:
        cellprofiler_results: Raw CellProfiler results
        
    Returns:
        Standardized feature dictionary
    """
    try:
        if 'feature_data' not in cellprofiler_results:
            return {'error': 'No feature data found'}
        
        # Convert to standard feature names
        feature_mapping = {
            'Area': 'morphological_area',
            'Perimeter': 'morphological_perimeter',
            'FormFactor': 'morphological_form_factor',
            'Solidity': 'morphological_solidity',
            'Eccentricity': 'morphological_eccentricity',
            'MajorAxisLength': 'morphological_major_axis',
            'MinorAxisLength': 'morphological_minor_axis',
            'MeanIntensity': 'intensity_mean',
            'MaxIntensity': 'intensity_max',
            'MinIntensity': 'intensity_min',
            'IntegratedIntensity': 'intensity_integrated',
            'Texture_Entropy': 'texture_entropy',
            'Texture_Contrast': 'texture_contrast',
            'Texture_Correlation': 'texture_correlation',
            'Texture_Homogeneity': 'texture_homogeneity'
        }
        
        standard_features = {}
        
        for obj_data in cellprofiler_results['feature_data']:
            obj_features = {}
            
            for cp_name, standard_name in feature_mapping.items():
                if cp_name in obj_data:
                    obj_features[standard_name] = obj_data[cp_name]
            
            # Add object identifier if available
            if 'ImageNumber' in obj_data:
                obj_features['image_id'] = obj_data['ImageNumber']
            if 'ObjectNumber' in obj_data:
                obj_features['object_id'] = obj_data['ObjectNumber']
            
            # Store by object ID
            obj_id = obj_data.get('ObjectNumber', 0)
            standard_features[f'object_{obj_id}'] = obj_features
        
        return {
            'standard_features': standard_features,
            'num_objects': len(cellprofiler_results['feature_data']),
            'feature_types': list(set([name.split('_')[0] for name in feature_mapping.values()]))
        }
        
    except Exception as e:
        logging.error(f"Failed to convert CellProfiler format: {e}")
        return {'error': str(e)}


def create_cellprofiler_batch_script(image_dir: str,
                                pipeline_path: str,
                                output_dir: str,
                                num_workers: int = 1) -> str:
    """
    Create a batch script for running CellProfiler on multiple images.
    
    Args:
        image_dir: Directory with input images
        pipeline_path: Path to CellProfiler pipeline
        output_dir: Output directory
        num_workers: Number of parallel workers
        
    Returns:
        Path to created batch script
    """
    script_content = f"""#!/bin/bash
# CellProfiler batch processing script
# Generated by onem_path

CELLPROFILER="{_find_cellprofiler_executable()}"
INPUT_DIR="{image_dir}"
PIPELINE="{pipeline_path}"
OUTPUT_DIR="{output_dir}"
WORKERS={num_workers}

echo "Starting CellProfiler batch processing..."
echo "Input directory: $INPUT_DIR"
echo "Pipeline: $PIPELINE"
echo "Output directory: $OUTPUT_DIR"
echo "Workers: $WORKERS"

"$CELLPROFILER" \\
    -c "$WORKERS" \\
    -r "$PIPELINE" \\
    -i "$INPUT_DIR" \\
    -o "$OUTPUT_DIR" \\
    -f "*.tif *.tiff *.png *.jpg *.jpeg"

if [ $? -eq 0 ]; then
    echo "CellProfiler batch processing completed successfully"
else
    echo "CellProfiler batch processing failed"
    exit 1
fi
"""
    
    # Write to file
    script_path = os.path.join(output_dir, 'run_cellprofiler.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path