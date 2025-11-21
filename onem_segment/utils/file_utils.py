"""
File utility functions for segmentation
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np

# Try to import nibabel
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Install with: pip install nibabel")


def load_nifti_image(file_path: str) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Load a NIfTI image and return data, header, and affine.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Tuple of (image_data, header, affine)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    
    try:
        # Load image
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        
        return data, header, affine
        
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")


def save_nifti_image(data: np.ndarray, affine: np.ndarray, 
                    reference_header: Dict = None, output_path: str = None):
    """
    Save numpy array as NIfTI file.
    
    Args:
        data: Image data to save
        affine: Affine transformation matrix
        reference_header: Reference header for metadata
        output_path: Path to save the file
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    if output_path is None:
        raise ValueError("output_path must be provided")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Create NIfTI image
        if reference_header is not None:
            img = nib.Nifti1Image(data, affine, header=reference_header)
        else:
            img = nib.Nifti1Image(data, affine)
        
        # Save image
        nib.save(img, output_path)
        
    except Exception as e:
        raise ValueError(f"Failed to save NIfTI file {output_path}: {e}")


def create_output_structure(base_dir: str, create_subdirs: bool = True) -> Dict:
    """
    Create output directory structure for segmentation results.
    
    Args:
        base_dir: Base output directory
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Dictionary with paths to created directories
    """
    base_path = Path(base_dir)
    
    structure = {
        'base': str(base_path),
        'segmentations': str(base_path / 'segmentations'),
        'probabilities': str(base_path / 'probabilities'),
        'preprocessed': str(base_path / 'preprocessed'),
        'reports': str(base_path / 'reports'),
        'logs': str(base_path / 'logs'),
        'temp': str(base_path / 'temp')
    }
    
    if create_subdirs:
        for path in structure.values():
            os.makedirs(path, exist_ok=True)
    
    return structure


def validate_nifti_file(file_path: str) -> Dict:
    """
    Validate NIfTI file and return information.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Dictionary with validation results
    """
    if not NIBABEL_AVAILABLE:
        return {'error': 'nibabel not available'}
    
    validation_result = {
        'valid': False,
        'error': None,
        'file_info': {}
    }
    
    try:
        if not os.path.exists(file_path):
            validation_result['error'] = f'File not found: {file_path}'
            return validation_result
        
        # Try to load file
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Collect file information
        validation_result['file_info'] = {
            'shape': data.shape,
            'data_type': str(data.dtype),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'has_nan': bool(np.any(np.isnan(data))),
            'has_inf': bool(np.any(np.isinf(data))),
            'is_empty': data.size == 0,
            'affine_shape': img.affine.shape,
            'voxel_size': img.header.get_zooms(),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
        
        # Validate data integrity
        validation_result['valid'] = True
        
        # Check for potential issues
        warnings = []
        
        if validation_result['file_info']['has_nan']:
            warnings.append('Data contains NaN values')
        
        if validation_result['file_info']['has_inf']:
            warnings.append('Data contains infinite values')
        
        if validation_result['file_info']['is_empty']:
            warnings.append('Image is empty')
        
        if validation_result['file_info']['data_type'] == 'float64':
            warnings.append('Double precision data, consider using float32')
        
        if validation_result['file_info']['file_size_mb'] > 500:
            warnings.append('Large file size (>500MB), may require more memory')
        
        validation_result['warnings'] = warnings
        
    except Exception as e:
        validation_result['error'] = str(e)
    
    return validation_result


def get_nifti_files(directory: str, recursive: bool = False, 
                   pattern: str = "*.nii.gz") -> List[str]:
    """
    Get list of NIfTI files in directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        pattern: File pattern to match
        
    Returns:
        List of NIfTI file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))
    
    # Filter and convert to strings
    nifti_files = [str(f) for f in files if f.is_file()]
    
    # Also check for .nii files
    nii_pattern = pattern.replace('.nii.gz', '.nii')
    if recursive:
        nii_files = list(dir_path.rglob(nii_pattern))
    else:
        nii_files = list(dir_path.glob(nii_pattern))
    
    nii_files = [str(f) for f in nii_files if f.is_file()]
    
    # Combine and remove duplicates
    all_files = list(set(nifti_files + nii_files))
    
    # Sort for consistency
    all_files.sort()
    
    return all_files


def backup_file(file_path: str, backup_suffix: str = '.bak') -> Optional[str]:
    """
    Create a backup of existing file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file
        
    Returns:
        Path to backup file if created, None otherwise
    """
    if not os.path.exists(file_path):
        return None
    
    backup_path = file_path + backup_suffix
    counter = 1
    
    # If backup already exists, add number
    while os.path.exists(backup_path):
        backup_path = f"{file_path}{backup_suffix}.{counter}"
        counter += 1
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def ensure_directory_exists(file_path: str):
    """
    Ensure directory exists for given file path.
    
    Args:
        file_path: File path whose directory should exist
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def get_output_filename(input_path: str, suffix: str = '_seg', 
                      output_format: str = 'nii.gz') -> str:
    """
    Generate output filename based on input path.
    
    Args:
        input_path: Input file path
        suffix: Suffix to add to filename
        output_format: Output file format
        
    Returns:
        Output filename
    """
    input_file = Path(input_path)
    
    # Remove existing extensions
    stem = input_file.stem
    if stem.endswith('.nii'):
        stem = stem[:-4]
    
    # Add suffix and extension
    output_name = f"{stem}{suffix}.{output_format}"
    
    return output_name


def copy_metadata(source_file: str, target_file: str) -> bool:
    """
    Copy metadata from source to target NIfTI file.
    
    Args:
        source_file: Source NIfTI file
        target_file: Target NIfTI file
        
    Returns:
        True if successful, False otherwise
    """
    if not NIBABEL_AVAILABLE:
        return False
    
    try:
        # Load source and target
        source_img = nib.load(source_file)
        target_img = nib.load(target_file)
        
        # Update target with source metadata
        new_target_img = nib.Nifti1Image(
            target_img.get_fdata(),
            source_img.affine,
            header=source_img.header
        )
        
        # Save updated target
        nib.save(new_target_img, target_file)
        
        return True
        
    except Exception:
        return False


def create_file_manifest(directory: str, recursive: bool = False) -> Dict:
    """
    Create a manifest of files in directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        
    Returns:
        Dictionary with file manifest
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return {'error': 'Directory does not exist'}
    
    manifest = {
        'directory': str(dir_path),
        'files': [],
        'total_files': 0,
        'total_size_mb': 0,
        'file_types': {}
    }
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'extension': file_path.suffix,
                'modified_time': file_path.stat().st_mtime
            }
            
            manifest['files'].append(file_info)
            manifest['total_size_mb'] += file_info['size_mb']
            
            # Count file types
            ext = file_info['extension']
            if ext not in manifest['file_types']:
                manifest['file_types'][ext] = 0
            manifest['file_types'][ext] += 1
    
    manifest['total_files'] = len(manifest['files'])
    
    return manifest