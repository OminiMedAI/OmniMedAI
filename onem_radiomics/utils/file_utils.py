"""
File utility functions for radiomics extraction
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Try to import nibabel
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def load_image_mask_pair(image_path: str, mask_path: str) -> Tuple[object, object]:
    """
    Load and validate an image-mask pair using nibabel.
    
    Args:
        image_path: Path to the NIfTI image file
        mask_path: Path to the NIfTI mask file
        
    Returns:
        Tuple of (image_nifti, mask_nifti) objects
        
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If file formats are incompatible
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    # Check file existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    try:
        # Load image and mask
        image = nib.load(image_path)
        mask = nib.load(mask_path)
        
        # Validate compatibility
        if image.shape != mask.shape:
            logging.warning(f"Image and mask shapes differ: image={image.shape}, mask={mask.shape}")
        
        if image.get_fdata().shape != mask.get_fdata().shape:
            logging.warning("Image and mask data shapes differ after loading")
        
        return image, mask
        
    except Exception as e:
        raise ValueError(f"Failed to load image-mask pair: {e}")


def get_matching_files(images_dir: str, masks_dir: str, 
                       file_pattern: str = "*.nii.gz") -> List[Tuple[str, str]]:
    """
    Find matching image-mask pairs in directories.
    
    Args:
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files
        file_pattern: File pattern to match (e.g., "*.nii.gz")
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Get all image files
    image_files = list(images_path.glob(file_pattern))
    
    matching_pairs = []
    
    for image_file in image_files:
        # Try different mask naming conventions
        stem = image_file.stem
        
        # Convention 1: image.nii.gz -> image_mask.nii.gz
        mask_name = f"{stem}_mask.nii.gz"
        mask_path = masks_path / mask_name
        
        if mask_path.exists():
            matching_pairs.append((str(image_file), str(mask_path)))
            continue
        
        # Convention 2: image.nii.gz -> image_seg.nii.gz
        mask_name = f"{stem}_seg.nii.gz"
        mask_path = masks_path / mask_name
        
        if mask_path.exists():
            matching_pairs.append((str(image_file), str(mask_path)))
            continue
        
        # Convention 3: image.nii.gz -> image.nii.gz (same name)
        mask_path = masks_path / image_file.name
        
        if mask_path.exists():
            matching_pairs.append((str(image_file), str(mask_path)))
            continue
        
        # Convention 4: Remove suffix like _CT, _MRI, etc.
        for suffix in ['_CT', '_MRI', '_PET', '_T1', '_T2', '_FLAIR', '_DWI']:
            if stem.endswith(suffix):
                base_stem = stem[:-len(suffix)]
                mask_name = f"{base_stem}_mask.nii.gz"
                mask_path = masks_path / mask_name
                
                if mask_path.exists():
                    matching_pairs.append((str(image_file), str(mask_path)))
                    break
        
        # Convention 5: Try with common mask suffixes
        if len(matching_pairs) == 0 or matching_pairs[-1][0] != str(image_file):
            for suffix in ['_mask', '_seg', '_roi', '_label']:
                mask_name = f"{stem}{suffix}.nii.gz"
                mask_path = masks_path / mask_name
                
                if mask_path.exists():
                    matching_pairs.append((str(image_file), str(mask_path)))
                    break
    
    return matching_pairs


def validate_file_paths(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of file paths, separating existing from missing files.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        Tuple of (existing_files, missing_files)
    """
    existing_files = []
    missing_files = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files


def ensure_directory_exists(file_path: str):
    """
    Ensure that the directory for the given file path exists.
    
    Args:
        file_path: File path whose directory should exist
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def get_file_info(file_path: str) -> dict:
    """
    Get basic information about a NIfTI file.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Dictionary with file information
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        nifti = nib.load(file_path)
        data = nifti.get_fdata()
        
        info = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'shape': nifti.shape,
            'dtype': str(data.dtype),
            'voxel_size_mm': nifti.header.get_zooms(),
            'affine_matrix': nifti.affine.tolist(),
            'data_min': float(data.min()),
            'data_max': float(data.max()),
            'data_mean': float(data.mean()),
            'data_std': float(data.std()),
            'non_zero_voxels': int(np.count_nonzero(data)),
            'total_voxels': int(data.size)
        }
        
        return info
        
    except Exception as e:
        raise ValueError(f"Failed to get file info for {file_path}: {e}")


def create_output_structure(output_dir: str, create_subdirs: bool = True) -> dict:
    """
    Create output directory structure for radiomics results.
    
    Args:
        output_dir: Main output directory
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Dictionary with paths to created directories
    """
    base_path = Path(output_dir)
    
    structure = {
        'base': str(base_path),
        'csv_files': str(base_path / 'csv'),
        'logs': str(base_path / 'logs'),
        'temp': str(base_path / 'temp'),
        'reports': str(base_path / 'reports')
    }
    
    if create_subdirs:
        for path in structure.values():
            os.makedirs(path, exist_ok=True)
    
    return structure


def backup_existing_file(file_path: str, backup_suffix: str = '.bak') -> Optional[str]:
    """
    Create a backup of an existing file.
    
    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix to add to backup file
        
    Returns:
        Path to backup file if original existed, None otherwise
    """
    if not os.path.exists(file_path):
        return None
    
    backup_path = file_path + backup_suffix
    counter = 1
    
    # If backup already exists, add number
    while os.path.exists(backup_path):
        backup_path = f"{file_path}{backup_suffix}.{counter}"
        counter += 1
    
    # Copy file to backup location
    import shutil
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def clean_temp_files(temp_dir: str, keep_patterns: Optional[List[str]] = None):
    """
    Clean temporary files in a directory.
    
    Args:
        temp_dir: Directory to clean
        keep_patterns: List of filename patterns to keep (e.g., ['*.log'])
    """
    if not os.path.exists(temp_dir):
        return
    
    temp_path = Path(temp_dir)
    keep_patterns = keep_patterns or []
    
    for file_path in temp_path.glob('*'):
        if file_path.is_file():
            # Check if file should be kept
            keep_file = False
            for pattern in keep_patterns:
                if file_path.match(pattern):
                    keep_file = True
                    break
            
            if not keep_file:
                try:
                    file_path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete temp file {file_path}: {e}")


# Import numpy for file info function
try:
    import numpy as np
except ImportError:
    np = None