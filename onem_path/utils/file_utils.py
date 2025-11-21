"""
File utility functions for pathology feature extraction
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Try to import PIL
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")


def get_pathology_files(directory: str, 
                     recursive: bool = False,
                     extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get list of pathology image files in directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    # Default supported extensions
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.svs']
    
    # Find files
    if recursive:
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(f'*{ext}'))
            files.extend(dir_path.rglob(f'*{ext.upper()}'))
    else:
        files = []
        for ext in extensions:
            files.extend(dir_path.glob(f'*{ext}'))
            files.extend(dir_path.glob(f'*{ext.upper()}'))
    
    # Filter to files only and convert to strings
    image_files = [str(f) for f in files if f.is_file()]
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files


def validate_pathology_image(file_path: str, 
                         min_size: Tuple[int, int] = (64, 64),
                         max_size: Optional[Tuple[int, int]] = None,
                         supported_formats: Optional[List[str]] = None) -> Dict:
    """
    Validate a pathology image file.
    
    Args:
        file_path: Path to image file
        min_size: Minimum image dimensions
        max_size: Maximum image dimensions (None for no limit)
        supported_formats: List of supported file formats
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': False,
        'error': None,
        'warnings': [],
        'file_info': {}
    }
    
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.svs']
    
    try:
        # Check file existence
        if not os.path.exists(file_path):
            validation_result['error'] = f'File not found: {file_path}'
            return validation_result
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in supported_formats:
            validation_result['warnings'].append(f'File format {file_ext} may not be supported')
        
        # Try to load image
        if PIL_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    # Get image information
                    validation_result['file_info'] = {
                        'file_path': file_path,
                        'file_name': Path(file_path).name,
                        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                        'format': img.format,
                        'mode': img.mode,
                        'size': img.size,
                        'has_transparency': img.mode in ('RGBA', 'LA', 'P')
                    }
                    
                    # Check dimensions
                    width, height = img.size
                    
                    if width < min_size[0] or height < min_size[1]:
                        validation_result['warnings'].append(
                            f'Image size {img.size} is smaller than minimum {min_size}'
                        )
                    
                    if max_size and (width > max_size[0] or height > max_size[1]):
                        validation_result['warnings'].append(
                            f'Image size {img.size} exceeds maximum {max_size}'
                        )
                    
                    # Check mode
                    if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
                        validation_result['warnings'].append(
                            f'Image mode {img.mode} may need conversion'
                        )
                    
                    # Check if image is truncated (common issue)
                    if hasattr(img, 'is_animated') and img.is_animated:
                        validation_result['warnings'].append('Animated image detected, using first frame')
                    
                    validation_result['valid'] = True
                    
            except UnidentifiedImageError:
                validation_result['error'] = 'Cannot identify image format'
            except Exception as e:
                validation_result['error'] = f'Failed to load image: {str(e)}'
        else:
            validation_result['error'] = 'PIL not available for image validation'
            
    except Exception as e:
        validation_result['error'] = f'Validation failed: {str(e)}'
    
    return validation_result


def create_output_structure(base_dir: str, create_subdirs: bool = True) -> Dict:
    """
    Create output directory structure for pathology features.
    
    Args:
        base_dir: Base output directory
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Dictionary with paths to created directories
    """
    base_path = Path(base_dir)
    
    structure = {
        'base': str(base_path),
        'features': str(base_path / 'features'),
        'cellprofiler_features': str(base_path / 'cellprofiler_features'),
        'titan_features': str(base_path / 'titan_features'),
        'combined_features': str(base_path / 'combined_features'),
        'reports': str(base_path / 'reports'),
        'logs': str(base_path / 'logs'),
        'temp': str(base_path / 'temp'),
        'visualizations': str(base_path / 'visualizations')
    }
    
    if create_subdirs:
        for path in structure.values():
            os.makedirs(path, exist_ok=True)
    
    return structure


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception:
        return ''


def get_image_file_info(file_path: str) -> Dict:
    """
    Get detailed information about an image file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    info = {
        'file_path': file_path,
        'file_name': Path(file_path).name,
        'file_size_bytes': os.path.getsize(file_path),
        'exists': os.path.exists(file_path),
        'readable': os.access(file_path, os.R_OK)
    }
    
    if not info['exists']:
        return info
    
    if PIL_AVAILABLE:
        try:
            with Image.open(file_path) as img:
                info.update({
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.size[0],
                    'height': img.size[1],
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P'),
                    'is_animated': getattr(img, 'is_animated', False)
                })
                
                # Calculate additional properties
                width, height = img.size
                info.update({
                    'aspect_ratio': width / height,
                    'total_pixels': width * height,
                    'megapixels': (width * height) / 1_000_000
                })
                
        except Exception as e:
            info['load_error'] = str(e)
    
    return info


def create_file_manifest(directory: str, 
                   recursive: bool = False,
                   include_hashes: bool = False) -> Dict:
    """
    Create a manifest of image files in directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        include_hashes: Whether to include file hashes
        
    Returns:
        Dictionary with file manifest
    """
    files = get_pathology_files(directory, recursive)
    
    manifest = {
        'directory': directory,
        'total_files': len(files),
        'total_size_bytes': 0,
        'file_formats': {},
        'image_sizes': {'min': None, 'max': None, 'avg': None},
        'files': []
    }
    
    for file_path in files:
        try:
            file_info = get_image_file_info(file_path)
            manifest['files'].append(file_info)
            
            # Update totals
            manifest['total_size_bytes'] += file_info['file_size_bytes']
            
            # Format statistics
            if 'format' in file_info:
                fmt = file_info['format']
                if fmt not in manifest['file_formats']:
                    manifest['file_formats'][fmt] = 0
                manifest['file_formats'][fmt] += 1
            
            # Size statistics
            if 'total_pixels' in file_info:
                pixels = file_info['total_pixels']
                if manifest['image_sizes']['min'] is None:
                    manifest['image_sizes']['min'] = pixels
                    manifest['image_sizes']['max'] = pixels
                    manifest['image_sizes']['avg'] = pixels
                else:
                    manifest['image_sizes']['min'] = min(manifest['image_sizes']['min'], pixels)
                    manifest['image_sizes']['max'] = max(manifest['image_sizes']['max'], pixels)
            
            # Add hash if requested
            if include_hashes:
                file_info['md5_hash'] = calculate_file_hash(file_path, 'md5')
                
        except Exception as e:
            manifest['files'].append({
                'file_path': file_path,
                'error': str(e)
            })
    
    # Calculate average
    if manifest['image_sizes']['min'] is not None:
        total_pixels = sum(f.get('total_pixels', 0) for f in manifest['files'] if 'error' not in f)
        count = len([f for f in manifest['files'] if 'error' not in f])
        if count > 0:
            manifest['image_sizes']['avg'] = total_pixels / count
    
    return manifest


def save_batch_report(results: List[Dict], output_dir: str, 
                   report_name: str = 'batch_report.json') -> str:
    """
    Save a comprehensive batch processing report.
    
    Args:
        results: List of processing results
        output_dir: Output directory
        report_name: Name of report file
        
    Returns:
        Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, report_name)
    
    # Analyze results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    report = {
        'summary': {
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
            'processing_time': sum(r.get('processing_time', 0) for r in results)
        },
        'failed_files': [
            {'file_path': r.get('file_path', 'unknown'), 'error': r['error']} 
            for r in failed
        ],
        'successful_files': [
            {
                'file_path': r['file_path'],
                'features_extracted': r.get('features_extracted', 0),
                'processing_time': r.get('processing_time', 0)
            } 
            for r in successful
        ],
        'file_formats': {},
        'processing_methods': {}
    }
    
    # Analyze file formats and methods
    for result in successful:
        # File format
        file_path = result['file_path']
        ext = Path(file_path).suffix.lower()
        if ext not in report['file_formats']:
            report['file_formats'][ext] = 0
        report['file_formats'][ext] += 1
        
        # Processing method
        method = result.get('processing_method', 'unknown')
        if method not in report['processing_methods']:
            report['processing_methods'][method] = 0
        report['processing_methods'][method] += 1
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report_path


def clean_temp_files(temp_dir: str, keep_patterns: Optional[List[str]] = None):
    """
    Clean temporary files in directory.
    
    Args:
        temp_dir: Temporary directory to clean
        keep_patterns: List of filename patterns to keep
    """
    if not os.path.exists(temp_dir):
        return
    
    keep_patterns = keep_patterns or []
    
    try:
        temp_path = Path(temp_dir)
        for file_path in temp_path.glob('*'):
            if file_path.is_file():
                # Check if file should be kept
                keep_file = any(file_path.match(pattern) for pattern in keep_patterns)
                
                if not keep_file:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logging.warning(f"Failed to delete temp file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Failed to clean temp directory {temp_dir}: {e}")


def create_progress_callback(total_steps: int, description: str = "Processing"):
    """
    Create a progress callback function.
    
    Args:
        total_steps: Total number of steps
        description: Description of the process
        
    Returns:
        Progress callback function
    """
    def progress_callback(step: int, message: str = ""):
        progress = (step / total_steps) * 100
        print(f"\r{description}: {progress:.1f}% - {message}", end='', flush=True)
        
        if step == total_steps:
            print()  # New line when complete
    
    return progress_callback


def format_processing_time(seconds: float) -> str:
    """
    Format processing time in human readable format.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"