"""
Image preprocessing utilities for segmentation
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union
from scipy import ndimage
from skimage import transform, filters, exposure

# Try to import required libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from skimage.measure import label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def preprocess_image(image_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Apply preprocessing steps to image data.
    
    Args:
        image_data: Input image data
        config: Preprocessing configuration
        
    Returns:
        Preprocessed image data
    """
    processed = image_data.copy()
    
    # Handle invalid values
    processed = handle_invalid_values(processed)
    
    # Normalization
    if config.get('normalization', 'none') != 'none':
        processed = normalize_image(processed, config['normalization'])
    
    # Resizing
    if config.get('resize_input', False):
        target_size = config.get('target_size', (256, 256))
        processed = resize_image(processed, target_size, 
                               config.get('interpolation_method', 'bilinear'))
    
    # Padding
    if config.get('padding', False):
        processed = pad_image(processed, config.get('target_size', processed.shape))
    
    # Intensity clipping
    if config.get('clip_intensity', False):
        percentiles = config.get('clip_percentiles', (1, 99))
        processed = clip_intensity(processed, percentiles)
    
    return processed


def handle_invalid_values(image_data: np.ndarray) -> np.ndarray:
    """
    Handle NaN and infinite values in image data.
    
    Args:
        image_data: Input image data
        
    Returns:
        Image data with invalid values handled
    """
    # Replace NaN with 0 or interpolation
    nan_mask = np.isnan(image_data)
    if np.any(nan_mask):
        # Simple approach: replace NaN with 0
        image_data[nan_mask] = 0
        logging.warning("NaN values found and replaced with 0")
    
    # Replace inf with max/min values
    inf_mask = np.isinf(image_data)
    if np.any(inf_mask):
        image_data[np.isposinf(image_data)] = np.finfo(image_data.dtype).max
        image_data[np.isneginf(image_data)] = np.finfo(image_data.dtype).min
        logging.warning("Infinite values found and replaced with finite values")
    
    return image_data


def normalize_image(image_data: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Normalize image data using specified method.
    
    Args:
        image_data: Input image data
        method: Normalization method ('z_score', 'min_max', 'robust', 'none')
        
    Returns:
        Normalized image data
    """
    if method == 'none':
        return image_data
    
    if method == 'z_score':
        return z_score_normalization(image_data)
    elif method == 'min_max':
        return min_max_normalization(image_data)
    elif method == 'robust':
        return robust_normalization(image_data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def z_score_normalization(image_data: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    mean = np.mean(image_data)
    std = np.std(image_data)
    
    if std == 0:
        return np.zeros_like(image_data)
    
    return (image_data - mean) / std


def min_max_normalization(image_data: np.ndarray, 
                        target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Apply min-max normalization to specified range."""
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    if max_val == min_val:
        return np.ones_like(image_data) * target_range[0]
    
    # Normalize to [0, 1]
    normalized = (image_data - min_val) / (max_val - min_val)
    
    # Scale to target range
    range_min, range_max = target_range
    return normalized * (range_max - range_min) + range_min


def robust_normalization(image_data: np.ndarray) -> np.ndarray:
    """Apply robust normalization using median and MAD."""
    median = np.median(image_data)
    mad = np.median(np.abs(image_data - median))  # Median Absolute Deviation
    
    if mad == 0:
        return np.zeros_like(image_data)
    
    return (image_data - median) / (1.4826 * mad)  # 1.4826 * MAD ≈ std for normal distribution


def resize_image(image_data: np.ndarray, target_size: Union[Tuple, int],
               method: str = 'bilinear') -> np.ndarray:
    """
    Resize image data to target size.
    
    Args:
        image_data: Input image data
        target_size: Target size (tuple for 2D/3D or single int for square)
        method: Interpolation method
        
    Returns:
        Resized image data
    """
    if isinstance(target_size, int):
        if len(image_data.shape) == 2:
            target_size = (target_size, target_size)
        else:
            # For 3D, apply to all spatial dimensions
            target_size = (target_size, target_size, image_data.shape[2])
    
    # Choose interpolation order
    order = get_interpolation_order(method)
    
    try:
        if len(image_data.shape) == 2:
            # 2D image
            resized = transform.resize(image_data, target_size, order=order, 
                                   preserve_range=True, anti_aliasing=True)
        elif len(image_data.shape) == 3:
            # 3D image
            if len(target_size) == 2:
                # Resize first two dimensions, keep third
                resized_2d = transform.resize(
                    image_data, target_size + (image_data.shape[2],),
                    order=order, preserve_range=True, anti_aliasing=True
                )
                resized = resized_2d
            else:
                # Resize all dimensions
                resized = transform.resize(image_data, target_size, order=order,
                                       preserve_range=True, anti_aliasing=True)
        else:
            raise ValueError(f"Unsupported image dimensions: {len(image_data.shape)}")
        
        return resized.astype(image_data.dtype)
        
    except Exception as e:
        logging.error(f"Failed to resize image: {e}")
        return image_data


def get_interpolation_order(method: str) -> int:
    """Get interpolation order for given method."""
    mapping = {
        'nearest': 0,
        'bilinear': 1,
        'bilinear_precise': 2,
        'cubic': 3,
        'quartic': 4,
        'quintic': 5
    }
    return mapping.get(method, 1)


def pad_image(image_data: np.ndarray, target_size: Tuple, 
             mode: str = 'constant', constant_values: float = 0) -> np.ndarray:
    """
    Pad image data to target size.
    
    Args:
        image_data: Input image data
        target_size: Target size
        mode: Padding mode
        constant_values: Values for constant padding
        
    Returns:
        Padded image data
    """
    current_size = image_data.shape[:len(target_size)]
    
    # Calculate padding needed
    padding = []
    for i, (curr, target) in enumerate(zip(current_size, target_size)):
        if curr < target:
            pad_total = target - curr
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding.extend([pad_before, pad_after])
        else:
            padding.extend([0, 0])
    
    # Add padding for additional dimensions (channels, etc.)
    while len(padding) < len(image_data.shape) * 2:
        padding.extend([0, 0])
    
    # Apply padding
    if padding:
        return np.pad(image_data, 
                     [tuple(padding[i:i+2]) for i in range(0, len(padding), 2)],
                     mode=mode, constant_values=constant_values)
    else:
        return image_data


def clip_intensity(image_data: np.ndarray, 
                 percentiles: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Clip image intensity values to specified percentiles.
    
    Args:
        image_data: Input image data
        percentiles: Lower and upper percentiles for clipping
        
    Returns:
        Clipped image data
    """
    p_low, p_high = percentiles
    
    low_val = np.percentile(image_data, p_low)
    high_val = np.percentile(image_data, p_high)
    
    return np.clip(image_data, low_val, high_val)


def enhance_contrast(image_data: np.ndarray, method: str = 'histogram_equalization') -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image_data: Input image data
        method: Contrast enhancement method
        
    Returns:
        Contrast-enhanced image data
    """
    if method == 'histogram_equalization':
        return histogram_equalization(image_data)
    elif method == 'adaptive_equalization':
        return adaptive_histogram_equalization(image_data)
    elif method == 'clahe':
        return clahe_equalization(image_data)
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


def histogram_equalization(image_data: np.ndarray) -> np.ndarray:
    """Apply histogram equalization."""
    try:
        if len(image_data.shape) == 2:
            # 2D image
            equalized = exposure.equalize_hist(image_data)
        else:
            # 3D image - apply slice by slice
            equalized = np.zeros_like(image_data)
            for i in range(image_data.shape[2]):
                equalized[:, :, i] = exposure.equalize_hist(image_data[:, :, i])
        
        return equalized
        
    except Exception as e:
        logging.error(f"Histogram equalization failed: {e}")
        return image_data


def adaptive_histogram_equalization(image_data: np.ndarray, 
                                kernel_size: int = 128) -> np.ndarray:
    """Apply adaptive histogram equalization."""
    try:
        if len(image_data.shape) == 2:
            equalized = exposure.equalize_adapthist(image_data, kernel_size=kernel_size)
        else:
            equalized = np.zeros_like(image_data)
            for i in range(image_data.shape[2]):
                equalized[:, :, i] = exposure.equalize_adapthist(
                    image_data[:, :, i], kernel_size=kernel_size
                )
        
        return equalized
        
    except Exception as e:
        logging.error(f"Adaptive histogram equalization failed: {e}")
        return image_data


def clahe_equalization(image_data: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    try:
        if len(image_data.shape) == 2:
            equalized = exposure.equalize_adapthist(image_data, 
                                                clip_limit=clip_limit)
        else:
            equalized = np.zeros_like(image_data)
            for i in range(image_data.shape[2]):
                equalized[:, :, i] = exposure.equalize_adapthist(
                    image_data[:, :, i], clip_limit=clip_limit
                )
        
        return equalized
        
    except Exception as e:
        logging.error(f"CLAHE failed: {e}")
        return image_data


def resample_image_spacing(image_data: np.ndarray, original_spacing: Tuple[float, ...],
                         target_spacing: Tuple[float, ...]) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Resample image to different spacing.
    
    Args:
        image_data: Input image data
        original_spacing: Original voxel spacing
        target_spacing: Target voxel spacing
        
    Returns:
        Tuple of (resampled_data, new_spacing)
    """
    if len(original_spacing) != len(image_data.shape):
        raise ValueError("Spacing dimension mismatch")
    
    # Calculate new shape
    new_shape = []
    for i, (orig_spacing, target_spacing) in enumerate(zip(original_spacing, target_spacing)):
        orig_size = image_data.shape[i]
        new_size = int(orig_size * orig_spacing / target_spacing)
        new_shape.append(new_size)
    
    new_shape = tuple(new_shape)
    
    try:
        # Use scipy for resampling
        zoom_factors = [new / orig for new, orig in zip(new_shape, image_data.shape)]
        resampled = ndimage.zoom(image_data, zoom_factors, order=3, mode='nearest')
        
        return resampled, target_spacing
        
    except Exception as e:
        logging.error(f"Resampling failed: {e}")
        return image_data, original_spacing


def crop_to_foreground(image_data: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """
    Crop image to foreground region.
    
    Args:
        image_data: Input image data
        threshold: Intensity threshold for foreground detection
        
    Returns:
        Tuple of (cropped_data, crop_slices)
    """
    if threshold is None:
        # Use Otsu threshold if available
        try:
            threshold = filters.threshold_otsu(image_data)
        except:
            # Fallback to mean
            threshold = np.mean(image_data)
    
    # Create binary mask
    mask = image_data > threshold
    
    # Find bounding box
    if not SKIMAGE_AVAILABLE:
        # Simple approach using numpy
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return image_data, tuple([slice(0, s) for s in image_data.shape])
        
        bbox_slices = []
        for dim_coords in coords:
            bbox_slices.append(slice(np.min(dim_coords), np.max(dim_coords) + 1))
        
        return image_data[tuple(bbox_slices)], tuple(bbox_slices)
    
    # Use skimage for more robust cropping
    labeled = label(mask)
    if labeled.max() == 0:
        return image_data, tuple([slice(0, s) for s in image_data.shape])
    
    # Find bounding box of all labels
    props = []
    try:
        from skimage.measure import regionprops
        props = regionprops(labeled)
        
        # Find overall bounding box
        min_coords = []
        max_coords = []
        for prop in props:
            bbox = prop.bbox
            if not min_coords:
                min_coords = list(bbox[:len(image_data.shape)])
                max_coords = list(bbox[len(image_data.shape):])
            else:
                for i in range(len(image_data.shape)):
                    min_coords[i] = min(min_coords[i], bbox[i])
                    max_coords[i] = max(max_coords[i], bbox[i + len(image_data.shape)])
        
        bbox_slices = tuple([slice(min_coords[i], max_coords[i]) for i in range(len(image_data.shape))])
        
        return image_data[bbox_slices], bbox_slices
        
    except Exception:
        # Fallback to simple cropping
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return image_data, tuple([slice(0, s) for s in image_data.shape])
        
        bbox_slices = []
        for dim_coords in coords:
            bbox_slices.append(slice(np.min(dim_coords), np.max(dim_coords) + 1))
        
        return image_data[tuple(bbox_slices)], tuple(bbox_slices)