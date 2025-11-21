"""
Image utility functions for pathology feature extraction
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

# Try to import PIL and skimage
try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from skimage import io, color, filters, exposure, transform, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def preprocess_pathology_image(image: Union[str, np.ndarray], 
                          config: dict = None) -> np.ndarray:
    """
    Preprocess a pathology image for feature extraction.
    
    Args:
        image: Input image as file path or numpy array
        config: Preprocessing configuration
        
    Returns:
        Preprocessed image as numpy array
    """
    if config is None:
        config = {}
    
    # Load image if path provided
    if isinstance(image, str):
        if PIL_AVAILABLE:
            img_array = load_pil_image(image)
        elif SKIMAGE_AVAILABLE:
            img_array = load_skimage_image(image)
        elif CV2_AVAILABLE:
            img_array = load_cv2_image(image)
        else:
            raise ImportError("No image loading library available")
    else:
        img_array = image.copy()
    
    # Apply preprocessing steps
    processed_image = img_array.copy()
    
    # 1. Color space conversion
    color_conversion = config.get('color_conversion', None)
    if color_conversion:
        processed_image = convert_color_space(processed_image, color_conversion)
    
    # 2. Staining normalization
    if config.get('normalize_staining', False):
        processed_image = normalize_staining(processed_image, config.get('staining_method', 'reinhard'))
    
    # 3. Contrast enhancement
    if config.get('enhance_contrast', False):
        processed_image = enhance_image_contrast(processed_image, config.get('contrast_method', 'clahe'))
    
    # 4. Noise reduction
    if config.get('reduce_noise', False):
        processed_image = reduce_image_noise(processed_image, config.get('noise_method', 'gaussian'))
    
    # 5. Resize
    target_size = config.get('target_size', None)
    if target_size:
        processed_image = resize_pathology_image(processed_image, target_size, 
                                        config.get('resize_method', 'bilinear'))
    
    # 6. Intensity normalization
    if config.get('normalize_intensity', True):
        processed_image = normalize_image_intensity(processed_image, config.get('intensity_method', 'percentile'))
    
    return processed_image


def load_pil_image(image_path: str) -> np.ndarray:
    """Load image using PIL."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        logging.error(f"Failed to load image with PIL {image_path}: {e}")
        raise


def load_skimage_image(image_path: str) -> np.ndarray:
    """Load image using scikit-image."""
    try:
        return io.imread(image_path)
    except Exception as e:
        logging.error(f"Failed to load image with skimage {image_path}: {e}")
        raise


def load_cv2_image(image_path: str) -> np.ndarray:
    """Load image using OpenCV."""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            # Convert BGR to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Failed to load image with OpenCV")
    except Exception as e:
        logging.error(f"Failed to load image with OpenCV {image_path}: {e}")
        raise


def convert_color_space(image: np.ndarray, target_space: str) -> np.ndarray:
    """
    Convert image to different color space.
    
    Args:
        image: Input image array
        target_space: Target color space ('rgb', 'hsv', 'lab', 'gray')
        
    Returns:
        Color-converted image
    """
    if not SKIMAGE_AVAILABLE:
        return image
    
    if target_space.lower() == 'gray':
        if len(image.shape) == 3:
            return color.rgb2gray(image)
        else:
            return image
    
    elif target_space.lower() == 'hsv':
        if len(image.shape) == 3:
            return color.rgb2hsv(image)
        else:
            return image
    
    elif target_space.lower() == 'lab':
        if len(image.shape) == 3:
            return color.rgb2lab(image)
        else:
            return image
    
    elif target_space.lower() == 'rgb':
        return image
    
    else:
        logging.warning(f"Unknown color space: {target_space}")
        return image


def normalize_staining(image: np.ndarray, method: str = 'reinhard') -> np.ndarray:
    """
    Normalize staining in pathology images.
    
    Args:
        image: Input image
        method: Normalization method ('reinhard', 'macenko', 'vahadane')
        
    Returns:
        Stain-normalized image
    """
    if not SKIMAGE_AVAILABLE:
        return image
    
    if len(image.shape) != 3:
        logging.warning("Staining normalization requires color image")
        return image
    
    try:
        if method.lower() == 'reinhard':
            return _reinhard_normalization(image)
        elif method.lower() == 'macenko':
            return _macenko_normalization(image)
        elif method.lower() == 'vahadane':
            return _vahadane_normalization(image)
        else:
            logging.warning(f"Unknown staining normalization method: {method}")
            return image
    except Exception as e:
        logging.error(f"Staining normalization failed: {e}")
        return image


def _reinhard_normalization(image: np.ndarray) -> np.ndarray:
    """Reinhard staining normalization (simplified version)."""
    try:
        # Convert to optical density
        od = -np.log((image.astype(np.float32) / 255 + 1e-6))
        
        # Extract Hematoxylin and Eosin channels
        # Simplified approach - use color deconvolution
        if SKIMAGE_AVAILABLE:
            from skimage import restoration
            
            # Standard H&E stain matrix (approximate)
            he_matrix = np.array([[0.65, 0.70, 0.29],
                                    [0.07, 0.99, 0.11]])
            
            # Perform stain deconvolution
            try:
                stains = restoration.unmixing_stains(od, he_matrix)
                h_channel = stains[:, :, 0]  # Hematoxylin
                e_channel = stains[:, :, 1]  # Eosin
            except:
                # Fallback to simple channel extraction
                h_channel = od[:, :, 0]  # R channel (Hematoxylin)
                e_channel = od[:, :, 1]  # G channel (Eosin)
        else:
            h_channel = od[:, :, 0]
            e_channel = od[:, :, 1]
        
        # Normalize channels
        h_norm = (h_channel - np.percentile(h_channel, 99)) / (np.percentile(h_channel, 99) - np.percentile(h_channel, 1))
        e_norm = (e_channel - np.percentile(e_channel, 99)) / (np.percentile(e_channel, 99) - np.percentile(e_channel, 1))
        
        # Clip and convert back
        h_norm = np.clip(h_norm, 0, 1)
        e_norm = np.clip(e_norm, 0, 1)
        
        # Recombine (simplified)
        normalized_image = np.stack([h_norm * 255, e_norm * 255, np.zeros_like(h_norm)], axis=2)
        
        return np.clip(normalized_image, 0, 255).astype(np.uint8)
        
    except Exception as e:
        logging.warning(f"Reinhard normalization failed: {e}")
        return image


def _macenko_normalization(image: np.ndarray) -> np.ndarray:
    """Macenko staining normalization (simplified version)."""
    try:
        # This is a simplified version - full Macenko is more complex
        # Convert to optical density
        od = -np.log((image.astype(np.float32) / 255 + 1e-6))
        
        # Simple channel-wise normalization
        for i in range(3):
            channel = od[:, :, i]
            channel_norm = (channel - np.mean(channel)) / (np.std(channel) + 1e-6)
            od[:, :, i] = np.clip(channel_norm, -3, 3)
        
        # Convert back to RGB
        rgb = np.exp(-od)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return rgb
        
    except Exception as e:
        logging.warning(f"Macenko normalization failed: {e}")
        return image


def _vahadane_normalization(image: np.ndarray) -> np.ndarray:
    """Vahadane staining normalization (simplified version)."""
    try:
        # This is a simplified version
        # Convert to optical density
        od = -np.log((image.astype(np.float32) / 255 + 1e-6))
        
        # Simple percentile-based normalization per channel
        for i in range(3):
            channel = od[:, :, i]
            p_low, p_high = np.percentile(channel, [1, 99])
            channel_norm = (channel - p_low) / (p_high - p_low + 1e-6)
            od[:, :, i] = np.clip(channel_norm, 0, 1)
        
        # Convert back to RGB
        rgb = np.exp(-od)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return rgb
        
    except Exception as e:
        logging.warning(f"Vahadane normalization failed: {e}")
        return image


def enhance_image_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance image contrast using various methods.
    
    Args:
        image: Input image
        method: Enhancement method ('clahe', 'histogram_eq', 'gamma')
        
    Returns:
        Contrast-enhanced image
    """
    try:
        if method.lower() == 'clahe':
            if CV2_AVAILABLE:
                return _clahe_opencv(image)
            elif SKIMAGE_AVAILABLE:
                return _clahe_skimage(image)
            else:
                return image
        
        elif method.lower() == 'histogram_eq':
            if CV2_AVAILABLE:
                return _histogram_eq_opencv(image)
            elif SKIMAGE_AVAILABLE:
                return _histogram_eq_skimage(image)
            else:
                return image
        
        elif method.lower() == 'gamma':
            return _gamma_correction(image, gamma=1.2)
        
        else:
            logging.warning(f"Unknown contrast enhancement method: {method}")
            return image
            
    except Exception as e:
        logging.error(f"Contrast enhancement failed: {e}")
        return image


def _clahe_opencv(image: np.ndarray) -> np.ndarray:
    """CLAHE using OpenCV."""
    if len(image.shape) == 3:
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def _clahe_skimage(image: np.ndarray) -> np.ndarray:
    """CLAHE using scikit-image."""
    if not SKIMAGE_AVAILABLE:
        return image
    
    if len(image.shape) == 3:
        # Apply to each channel
        enhanced_channels = []
        for i in range(3):
            enhanced = exposure.equalize_adapthist(image[:, :, i])
            enhanced_channels.append(enhanced)
        return np.stack(enhanced_channels, axis=2)
    else:
        return exposure.equalize_adapthist(image)


def _histogram_eq_opencv(image: np.ndarray) -> np.ndarray:
    """Histogram equalization using OpenCV."""
    if len(image.shape) == 3:
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply histogram equalization to Y channel
        y_eq = cv2.equalizeHist(y)
        
        # Merge channels and convert back
        enhanced_yuv = cv2.merge([y_eq, u, v])
        enhanced_rgb = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2RGB)
        return enhanced_rgb
    else:
        return cv2.equalizeHist(image)


def _histogram_eq_skimage(image: np.ndarray) -> np.ndarray:
    """Histogram equalization using scikit-image."""
    if not SKIMAGE_AVAILABLE:
        return image
    
    if len(image.shape) == 3:
        # Apply to each channel
        enhanced_channels = []
        for i in range(3):
            enhanced = exposure.equalize_hist(image[:, :, i])
            enhanced_channels.append(enhanced)
        return np.stack(enhanced_channels, axis=2)
    else:
        return exposure.equalize_hist(image)


def _gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to image."""
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
    
    if CV2_AVAILABLE:
        return cv2.LUT(image, table)
    else:
        # Apply gamma correction manually
        enhanced = np.power(image / 255.0, inv_gamma) * 255.0
        return np.clip(enhanced, 0, 255).astype(np.uint8)


def reduce_image_noise(image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """
    Reduce noise in image using various methods.
    
    Args:
        image: Input image
        method: Noise reduction method ('gaussian', 'bilateral', 'median')
        
    Returns:
        Denoised image
    """
    try:
        if method.lower() == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0) if CV2_AVAILABLE else image
        
        elif method.lower() == 'bilateral':
            if CV2_AVAILABLE:
                return cv2.bilateralFilter(image, 9, 75, 75)
            else:
                return image
        
        elif method.lower() == 'median':
            if CV2_AVAILABLE:
                return cv2.medianBlur(image, 5)
            elif SKIMAGE_AVAILABLE:
                from skimage import filters
                return filters.median(image)
            else:
                return image
        
        else:
            logging.warning(f"Unknown noise reduction method: {method}")
            return image
            
    except Exception as e:
        logging.error(f"Noise reduction failed: {e}")
        return image


def resize_pathology_image(image: np.ndarray, target_size: Tuple[int, int], 
                         method: str = 'bilinear') -> np.ndarray:
    """
    Resize pathology image maintaining quality.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        method: Interpolation method ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        Resized image
    """
    try:
        if CV2_AVAILABLE:
            return _resize_opencv(image, target_size, method)
        elif SKIMAGE_AVAILABLE:
            return _resize_skimage(image, target_size, method)
        elif PIL_AVAILABLE:
            return _resize_pil(image, target_size, method)
        else:
            return image
            
    except Exception as e:
        logging.error(f"Image resize failed: {e}")
        return image


def _resize_opencv(image: np.ndarray, target_size: Tuple[int, int], method: str) -> np.ndarray:
    """Resize using OpenCV."""
    if method.lower() == 'nearest':
        interp = cv2.INTER_NEAREST
    elif method.lower() == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif method.lower() == 'bicubic':
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    
    return cv2.resize(image, target_size, interpolation=interp)


def _resize_skimage(image: np.ndarray, target_size: Tuple[int, int], method: str) -> np.ndarray:
    """Resize using scikit-image."""
    if not SKIMAGE_AVAILABLE:
        return image
    
    if method.lower() == 'nearest':
        order = 0
    elif method.lower() == 'bilinear':
        order = 1
    elif method.lower() == 'bicubic':
        order = 3
    else:
        order = 1
    
    return transform.resize(image, target_size, order=order, preserve_range=True, anti_aliasing=False)


def _resize_pil(image: np.ndarray, target_size: Tuple[int, int], method: str) -> np.ndarray:
    """Resize using PIL."""
    if not PIL_AVAILABLE:
        return image
    
    # Convert to PIL image
    if len(image.shape) == 3:
        pil_img = Image.fromarray(image.astype(np.uint8), 'RGB')
    else:
        pil_img = Image.fromarray(image.astype(np.uint8), 'L')
    
    # Choose resampling method
    if method.lower() == 'nearest':
        resample = Image.NEAREST
    elif method.lower() == 'bilinear':
        resample = Image.BILINEAR
    elif method.lower() == 'bicubic':
        resample = Image.BICUBIC
    else:
        resample = Image.BILINEAR
    
    # Resize and convert back
    resized_pil = pil_img.resize(target_size, resample)
    return np.array(resized_pil)


def normalize_image_intensity(image: np.ndarray, method: str = 'percentile') -> np.ndarray:
    """
    Normalize image intensity.
    
    Args:
        image: Input image
        method: Normalization method ('percentile', 'zscore', 'minmax')
        
    Returns:
        Intensity-normalized image
    """
    try:
        if method.lower() == 'percentile':
            return _percentile_normalization(image)
        elif method.lower() == 'zscore':
            return _zscore_normalization(image)
        elif method.lower() == 'minmax':
            return _minmax_normalization(image)
        else:
            logging.warning(f"Unknown intensity normalization method: {method}")
            return image
            
    except Exception as e:
        logging.error(f"Intensity normalization failed: {e}")
        return image


def _percentile_normalization(image: np.ndarray, 
                           p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based intensity normalization."""
    if len(image.shape) == 3:
        # Apply to each channel
        normalized_channels = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            low_val, high_val = np.percentile(channel, [p_low, p_high])
            normalized_channel = np.clip((channel - low_val) / (high_val - low_val + 1e-8), 0, 1)
            normalized_channels.append(normalized_channel)
        return np.stack(normalized_channels, axis=2)
    else:
        # Single channel
        img_float = image.astype(np.float32)
        low_val, high_val = np.percentile(img_float, [p_low, p_high])
        return np.clip((img_float - low_val) / (high_val - low_val + 1e-8), 0, 1)


def _zscore_normalization(image: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    if len(image.shape) == 3:
        # Apply to each channel
        normalized_channels = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            if std_val > 0:
                normalized_channel = (channel - mean_val) / std_val
            else:
                normalized_channel = channel
            normalized_channels.append(normalized_channel)
        return np.stack(normalized_channels, axis=2)
    else:
        # Single channel
        img_float = image.astype(np.float32)
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)
        if std_val > 0:
            return (img_float - mean_val) / std_val
        else:
            return img_float


def _minmax_normalization(image: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    if len(image.shape) == 3:
        # Apply to each channel
        normalized_channels = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                normalized_channel = (channel - min_val) / (max_val - min_val)
            else:
                normalized_channel = channel
            normalized_channels.append(normalized_channel)
        return np.stack(normalized_channels, axis=2)
    else:
        # Single channel
        img_float = image.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            return (img_float - min_val) / (max_val - min_val)
        else:
            return img_float