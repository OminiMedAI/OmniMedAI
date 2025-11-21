"""
医学图像工具函数
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
import logging

try:
    import nibabel as nib
    import pydicom
    import SimpleITK as sitk
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def get_image_info(image_path: str) -> Dict[str, Any]:
    """获取医学图像信息"""
    if not HAS_DEPS:
        raise ImportError("Required dependencies not found. Install with: pip install nibabel pydicom SimpleITK")
    
    image_path = str(image_path)
    
    if image_path.endswith(('.nii', '.nii.gz')):
        return _get_nifti_info(image_path)
    elif image_path.endswith(('.dcm', '.dicom')) or _is_dicom_file(image_path):
        return _get_dicom_info(image_path)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")


def _get_nifti_info(image_path: str) -> Dict[str, Any]:
    """获取 NIfTI 图像信息"""
    img = nib.load(image_path)
    data = img.get_fdata()
    
    # 获取体素间距
    spacing = tuple(abs(img.affine[i, i]) for i in range(3))
    
    # 计算体积
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    total_volume = data.size * voxel_volume
    
    return {
        'format': 'NIfTI',
        'shape': data.shape,
        'dtype': str(data.dtype),
        'spacing': spacing,
        'voxel_volume': voxel_volume,
        'total_volume': total_volume,
        'affine': img.affine.tolist(),
        'intensity_range': (float(data.min()), float(data.max())),
        'mean_intensity': float(data.mean()),
        'std_intensity': float(data.std()),
        'non_zero_count': int(np.count_nonzero(data))
    }


def _get_dicom_info(image_path: str) -> Dict[str, Any]:
    """获取 DICOM 图像信息"""
    try:
        ds = pydicom.dcmread(image_path, stop_before_pixels=True)
        
        info = {
            'format': 'DICOM',
            'patient_id': ds.get('PatientID', 'Unknown'),
            'study_id': ds.get('StudyInstanceUID', 'Unknown'),
            'series_id': ds.get('SeriesInstanceUID', 'Unknown'),
            'modality': ds.get('Modality', 'Unknown'),
            'study_date': ds.get('StudyDate', 'Unknown'),
            'series_description': ds.get('SeriesDescription', 'Unknown')
        }
        
        # 图像尺寸信息
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            info['rows'] = ds.Rows
            info['columns'] = ds.Columns
        
        if hasattr(ds, 'SliceThickness'):
            info['slice_thickness'] = float(ds.SliceThickness)
        
        if hasattr(ds, 'PixelSpacing'):
            info['pixel_spacing'] = [float(x) for x in ds.PixelSpacing]
        
        return info
        
    except Exception as e:
        return {'format': 'DICOM', 'error': str(e)}


def _is_dicom_file(file_path: str) -> bool:
    """检查文件是否为 DICOM 格式"""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False


def normalize_intensity(image_data: np.ndarray, 
                      method: str = 'z_score',
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
    """归一化图像强度"""
    if mask is not None:
        valid_voxels = image_data[mask > 0]
    else:
        valid_voxels = image_data[image_data != 0]
    
    if len(valid_voxels) == 0:
        return image_data
    
    if method == 'z_score':
        mean_val = np.mean(valid_voxels)
        std_val = np.std(valid_voxels)
        
        if std_val > 0:
            normalized = (image_data - mean_val) / std_val
        else:
            normalized = image_data
            
    elif method == 'min_max':
        min_val = np.min(valid_voxels)
        max_val = np.max(valid_voxels)
        
        if max_val > min_val:
            normalized = (image_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image_data)
            
    elif method == 'percentile':
        p1 = np.percentile(valid_voxels, 1)
        p99 = np.percentile(valid_voxels, 99)
        
        if p99 > p1:
            normalized = np.clip((image_data - p1) / (p99 - p1), 0, 1)
        else:
            normalized = np.zeros_like(image_data)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized.astype(np.float32)


def apply_window(image_data: np.ndarray, 
                window_center: float,
                window_width: float) -> np.ndarray:
    """应用窗宽窗位"""
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    windowed = np.clip(image_data, window_min, window_max)
    
    # 归一化到 [0, 1]
    if window_max > window_min:
        windowed = (windowed - window_min) / (window_max - window_min)
    
    return windowed.astype(np.float32)


def resample_spacing(image_data: np.ndarray,
                   original_spacing: Tuple[float, float, float],
                   target_spacing: Tuple[float, float, float],
                   order: int = 1) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """重采样图像间距"""
    import scipy.ndimage as ndimage
    
    zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    
    resampled = ndimage.zoom(image_data, zoom_factors, order=order)
    
    return resampled, target_spacing


def calculate_roi_statistics(image_data: np.ndarray,
                           roi_mask: np.ndarray,
                           spacing: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """计算 ROI 统计信息"""
    roi_voxels = image_data[roi_mask > 0]
    
    if len(roi_voxels) == 0:
        return {'error': 'Empty ROI'}
    
    stats = {
        'voxel_count': int(len(roi_voxels)),
        'mean': float(np.mean(roi_voxels)),
        'std': float(np.std(roi_voxels)),
        'min': float(np.min(roi_voxels)),
        'max': float(np.max(roi_voxels)),
        'median': float(np.median(roi_voxels)),
        'q25': float(np.percentile(roi_voxels, 25)),
        'q75': float(np.percentile(roi_voxels, 75))
    }
    
    if spacing:
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        stats['volume_mm3'] = float(len(roi_voxels) * voxel_volume)
    
    return stats


def create_binary_mask(image_data: np.ndarray,
                      threshold: Union[float, str] = 'otsu',
                      connectivity: int = 1) -> np.ndarray:
    """创建二值掩码"""
    if threshold == 'otsu':
        import cv2
        # 将图像归一化到 0-255
        normalized = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = binary.astype(bool)
    elif threshold == 'mean':
        mask = image_data > np.mean(image_data)
    elif threshold == 'median':
        mask = image_data > np.median(image_data)
    else:
        mask = image_data > float(threshold)
    
    # 形态学操作
    if connectivity > 0:
        from scipy import ndimage
        mask = ndimage.binary_closing(mask, iterations=connectivity)
        mask = ndimage.binary_opening(mask, iterations=connectivity)
    
    return mask.astype(np.uint8)


def find_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """找到最大的连通分量"""
    from scipy import ndimage
    
    labeled_mask, num_features = ndimage.label(binary_mask)
    
    if num_features == 0:
        return binary_mask
    
    # 计算每个连通分量的体素数量
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # 忽略背景
    
    # 找到最大连通分量
    largest_component = np.argmax(component_sizes) + 1
    
    return (labeled_mask == largest_component).astype(np.uint8)


def extract_voxel_coordinates(mask: np.ndarray) -> np.ndarray:
    """提取掩码中体素的坐标"""
    coords = np.where(mask > 0)
    return np.column_stack(coords)


def calculate_center_of_mass(mask: np.ndarray,
                           spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """计算质心"""
    coords = extract_voxel_coordinates(mask)
    
    if len(coords) == 0:
        return np.array([0, 0, 0])
    
    center = np.mean(coords, axis=0)
    
    if spacing:
        center = center * np.array(spacing)
    
    return center


def pad_or_crop_to_shape(image_data: np.ndarray,
                        target_shape: Tuple[int, int, int],
                        pad_mode: str = 'constant',
                        pad_value: float = 0) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """填充或裁剪图像到目标形状"""
    current_shape = np.array(image_data.shape)
    target_shape = np.array(target_shape)
    
    # 计算差异
    shape_diff = target_shape - current_shape
    
    # 如果需要填充
    if np.any(shape_diff > 0):
        padding = []
        for i, diff in enumerate(shape_diff):
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                padding.append((pad_before, pad_after))
            else:
                padding.append((0, 0))
        
        image_data = np.pad(image_data, padding, mode=pad_mode, constant_values=pad_value)
        current_shape = np.array(image_data.shape)
    
    # 如果需要裁剪
    if np.any(current_shape > target_shape):
        start_coords = (current_shape - target_shape) // 2
        end_coords = start_coords + target_shape
        
        slices = [slice(start, end) for start, end in zip(start_coords, end_coords)]
        image_data = image_data[tuple(slices)]
    
    offset = (current_shape - target_shape) // 2
    
    return image_data, tuple(offset)


def validate_medical_image(image_path: str) -> Dict[str, Any]:
    """验证医学图像文件"""
    result = {
        'valid': False,
        'error': None,
        'info': None
    }
    
    try:
        info = get_image_info(image_path)
        
        # 基本验证
        if 'shape' in info:
            shape = info['shape']
            if len(shape) >= 2 and all(s > 0 for s in shape):
                result['valid'] = True
                result['info'] = info
            else:
                result['error'] = f'Invalid image dimensions: {shape}'
        else:
            result['error'] = 'No image dimensions found'
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def convert_intensity_units(image_data: np.ndarray,
                          from_unit: str,
                          to_unit: str,
                          hu_calibration: Optional[Dict[str, float]] = None) -> np.ndarray:
    """转换单位（如 Hounsfield Unit）"""
    if from_unit == 'HU' and to_unit == 'normalized':
        if hu_calibration is None:
            # 默认 CT 校准参数
            air_hu = -1000
            water_hu = 0
        else:
            air_hu = hu_calibration.get('air', -1000)
            water_hu = hu_calibration.get('water', 0)
        
        # 转换到 [0, 1] 范围
        normalized = np.clip((image_data - air_hu) / (water_hu - air_hu), 0, 1)
        return normalized
    
    elif from_unit == 'normalized' and to_unit == 'HU':
        if hu_calibration is None:
            air_hu = -1000
            water_hu = 0
        else:
            air_hu = hu_calibration.get('air', -1000)
            water_hu = hu_calibration.get('water', 0)
        
        hu = image_data * (water_hu - air_hu) + air_hu
        return hu
    
    else:
        return image_data