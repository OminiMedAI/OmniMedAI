"""
图像处理器
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

try:
    import nibabel as nib
    import cv2
    from scipy import ndimage
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from ..utils import medical_utils


class ImageProcessor:
    """通用图像处理器"""
    
    def __init__(self):
        """初始化图像处理器"""
        if not HAS_DEPS:
            raise ImportError("Missing dependencies. Install with: pip install nibabel opencv-python scipy")
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def resample_image(self, 
                       image_path: str, 
                       target_spacing: Tuple[float, float, float],
                       output_path: str,
                       interpolation: str = 'linear') -> str:
        """
        重采样图像到目标分辨率
        
        Args:
            image_path: 输入图像路径
            target_spacing: 目标体素间距 (z, y, x)
            output_path: 输出路径
            interpolation: 插值方法 ('linear', 'nearest', 'cubic')
            
        Returns:
            输出文件路径
        """
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata()
        original_spacing = self._get_voxel_spacing(img)
        original_shape = data.shape
        
        # 计算重采样比例
        zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
        
        # 选择插值顺序
        order = {'linear': 1, 'nearest': 0, 'cubic': 3}.get(interpolation, 1)
        
        # 执行重采样
        resampled_data = ndimage.zoom(data, zoom_factors, order=order)
        
        # 更新仿射矩阵
        new_affine = self._update_affine_for_resampling(img.affine, zoom_factors)
        
        # 保存重采样后的图像
        resampled_img = nib.Nifti1Image(resampled_data, new_affine, img.header)
        nib.save(resampled_img, output_path)
        
        self.logger.info(f"Resampled {image_path} from {original_spacing} to {target_spacing}")
        self.logger.info(f"Shape changed from {original_shape} to {resampled_data.shape}")
        
        return output_path
    
    def normalize_image(self, 
                       image_path: str, 
                       method: str = 'z_score',
                       output_path: Optional[str] = None) -> str:
        """
        归一化图像
        
        Args:
            image_path: 输入图像路径
            method: 归一化方法 ('z_score', 'min_max', 'percentile')
            output_path: 输出路径，如果为None则原地修改
            
        Returns:
            输出文件路径
        """
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # 执行归一化
        if method == 'z_score':
            normalized_data = self._z_score_normalize(data)
        elif method == 'min_max':
            normalized_data = self._min_max_normalize(data)
        elif method == 'percentile':
            normalized_data = self._percentile_normalize(data)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # 确定输出路径
        if output_path is None:
            output_path = image_path
        
        # 保存归一化后的图像
        normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
        nib.save(normalized_img, output_path)
        
        self.logger.info(f"Normalized {image_path} using {method} method")
        
        return output_path
    
    def crop_center(self, 
                   image_path: str, 
                   crop_size: Tuple[int, int, int],
                   output_path: str) -> str:
        """
        裁剪图像中心区域
        
        Args:
            image_path: 输入图像路径
            crop_size: 裁剪尺寸 (z, y, x)
            output_path: 输出路径
            
        Returns:
            输出文件路径
        """
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata()
        original_shape = data.shape
        
        # 计算裁剪起始位置
        start_coords = []
        for dim, target_size in zip(original_shape, crop_size):
            start = max(0, (dim - target_size) // 2)
            start_coords.append(start)
        
        # 计算裁剪结束位置
        end_coords = [start + size for start, size in zip(start_coords, crop_size)]
        
        # 执行裁剪
        cropped_data = data[
            start_coords[0]:end_coords[0],
            start_coords[1]:end_coords[1],
            start_coords[2]:end_coords[2]
        ]
        
        # 调整仿射矩阵
        new_affine = self._adjust_affine_for_crop(img.affine, start_coords)
        
        # 保存裁剪后的图像
        cropped_img = nib.Nifti1Image(cropped_data, new_affine, img.header)
        nib.save(cropped_img, output_path)
        
        self.logger.info(f"Cropped {image_path} from {original_shape} to {cropped_data.shape}")
        
        return output_path
    
    def pad_image(self, 
                  image_path: str, 
                  target_shape: Tuple[int, int, int],
                  output_path: str,
                  pad_value: Union[float, str] = 'min') -> str:
        """
        填充图像到目标形状
        
        Args:
            image_path: 输入图像路径
            target_shape: 目标形状 (z, y, x)
            output_path: 输出路径
            pad_value: 填充值 ('min', 'max', 'mean' 或具体数值)
            
        Returns:
            输出文件路径
        """
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata()
        original_shape = data.shape
        
        # 计算填充量
        padding = []
        for orig, target in zip(original_shape, target_shape):
            total_pad = max(0, target - orig)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.append((pad_before, pad_after))
        
        # 确定填充值
        if pad_value == 'min':
            fill_value = data.min()
        elif pad_value == 'max':
            fill_value = data.max()
        elif pad_value == 'mean':
            fill_value = data.mean()
        else:
            fill_value = float(pad_value)
        
        # 执行填充
        padded_data = np.pad(data, padding, mode='constant', constant_values=fill_value)
        
        # 调整仿射矩阵
        new_affine = self._adjust_affine_for_pad(img.affine, padding)
        
        # 保存填充后的图像
        padded_img = nib.Nifti1Image(padded_data, new_affine, img.header)
        nib.save(padded_img, output_path)
        
        self.logger.info(f"Padded {image_path} from {original_shape} to {padded_data.shape}")
        
        return output_path
    
    def apply_windowing(self, 
                       image_path: str, 
                       window_center: float,
                       window_width: float,
                       output_path: str) -> str:
        """
        应用窗宽窗位
        
        Args:
            image_path: 输入图像路径
            window_center: 窗位
            window_width: 窗宽
            output_path: 输出路径
            
        Returns:
            输出文件路径
        """
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # 计算窗宽窗位范围
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # 应用窗宽窗位
        windowed_data = np.clip(data, window_min, window_max)
        
        # 保存窗宽窗位后的图像
        windowed_img = nib.Nifti1Image(windowed_data, img.affine, img.header)
        nib.save(windowed_img, output_path)
        
        self.logger.info(f"Applied windowing to {image_path}: center={window_center}, width={window_width}")
        
        return output_path
    
    def _get_voxel_spacing(self, img: nib.Nifti1Image) -> Tuple[float, float, float]:
        """获取体素间距"""
        return tuple(abs(img.affine[i, i]) for i in range(3))
    
    def _update_affine_for_resampling(self, 
                                     affine: np.ndarray, 
                                     zoom_factors: List[float]) -> np.ndarray:
        """更新重采样后的仿射矩阵"""
        new_affine = affine.copy()
        for i in range(3):
            new_affine[i, i] = new_affine[i, i] / zoom_factors[i]
        return new_affine
    
    def _adjust_affine_for_crop(self, 
                               affine: np.ndarray, 
                               start_coords: List[int]) -> np.ndarray:
        """调整裁剪后的仿射矩阵"""
        new_affine = affine.copy()
        translation = np.dot(affine[:3, :3], np.array(start_coords))
        new_affine[:3, 3] = new_affine[:3, 3] + translation
        return new_affine
    
    def _adjust_affine_for_pad(self, 
                              affine: np.ndarray, 
                              padding: List[Tuple[int, int]]) -> np.ndarray:
        """调整填充后的仿射矩阵"""
        new_affine = affine.copy()
        translation = np.dot(affine[:3, :3], np.array([-pad[0] for pad in padding]))
        new_affine[:3, 3] = new_affine[:3, 3] + translation
        return new_affine
    
    def _z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score 归一化"""
        non_zero_mask = data != 0
        if np.sum(non_zero_mask) == 0:
            return data
        
        mean_val = np.mean(data[non_zero_mask])
        std_val = np.std(data[non_zero_mask])
        
        if std_val > 0:
            normalized = (data - mean_val) / std_val
            normalized[~non_zero_mask] = 0
            return normalized
        else:
            return data
    
    def _min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0, 1]"""
        min_val = data.min()
        max_val = data.max()
        
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(data)
    
    def _percentile_normalize(self, 
                             data: np.ndarray, 
                             low_percentile: float = 1.0,
                             high_percentile: float = 99.0) -> np.ndarray:
        """百分位数归一化"""
        low_val = np.percentile(data, low_percentile)
        high_val = np.percentile(data, high_percentile)
        
        if high_val > low_val:
            return np.clip((data - low_val) / (high_val - low_val), 0, 1)
        else:
            return np.zeros_like(data)