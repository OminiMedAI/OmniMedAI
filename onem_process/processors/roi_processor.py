"""
3D ROI 区域处理器
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
    MISSING_DEPS = ""

from ..utils import medical_utils


class ROIProcessor:
    """3D ROI 区域处理器"""
    
    def __init__(self, padding: Tuple[int, int, int] = (10, 10, 10)):
        """
        初始化 ROI 处理器
        
        Args:
            padding: ROI 扩展边距 (z, y, x)
        """
        if not HAS_DEPS:
            raise ImportError(f"Missing dependencies. Install with: pip install nibabel opencv-python scipy")
        
        self.padding = padding
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
    
    def extract_roi_from_mask(self, 
                              image_path: str, 
                              mask_path: str, 
                              output_dir: str,
                              roi_name: Optional[str] = None) -> Tuple[str, str]:
        """
        从掩码中提取 ROI 区域
        
        Args:
            image_path: 图像文件路径
            mask_path: 掩码文件路径
            output_dir: 输出目录
            roi_name: ROI 名称，如果为None则自动生成
            
        Returns:
            (roi_image_path, roi_mask_path) ROI 图像和掩码路径
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_dir = Path(output_dir)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # 加载图像和掩码
        image_img = nib.load(str(image_path))
        mask_img = nib.load(str(mask_path))
        
        image_data = image_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # 确保图像和掩码形状一致
        if image_data.shape != mask_data.shape:
            self.logger.warning(f"Image and mask shapes differ: {image_data.shape} vs {mask_data.shape}")
            # 调整掩码形状
            mask_data = self._resize_to_match(mask_data, image_data.shape)
        
        # 找到掩码的边界框
        bbox = self._find_bounding_box(mask_data)
        
        if bbox is None:
            self.logger.warning("No ROI found in mask (mask is empty)")
            return None, None
        
        # 扩展边界框
        expanded_bbox = self._expand_bbox(bbox, image_data.shape, self.padding)
        
        # 提取 ROI
        roi_image = self._extract_roi(image_data, expanded_bbox)
        roi_mask = self._extract_roi(mask_data, expanded_bbox)
        
        # 生成输出路径
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if roi_name is None:
            roi_name = f"{image_path.stem}_roi"
        
        roi_image_path = output_dir / f"{roi_name}_image.nii.gz"
        roi_mask_path = output_dir / f"{roi_name}_mask.nii.gz"
        
        # 调整仿射矩阵
        roi_affine = self._adjust_affine_matrix(image_img.affine, expanded_bbox[:3])
        
        # 保存 ROI
        roi_image_img = nib.Nifti1Image(roi_image.astype(image_data.dtype), roi_affine, image_img.header)
        roi_mask_img = nib.Nifti1Image(roi_mask.astype(mask_data.dtype), roi_affine, mask_img.header)
        
        nib.save(roi_image_img, str(roi_image_path))
        nib.save(roi_mask_img, str(roi_mask_path))
        
        self.logger.info(f"ROI extracted: {roi_image_path}, {roi_mask_path}")
        self.logger.info(f"Original shape: {image_data.shape}, ROI shape: {roi_image.shape}")
        
        return str(roi_image_path), str(roi_mask_path)
    
    def batch_extract_rois(self, 
                          image_dir: str, 
                          mask_dir: str, 
                          output_dir: str,
                          file_pattern: str = "*.nii.gz") -> List[Dict[str, str]]:
        """
        批量提取 ROI
        
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            
        Returns:
            提取结果列表
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        output_dir = Path(output_dir)
        
        results = []
        
        # 查找所有图像文件
        image_files = list(image_dir.glob(file_pattern))
        
        for image_file in image_files:
            # 查找对应的掩码文件
            mask_file = self._find_matching_mask(image_file, mask_dir)
            
            if mask_file is None:
                self.logger.warning(f"No matching mask found for {image_file}")
                continue
            
            try:
                roi_image_path, roi_mask_path = self.extract_roi_from_mask(
                    str(image_file), 
                    str(mask_file), 
                    str(output_dir),
                    image_file.stem
                )
                
                if roi_image_path and roi_mask_path:
                    results.append({
                        'image_file': str(image_file),
                        'mask_file': str(mask_file),
                        'roi_image': roi_image_path,
                        'roi_mask': roi_mask_path
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
        
        self.logger.info(f"Batch processing completed. Successfully processed {len(results)} files.")
        return results
    
    def _find_bounding_box(self, mask_data: np.ndarray) -> Optional[Tuple[Tuple[int, int], ...]]:
        """找到掩码的边界框"""
        if mask_data.max() == 0:
            return None
        
        # 找到非零体素的坐标
        coords = np.where(mask_data > 0)
        
        if len(coords[0]) == 0:
            return None
        
        # 计算每个维度的最小和最大坐标
        bbox = tuple((int(coords[i].min()), int(coords[i].max())) for i in range(len(coords)))
        return bbox
    
    def _expand_bbox(self, 
                    bbox: Tuple[Tuple[int, int], ...], 
                    shape: Tuple[int, ...], 
                    padding: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
        """扩展边界框"""
        expanded = []
        for i, ((start, end), pad) in enumerate(zip(bbox, padding)):
            start_expanded = max(0, start - pad)
            end_expanded = min(shape[i], end + pad + 1)
            expanded.append((start_expanded, end_expanded))
        
        return tuple(expanded[0] + expanded[1] + expanded[2])
    
    def _extract_roi(self, data: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
        """提取 ROI 区域"""
        z_start, z_end, y_start, y_end, x_start, x_end = bbox
        return data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    def _adjust_affine_matrix(self, 
                             original_affine: np.ndarray, 
                             bbox_start: Tuple[int, int, int]) -> np.ndarray:
        """调整仿射矩阵以适应 ROI"""
        # 创建新的仿射矩阵
        new_affine = original_affine.copy()
        
        # 调整原点坐标
        translation = np.dot(original_affine[:3, :3], np.array(bbox_start))
        new_affine[:3, 3] = original_affine[:3, 3] + translation
        
        return new_affine
    
    def _resize_to_match(self, 
                        mask_data: np.ndarray, 
                        target_shape: Tuple[int, ...]) -> np.ndarray:
        """调整掩码大小以匹配目标形状"""
        # 使用最近邻插值调整大小
        zoom_factors = [t / m for t, m in zip(target_shape, mask_data.shape)]
        return ndimage.zoom(mask_data, zoom_factors, order=0)
    
    def _find_matching_mask(self, image_file: Path, mask_dir: Path) -> Optional[Path]:
        """查找与图像文件匹配的掩码文件"""
        # 尝试多种匹配策略
        strategies = [
            lambda img, mask_dir: mask_dir / img.name,  # 完全相同名称
            lambda img, mask_dir: mask_dir / f"{img.stem}_mask{img.suffix}",  # 加后缀
            lambda img, mask_dir: mask_dir / f"{img.stem}_seg{img.suffix}",  # seg 后缀
            lambda img, mask_dir: mask_dir / f"{img.stem}_label{img.suffix}",  # label 后缀
        ]
        
        for strategy in strategies:
            mask_file = strategy(image_file, mask_dir)
            if mask_file.exists():
                return mask_file
        
        return None
    
    def get_roi_statistics(self, roi_path: str) -> Dict[str, Any]:
        """
        获取 ROI 统计信息
        
        Args:
            roi_path: ROI 文件路径
            
        Returns:
            统计信息字典
        """
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()
        
        # 计算体积（voxel 数量）
        voxel_count = np.sum(roi_data > 0)
        
        # 计算实际体积（mm³）
        voxel_volume = np.abs(np.linalg.det(roi_img.affine[:3, :3]))
        actual_volume = voxel_count * voxel_volume
        
        # 计算质心
        coords = np.where(roi_data > 0)
        if len(coords[0]) > 0:
            centroid = tuple(int(np.mean(coords[i])) for i in range(len(coords)))
        else:
            centroid = (0, 0, 0)
        
        # 计算边界框
        if len(coords[0]) > 0:
            bbox = tuple((int(coords[i].min()), int(coords[i].max())) for i in range(len(coords)))
            bbox_size = tuple(end - start + 1 for start, end in bbox)
        else:
            bbox = None
            bbox_size = (0, 0, 0)
        
        return {
            'file_path': roi_path,
            'shape': roi_data.shape,
            'voxel_count': int(voxel_count),
            'actual_volume_mm3': float(actual_volume),
            'centroid': centroid,
            'bounding_box': bbox,
            'bounding_box_size': bbox_size,
            'voxel_spacing': tuple(np.abs(np.diag(roi_img.affine)[:3])),
            'intensity_range': (float(roi_data.min()), float(roi_data.max())) if voxel_count > 0 else (0, 0),
            'mean_intensity': float(np.mean(roi_data[roi_data > 0])) if voxel_count > 0 else 0,
            'std_intensity': float(np.std(roi_data[roi_data > 0])) if voxel_count > 0 else 0
        }