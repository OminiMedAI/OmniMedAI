"""
局部放射组学特征提取器
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    import nibabel as nib
    from radiomics import featureextractor, getFeatureClasses
    from radiomics import imageoperations
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS = "Install with: pip install pyradiomics"

from ..utils import habitat_utils, radiomics_utils


class LocalRadiomicsExtractor:
    """局部放射组学特征提取器"""
    
    def __init__(self, 
                 kernel_size: Tuple[int, int, int] = (5, 5, 5),
                 feature_types: Optional[List[str]] = None,
                 bin_width: int = 25,
                 resampled_pixel_spacing: Optional[Tuple[float, float, float]] = None,
                 interpolator: str = 'sitkBSpline',
                 weight_center: Optional[Tuple[float, float, float]] = None,
                 weight_radius: Optional[float] = None,
                 n_jobs: int = 1):
        """
        初始化局部放射组学特征提取器
        
        Args:
            kernel_size: 3D 核大小 (z, y, x)
            feature_types: 提取的特征类型
            bin_width: 直方图分箱宽度
            resampled_pixel_spacing: 重采样体素间距
            interpolator: 插值方法
            weight_center: 权重中心
            weight_radius: 权重半径
            n_jobs: 并行处理数量
        """
        if not HAS_DEPS:
            raise ImportError(f"Missing dependencies: {MISSING_DEPS}")
        
        self.kernel_size = kernel_size
        self.feature_types = feature_types or [
            'firstorder', 'glcm', 'glrlm', 'glszm', 'gltdm', 'ngtdm'
        ]
        self.bin_width = bin_width
        self.resampled_pixel_spacing = resampled_pixel_spacing
        self.interpolator = interpolator
        self.weight_center = weight_center
        self.weight_radius = weight_radius
        self.n_jobs = n_jobs
        
        self.logger = self._setup_logger()
        
        # 初始化放射组学提取器
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self._configure_extractor()
        
        # 抑制 radiomics 警告
        warnings.filterwarnings('ignore', category=UserWarning, module='radiomics')
    
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
    
    def _configure_extractor(self) -> None:
        """配置放射组学提取器"""
        settings = {
            'binWidth': self.bin_width,
            'interpolator': self.interpolator,
            'resampledPixelSpacing': self.resampled_pixel_spacing,
            'weightingNorm': None
        }
        
        if self.weight_center is not None:
            settings['weightCenter'] = self.weight_center
        if self.weight_radius is not None:
            settings['weightRadius'] = self.weight_radius
        
        for key, value in settings.items():
            if value is not None:
                self.extractor.settings[key] = value
        
        # 启用指定特征类型
        for feature_type in self.feature_types:
            self.extractor.enableFeatureClassByName(feature_type)
        
        self.logger.info(f"Configured extractor with features: {self.feature_types}")
    
    def extract_local_features(self,
                            image_path: str,
                            mask_path: str,
                            output_path: Optional[str] = None,
                            coordinates: Optional[List[Tuple[int, int, int]]] = None,
                            step_size: int = 1) -> Dict[str, Any]:
        """
        提取局部放射组学特征
        
        Args:
            image_path: 图像文件路径
            mask_path: 掩码文件路径
            output_path: 输出 .npy 文件路径
            coordinates: 指定坐标列表，如果为None则提取mask中所有点
            step_size: 步长，用于减少计算量
            
        Returns:
            包含特征和元数据的字典
        """
        # 加载图像和掩码
        image_data, image_header, image_spacing = self._load_nifti(image_path)
        mask_data, _, _ = self._load_nifti(mask_path)
        
        self.logger.info(f"Loaded image: {image_data.shape}, mask: {mask_data.shape}")
        
        # 确定计算坐标
        if coordinates is None:
            coordinates = self._get_roi_coordinates(mask_data, step_size)
        else:
            coordinates = [(z, y, x) for z, y, x in coordinates 
                         if self._is_valid_coordinate(z, y, x, image_data.shape)]
        
        self.logger.info(f"Computing features for {len(coordinates)} points")
        
        # 提取特征
        features = self._compute_features_batch(
            image_data, mask_data, image_spacing, coordinates
        )
        
        # 保存结果
        result = {
            'features': features,
            'coordinates': coordinates,
            'image_shape': image_data.shape,
            'kernel_size': self.kernel_size,
            'feature_names': list(features.keys()) if features else [],
            'metadata': {
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'image_spacing': image_spacing,
                'n_voxels': len(coordinates),
                'feature_types': self.feature_types
            }
        }
        
        if output_path:
            self._save_features(result, output_path)
        
        return result
    
    def _load_nifti(self, file_path: str) -> Tuple[np.ndarray, Any, Tuple[float, float, float]]:
        """加载 NIfTI 文件"""
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        spacing = tuple(img.header.get_zooms()[:3])
        
        return data, header, spacing
    
    def _get_roi_coordinates(self, mask_data: np.ndarray, step_size: int = 1) -> List[Tuple[int, int, int]]:
        """获取 ROI 区域内的坐标"""
        coords = np.where(mask_data > 0)
        
        # 应用步长
        coordinates = []
        for i in range(0, len(coords[0]), step_size):
            z, y, x = coords[0][i], coords[1][i], coords[2][i]
            coordinates.append((int(z), int(y), int(x)))
        
        return coordinates
    
    def _is_valid_coordinate(self, z: int, y: int, x: int, shape: Tuple[int, int, int]) -> bool:
        """检查坐标是否有效"""
        return (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2])
    
    def _compute_features_batch(self,
                              image_data: np.ndarray,
                              mask_data: np.ndarray,
                              image_spacing: Tuple[float, float, float],
                              coordinates: List[Tuple[int, int, int]]) -> Dict[str, np.ndarray]:
        """批量计算特征"""
        if self.n_jobs == 1:
            return self._compute_features_sequential(
                image_data, mask_data, image_spacing, coordinates
            )
        else:
            return self._compute_features_parallel(
                image_data, mask_data, image_spacing, coordinates
            )
    
    def _compute_features_sequential(self,
                                   image_data: np.ndarray,
                                   mask_data: np.ndarray,
                                   image_spacing: Tuple[float, float, float],
                                   coordinates: List[Tuple[int, int, int]]) -> Dict[str, np.ndarray]:
        """顺序计算特征"""
        features = {}
        
        for i, (z, y, x) in enumerate(coordinates):
            if i % 100 == 0:
                self.logger.info(f"Processing point {i+1}/{len(coordinates)}")
            
            try:
                point_features = self._extract_single_point_features(
                    image_data, mask_data, image_spacing, z, y, x
                )
                
                if not features:
                    # 初始化特征数组
                    for feature_name, feature_value in point_features.items():
                        features[feature_name] = []
                
                # 添加特征值
                for feature_name, feature_value in point_features.items():
                    features[feature_name].append(feature_value)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process point ({z}, {y}, {x}): {e}")
                continue
        
        # 转换为 numpy 数组
        for feature_name in features:
            features[feature_name] = np.array(features[feature_name])
        
        return features
    
    def _compute_features_parallel(self,
                                image_data: np.ndarray,
                                mask_data: np.ndarray,
                                image_spacing: Tuple[float, float, float],
                                coordinates: List[Tuple[int, int, int]]) -> Dict[str, np.ndarray]:
        """并行计算特征"""
        features_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交任务
            future_to_coord = {
                executor.submit(
                    self._extract_single_point_features,
                    image_data, mask_data, image_spacing, z, y, x
                ): (z, y, x)
                for z, y, x in coordinates
            }
            
            # 收集结果
            for future in as_completed(future_to_coord):
                coord = future_to_coord[future]
                try:
                    point_features = future.result()
                    
                    if not features_dict:
                        # 初始化特征字典
                        for feature_name in point_features:
                            features_dict[feature_name] = {}
                    
                    # 存储特征
                    for feature_name, feature_value in point_features.items():
                        features_dict[feature_name][coord] = feature_value
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process point {coord}: {e}")
        
        # 转换为有序数组
        features = {}
        feature_coords = [coord for coord in coordinates if coord in list(list(features_dict.values())[0].keys())]
        
        for feature_name, coord_dict in features_dict.items():
            features[feature_name] = np.array([coord_dict[coord] for coord in feature_coords])
        
        return features
    
    def _extract_single_point_features(self,
                                    image_data: np.ndarray,
                                    mask_data: np.ndarray,
                                    image_spacing: Tuple[float, float, float],
                                    center_z: int,
                                    center_y: int,
                                    center_x: int) -> Dict[str, float]:
        """提取单个点的特征"""
        # 提取局部区域
        local_image, local_mask = self._extract_local_region(
            image_data, mask_data, center_z, center_y, center_x
        )
        
        if local_image is None or np.sum(local_mask) == 0:
            return {}
        
        # 转换为 SimpleITK 图像
        sitk_image, sitk_mask = radiomics_utils.numpy_to_sitk(
            local_image, local_mask, image_spacing
        )
        
        # 提取特征
        try:
            feature_values = self.extractor.execute(sitk_image, sitk_mask)
            
            # 过滤并返回数值特征
            numeric_features = {}
            for feature_name, feature_value in feature_values.items():
                if isinstance(feature_value, (int, float)):
                    numeric_features[feature_name] = float(feature_value)
            
            return numeric_features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed at ({center_z}, {center_y}, {center_x}): {e}")
            return {}
    
    def _extract_local_region(self,
                             image_data: np.ndarray,
                             mask_data: np.ndarray,
                             center_z: int,
                             center_y: int,
                             center_x: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """提取局部区域"""
        kz, ky, kx = self.kernel_size
        kz_half, ky_half, kx_half = kz // 2, ky // 2, kx // 2
        
        # 计算边界
        z_start = max(0, center_z - kz_half)
        z_end = min(image_data.shape[0], center_z + kz_half + 1)
        y_start = max(0, center_y - ky_half)
        y_end = min(image_data.shape[1], center_y + ky_half + 1)
        x_start = max(0, center_x - kx_half)
        x_end = min(image_data.shape[2], center_x + kx_half + 1)
        
        # 提取区域
        local_image = image_data[z_start:z_end, y_start:y_end, x_start:x_end]
        local_mask = mask_data[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # 创建以指定点为中心的掩码
        local_region_mask = np.zeros_like(local_mask, dtype=np.uint8)
        
        # 计算在局部区域中的相对坐标
        rel_z = center_z - z_start
        rel_y = center_y - y_start
        rel_x = center_x - x_start
        
        # 创建球状掩码
        kz_actual, ky_actual, kx_actual = local_image.shape
        kz_half_actual, ky_half_actual, kx_half_actual = kz_actual // 2, ky_actual // 2, kx_actual // 2
        
        for dz in range(-kz_half_actual, kz_half_actual + 1):
            for dy in range(-ky_half_actual, ky_half_actual + 1):
                for dx in range(-kx_half_actual, kx_half_actual + 1):
                    abs_dz, abs_dy, abs_dx = abs(dz), abs(dy), abs(dx)
                    if (abs_dz**2 + abs_dy**2 + abs_dx**2 <= 
                        (kz_half_actual**2 + ky_half_actual**2 + kx_half_actual**2)):
                        
                        z_idx = rel_z + dz
                        y_idx = rel_y + dy
                        x_idx = rel_x + dx
                        
                        if (0 <= z_idx < kz_actual and 
                            0 <= y_idx < ky_actual and 
                            0 <= x_idx < kx_actual):
                            local_region_mask[z_idx, y_idx, x_idx] = 1
        
        # 只保留原掩码内的区域
        local_mask = local_mask * local_region_mask
        
        if np.sum(local_mask) == 0:
            return None, None
        
        return local_image, local_mask
    
    def _save_features(self, features_dict: Dict[str, Any], output_path: str) -> None:
        """保存特征到 .npy 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存主要数据
        np.save(output_path, features_dict)
        
        # 保存元数据
        metadata_path = output_path.with_suffix('.json')
        habitat_utils.save_json(features_dict['metadata'], metadata_path)
        
        self.logger.info(f"Saved features to {output_path}")
    
    def batch_extract_features(self,
                           images_dir: str,
                           masks_dir: str,
                           output_dir: str,
                           file_pattern: str = "*.nii.gz",
                           step_size: int = 1) -> List[Dict[str, str]]:
        """
        批量提取特征
        
        Args:
            images_dir: 图像目录
            masks_dir: 掩码目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            step_size: 步长
            
        Returns:
            处理结果列表
        """
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找图像文件
        image_files = list(images_dir.glob(file_pattern))
        results = []
        
        for image_file in image_files:
            # 查找对应的掩码文件
            mask_file = habitat_utils.find_matching_mask(image_file, masks_dir)
            
            if mask_file is None:
                self.logger.warning(f"No matching mask found for {image_file}")
                continue
            
            try:
                # 生成输出文件名
                output_file = output_dir / f"{image_file.stem}_features.npy"
                
                # 提取特征
                result = self.extract_local_features(
                    str(image_file),
                    str(mask_file),
                    str(output_file),
                    step_size=step_size
                )
                
                results.append({
                    'image_file': str(image_file),
                    'mask_file': str(mask_file),
                    'output_file': str(output_file),
                    'n_voxels': result['metadata']['n_voxels'],
                    'feature_count': len(result['metadata']['feature_types']) * len(result['feature_names'])
                })
                
                self.logger.info(f"Completed: {image_file.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
                results.append({
                    'image_file': str(image_file),
                    'mask_file': str(mask_file),
                    'error': str(e)
                })
        
        return results
    
    def load_features(self, features_path: str) -> Dict[str, Any]:
        """加载保存的特征文件"""
        features_path = Path(features_path)
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # 加载特征数据
        features_data = np.load(features_path, allow_pickle=True).item()
        
        # 加载元数据（如果存在）
        metadata_path = features_path.with_suffix('.json')
        if metadata_path.exists():
            metadata = habitat_utils.load_json(metadata_path)
            features_data['metadata'].update(metadata)
        
        return features_data
    
    def get_feature_summary(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """获取特征摘要统计"""
        if not features_dict or 'features' not in features_dict:
            return {}
        
        features = features_dict['features']
        summary = {}
        
        for feature_name, feature_values in features.items():
            if len(feature_values) > 0:
                feature_array = np.array(feature_values)
                summary[feature_name] = {
                    'count': len(feature_array),
                    'mean': float(np.mean(feature_array)),
                    'std': float(np.std(feature_array)),
                    'min': float(np.min(feature_array)),
                    'max': float(np.max(feature_array)),
                    'q25': float(np.percentile(feature_array, 25)),
                    'median': float(np.median(feature_array)),
                    'q75': float(np.percentile(feature_array, 75))
                }
        
        return summary