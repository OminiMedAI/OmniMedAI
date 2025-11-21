"""
放射组学工具函数
"""

import numpy as np
from typing import Tuple, Any, Dict, List, Optional

try:
    import SimpleITK as sitk
    from radiomics import imageoperations
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS = "Install with: pip install SimpleITK pyradiomics"


def numpy_to_sitk(image_data: np.ndarray, 
                 mask_data: np.ndarray,
                 spacing: Tuple[float, float, float]) -> Tuple[Any, Any]:
    """
    将 numpy 数组转换为 SimpleITK 图像
    
    Args:
        image_data: 图像数据
        mask_data: 掩码数据
        spacing: 体素间距
        
    Returns:
        (sitk_image, sitk_mask)
    """
    if not HAS_DEPS:
        raise ImportError(f"Missing dependencies: {MISSING_DEPS}")
    
    # 确保数据类型正确
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
    
    if mask_data.dtype != np.uint8:
        mask_data = mask_data.astype(np.uint8)
    
    # 创建 SimpleITK 图像
    sitk_image = sitk.GetImageFromArray(image_data)
    sitk_mask = sitk.GetImageFromArray(mask_data)
    
    # 设置间距
    sitk_image.SetSpacing(spacing)
    sitk_mask.SetSpacing(spacing)
    
    # 设置原点
    sitk_image.SetOrigin((0.0, 0.0, 0.0))
    sitk_mask.SetOrigin((0.0, 0.0, 0.0))
    
    return sitk_image, sitk_mask


def sitk_to_numpy(sitk_image: Any, sitk_mask: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 SimpleITK 图像转换为 numpy 数组
    
    Args:
        sitk_image: SimpleITK 图像
        sitk_mask: SimpleITK 掩码
        
    Returns:
        (image_data, mask_data)
    """
    if not HAS_DEPS:
        raise ImportError(f"Missing dependencies: {MISSING_DEPS}")
    
    image_data = sitk.GetArrayFromImage(sitk_image)
    mask_data = sitk.GetArrayFromImage(sitk_mask)
    
    return image_data, mask_data


def preprocess_image(image_data: np.ndarray,
                   intensity_normalization: str = 'none',
                   rescale_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
    """
    预处理图像数据
    
    Args:
        image_data: 输入图像数据
        intensity_normalization: 强度归一化方法 ('none', 'zscore', 'minmax', 'robust')
        rescale_range: 重缩放范围
        
    Returns:
        预处理后的图像数据
    """
    processed_data = image_data.copy().astype(np.float32)
    
    # 移除无效值
    finite_mask = np.isfinite(processed_data)
    if not np.any(finite_mask):
        return processed_data
    
    finite_values = processed_data[finite_mask]
    
    if intensity_normalization == 'zscore':
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        if std_val > 0:
            processed_data = (processed_data - mean_val) / std_val
    
    elif intensity_normalization == 'minmax':
        min_val = np.min(finite_values)
        max_val = np.max(finite_values)
        if max_val > min_val:
            processed_data = (processed_data - min_val) / (max_val - min_val)
    
    elif intensity_normalization == 'robust':
        q1, q3 = np.percentile(finite_values, [25, 75])
        iqr = q3 - q1
        median_val = np.median(finite_values)
        if iqr > 0:
            processed_data = (processed_data - median_val) / iqr
    
    # 重缩放到指定范围
    if rescale_range and (rescale_range[0] != 0.0 or rescale_range[1] != 1.0):
        min_val = np.min(processed_data[finite_mask])
        max_val = np.max(processed_data[finite_mask])
        
        if max_val > min_val:
            current_range = max_val - min_val
            target_range = rescale_range[1] - rescale_range[0]
            processed_data = (processed_data - min_val) / current_range * target_range + rescale_range[0]
    
    return processed_data


def apply_intensity_window(image_data: np.ndarray,
                        window_center: float,
                        window_width: float,
                        rescale_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
    """
    应用强度窗
    
    Args:
        image_data: 输入图像数据
        window_center: 窗位
        window_width: 窗宽
        rescale_range: 重缩放范围
        
    Returns:
        窗宽处理后的图像数据
    """
    # 计算窗宽范围
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # 应用窗宽
    windowed_data = np.clip(image_data, window_min, window_max)
    
    # 重缩放到指定范围
    if window_max > window_min:
        windowed_data = (windowed_data - window_min) / (window_max - window_min)
        windowed_data = windowed_data * (rescale_range[1] - rescale_range[0]) + rescale_range[0]
    
    return windowed_data


def extract_texture_features(image_data: np.ndarray,
                        mask_data: np.ndarray,
                        distances: List[int] = [1],
                        angles: List[int] = [0, 45, 90, 135],
                        levels: int = 256) -> Dict[str, float]:
    """
    提取纹理特征（简化版 GLCM）
    
    Args:
        image_data: 输入图像数据
        mask_data: 掩码数据
        distances: 像素距离列表
        angles: 角度列表
        levels: 灰度级数
        
    Returns:
        纹理特征字典
    """
    try:
        from skimage.feature import greycomatrix, greycoprops
        
        # 预处理图像
        roi_image = image_data[mask_data > 0]
        
        if len(roi_image) == 0:
            return {}
        
        # 归一化到指定灰度级数
        min_val = np.min(roi_image)
        max_val = np.max(roi_image)
        
        if max_val > min_val:
            normalized_image = ((image_data - min_val) / (max_val - min_val) * (levels - 1)).astype(np.uint8)
        else:
            normalized_image = np.zeros_like(image_data, dtype=np.uint8)
        
        # 确保掩码也归一化
        normalized_mask = mask_data
        
        # 计算GLCM
        try:
            glcm = greycomatrix(normalized_image, normalized_mask,
                             distances=distances, angles=angles,
                             levels=levels, symmetric=True, normed=True)
            
            # 提取特征
            features = {}
            
            # 对比度
            features['contrast'] = np.mean(greycoprops(glcm, 'contrast'))
            
            # 相关性
            features['correlation'] = np.mean(greycoprops(glcm, 'correlation'))
            
            # 能量
            features['energy'] = np.mean(greycoprops(glcm, 'energy'))
            
            # 同质性
            features['homogeneity'] = np.mean(greycoprops(glcm, 'homogeneity'))
            
            # 逆差矩
            features['idm'] = np.mean(greycoprops(glcm, 'homogeneity'))
            
            return features
            
        except Exception as e:
            print(f"GLCM calculation failed: {e}")
            return {}
            
    except ImportError:
        return {}


def extract_intensity_features(image_data: np.ndarray,
                          mask_data: np.ndarray) -> Dict[str, float]:
    """
    提取强度特征
    
    Args:
        image_data: 输入图像数据
        mask_data: 掩码数据
        
    Returns:
        强度特征字典
    """
    roi_image = image_data[mask_data > 0]
    
    if len(roi_image) == 0:
        return {}
    
    features = {}
    
    # 基本统计特征
    features['mean'] = np.mean(roi_image)
    features['std'] = np.std(roi_image)
    features['min'] = np.min(roi_image)
    features['max'] = np.max(roi_image)
    features['median'] = np.median(roi_image)
    
    # 百分位数
    features['q10'] = np.percentile(roi_image, 10)
    features['q25'] = np.percentile(roi_image, 25)
    features['q75'] = np.percentile(roi_image, 75)
    features['q90'] = np.percentile(roi_image, 90)
    
    # 形状特征
    features['skewness'] = float(((roi_image - np.mean(roi_image))**3).mean() / (np.std(roi_image)**3 + 1e-8))
    features['kurtosis'] = float(((roi_image - np.mean(roi_image))**4).mean() / (np.std(roi_image)**4 + 1e-8))
    
    # 范围特征
    features['range'] = np.max(roi_image) - np.min(roi_image)
    features['iqr'] = np.percentile(roi_image, 75) - np.percentile(roi_image, 25)
    
    # 离散特征
    features['entropy'] = -np.sum(np.histogram(roi_image, bins=256)[0] / len(roi_image) * 
                               np.log2(np.histogram(roi_image, bins=256)[0] / len(roi_image) + 1e-8))
    
    return features


def extract_morphological_features(mask_data: np.ndarray) -> Dict[str, float]:
    """
    提取形态特征（基于掩码）
    
    Args:
        mask_data: 掩码数据
        
    Returns:
        形状特征字典
    """
    try:
        from skimage.measure import regionprops, label
        
        # 标记连通区域
        labeled_mask = label(mask_data > 0)
        
        if len(np.unique(labeled_mask)) <= 1:  # 只有背景
            return {}
        
        # 计算区域属性
        props = regionprops(labeled_mask)
        
        if not props:
            return {}
        
        # 主要区域的特征
        main_prop = props[0]  # 通常最大的区域
        
        features = {}
        
        # 基本形状特征
        features['area'] = main_prop.area
        features['perimeter'] = main_prop.perimeter
        features['equivalent_diameter'] = main_prop.equivalent_diameter
        features['solidity'] = main_prop.solidity
        features['compactness'] = 4 * np.pi * main_prop.area / (main_prop.perimeter**2 + 1e-8)
        
        # 椭圆拟合特征
        features['eccentricity'] = main_prop.eccentricity
        features['major_axis_length'] = main_prop.major_axis_length
        features['minor_axis_length'] = main_prop.minor_axis_length
        
        # 3D 特征（如果适用）
        if hasattr(main_prop, 'volume'):
            features['volume'] = main_prop.volume
            features['surface_area'] = main_prop.surface_area
            features['sphericity'] = main_prop.sphericity
        
        # 方向特征
        if hasattr(main_prop, 'orientation'):
            features['orientation'] = main_prop.orientation
        
        return features
        
    except ImportError:
        return {}
    except Exception as e:
        print(f"Morphological feature extraction failed: {e}")
        return {}


def compute_feature_importance_radiomics(features_dict: Dict[str, Dict[str, float]],
                                   method: str = 'variance') -> Dict[str, float]:
    """
    计算放射组学特征重要性
    
    Args:
        features_dict: 特征字典
        method: 重要性计算方法
        
    Returns:
        特征重要性字典
    """
    if not features_dict:
        return {}
    
    # 收集所有特征值
    all_values = {}
    
    for feature_name, feature_values in features_dict.items():
        if isinstance(feature_values, list):
            values = np.array(feature_values)
            # 过滤有效值
            valid_mask = np.isfinite(values)
            if np.any(valid_mask):
                all_values[feature_name] = values[valid_mask]
    
    if not all_values:
        return {}
    
    importance = {}
    
    if method == 'variance':
        # 基于方差的重要性
        for feature_name, values in all_values.items():
            importance[feature_name] = float(np.var(values))
    
    elif method == 'entropy':
        # 基于信息熵的重要性
        for feature_name, values in all_values.items():
            hist, _ = np.histogram(values, bins=50)
            hist = hist / np.sum(hist)  # 归一化
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            importance[feature_name] = float(entropy)
    
    elif method == 'range':
        # 基于值域的重要性
        for feature_name, values in all_values.items():
            importance[feature_name] = float(np.max(values) - np.min(values))
    
    # 归一化重要性
    max_importance = max(importance.values()) if importance.values() else 1.0
    for feature_name in importance:
        importance[feature_name] /= max_importance
    
    return importance


def validate_radiomics_features(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证放射组学特征的质量
    
    Args:
        features_dict: 特征字典
        
    Returns:
        验证结果
    """
    validation_result = {
        'valid': True,
        'warnings': [],
        'statistics': {}
    }
    
    if 'features' not in features_dict:
        validation_result['valid'] = False
        validation_result['warnings'].append("No features found")
        return validation_result
    
    features = features_dict['features']
    
    # 统计信息
    total_features = len(features)
    features_with_data = sum(1 for values in features.values() if len(values) > 0)
    
    validation_result['statistics'] = {
        'total_feature_types': total_features,
        'features_with_data': features_with_data,
        'coordinates_count': len(features_dict.get('coordinates', [])),
        'missing_features': total_features - features_with_data
    }
    
    # 检查特征质量
    missing_values_count = 0
    infinite_values_count = 0
    zero_variance_count = 0
    
    for feature_name, feature_values in features.items():
        if len(feature_values) == 0:
            continue
        
        feature_array = np.array(feature_values)
        
        # 检查缺失值
        missing_count = np.sum(~np.isfinite(feature_array))
        missing_values_count += missing_count
        
        # 检查零方差
        if np.var(feature_array) == 0:
            zero_variance_count += 1
        
        # 检查无穷值
        infinite_count = np.sum(np.isinf(feature_array))
        infinite_values_count += infinite_count
    
    # 添加警告
    if missing_values_count > 0:
        validation_result['warnings'].append(f"Found {missing_values_count} missing values")
    
    if infinite_values_count > 0:
        validation_result['warnings'].append(f"Found {infinite_values_count} infinite values")
    
    if zero_variance_count > 0:
        validation_result['warnings'].append(f"Found {zero_variance_count} features with zero variance")
    
    # 总体质量评分
    total_values = sum(len(values) for values in features.values())
    if total_values > 0:
        quality_score = 1.0 - (missing_values_count + infinite_values_count) / total_values
        validation_result['quality_score'] = quality_score
        
        if quality_score < 0.9:
            validation_result['valid'] = False
            validation_result['warnings'].append(f"Low quality score: {quality_score:.3f}")
    
    return validation_result