"""
生态分析工具函数
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def find_matching_mask(image_path: Path, masks_dir: Path) -> Optional[Path]:
    """查找与图像匹配的掩码文件"""
    if not masks_dir.exists():
        return None
    
    image_stem = image_path.stem
    
    # 尝试多种匹配模式
    patterns = [
        f"{image_stem}_mask.nii.gz",
        f"{image_stem}_mask.nii",
        f"{image_stem}_seg.nii.gz",
        f"{image_stem}_seg.nii",
        f"{image_stem}_label.nii.gz",
        f"{image_stem}_label.nii",
        f"{image_stem}_roi.nii.gz",
        f"{image_stem}_roi.nii",
        f"{image_stem}.nii.gz",  # 同名文件
        f"{image_stem}.nii"
    ]
    
    for pattern in patterns:
        mask_file = masks_dir / pattern
        if mask_file.exists():
            return mask_file
    
    return None


def save_json(data: Dict[Any, Any], file_path: str) -> None:
    """保存数据为 JSON 文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Dict[Any, Any]:
    """加载 JSON 文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_directory(base_dir: str, sub_dir: str) -> Path:
    """创建输出目录"""
    output_dir = Path(base_dir) / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_image_mask_pair(image_path: str, mask_path: str) -> Dict[str, Any]:
    """验证图像和掩码对"""
    validation_result = {
        'valid': False,
        'error': None,
        'image_info': None,
        'mask_info': None,
        'compatibility': None
    }
    
    try:
        import nibabel as nib
        
        # 检查图像文件
        if not Path(image_path).exists():
            validation_result['error'] = f"Image file not found: {image_path}"
            return validation_result
        
        img = nib.load(image_path)
        image_data = img.get_fdata()
        image_shape = image_data.shape
        image_affine = img.affine
        
        validation_result['image_info'] = {
            'shape': image_shape,
            'dtype': str(image_data.dtype),
            'affine': image_affine.tolist(),
            'has_nan': bool(np.isnan(image_data).any()),
            'has_inf': bool(np.isinf(image_data).any()),
            'value_range': (float(np.min(image_data)), float(np.max(image_data)))
        }
        
        # 检查掩码文件
        if not Path(mask_path).exists():
            validation_result['error'] = f"Mask file not found: {mask_path}"
            return validation_result
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_shape = mask_data.shape
        
        validation_result['mask_info'] = {
            'shape': mask_shape,
            'dtype': str(mask_data.dtype),
            'unique_values': list(np.unique(mask_data)),
            'roi_voxels': int(np.sum(mask_data > 0)),
            'roi_percentage': float(np.sum(mask_data > 0) / mask_data.size * 100)
        }
        
        # 检查兼容性
        shape_match = image_shape == mask_shape
        spacing_match = np.allclose(img.header.get_zooms()[:3], 
                                  mask_img.header.get_zooms()[:3], rtol=0.01)
        
        validation_result['compatibility'] = {
            'shape_match': shape_match,
            'spacing_match': spacing_match,
            'compatible': shape_match and spacing_match
        }
        
        if not validation_result['compatibility']['compatible']:
            validation_result['error'] = "Image and mask are not compatible"
        else:
            validation_result['valid'] = True
        
    except Exception as e:
        validation_result['error'] = f"Validation error: {str(e)}"
    
    return validation_result


def calculate_feature_importance(features_dict_list: List[Dict[str, Any]], 
                             method: str = 'variance') -> Dict[str, float]:
    """计算特征重要性"""
    if not features_dict_list:
        return {}
    
    # 合并所有特征
    all_features = []
    feature_names = None
    
    for features_dict in features_dict_list:
        if 'features' not in features_dict:
            continue
        
        features = features_dict['features']
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        # 获取第一个特征来获取样本数量
        first_feature = list(features.values())[0]
        n_samples = len(first_feature)
        
        for i in range(n_samples):
            feature_vector = []
            for feature_name in feature_names:
                feature_values = features[feature_name]
                if i < len(feature_values) and np.isfinite(feature_values[i]):
                    feature_vector.append(feature_values[i])
                else:
                    feature_vector = None
                    break
            
            if feature_vector:
                all_features.append(feature_vector)
    
    if not all_features:
        return {}
    
    all_features = np.array(all_features)
    
    if method == 'variance':
        # 基于方差的重要性
        variances = np.var(all_features, axis=0)
        importance = {name: float(var) for name, var in zip(feature_names, variances)}
        
    elif method == 'correlation':
        # 基于相关性的重要性（低相关性 = 高重要性）
        correlation_matrix = np.corrcoef(all_features.T)
        # 计算每个特征与其他特征的平均相关性
        avg_correlations = np.mean(np.abs(correlation_matrix), axis=1)
        importance = {name: float(1.0 / (corr + 1e-8)) 
                    for name, corr in zip(feature_names, avg_correlations)}
        
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    # 归一化到 [0, 1]
    max_importance = max(importance.values()) if importance.values() else 1.0
    importance = {name: val / max_importance for name, val in importance.items()}
    
    return importance


def create_habitat_summary(results: List[Dict[str, Any]], 
                        output_path: Optional[str] = None) -> Dict[str, Any]:
    """创建生态分析摘要"""
    summary = {
        'total_images': len(results),
        'successful_processed': 0,
        'failed_processed': 0,
        'total_voxels_processed': 0,
        'total_clusters_found': 0,
        'average_clusters_per_image': 0,
        'processing_statistics': {},
        'error_summary': {}
    }
    
    all_cluster_counts = []
    all_voxel_counts = []
    
    for result in results:
        if 'error' in result:
            summary['failed_processed'] += 1
            
            error_msg = result['error']
            if error_msg not in summary['error_summary']:
                summary['error_summary'][error_msg] = 0
            summary['error_summary'][error_msg] += 1
        else:
            summary['successful_processed'] += 1
            
            if 'n_clusters' in result:
                summary['total_clusters_found'] += result['n_clusters']
                all_cluster_counts.append(result['n_clusters'])
            
            if 'total_voxels' in result:
                summary['total_voxels_processed'] += result['total_voxels']
                all_voxel_counts.append(result['total_voxels'])
    
    # 计算统计信息
    if all_cluster_counts:
        summary['average_clusters_per_image'] = np.mean(all_cluster_counts)
        summary['cluster_distribution'] = {
            'min': int(np.min(all_cluster_counts)),
            'max': int(np.max(all_cluster_counts)),
            'mean': float(np.mean(all_cluster_counts)),
            'std': float(np.std(all_cluster_counts))
        }
    
    if all_voxel_counts:
        summary['voxel_distribution'] = {
            'min': int(np.min(all_voxel_counts)),
            'max': int(np.max(all_voxel_counts)),
            'mean': float(np.mean(all_voxel_counts)),
            'std': float(np.std(all_voxel_counts)),
            'total': int(np.sum(all_voxel_counts))
        }
    
    # 处理成功率
    if summary['total_images'] > 0:
        summary['success_rate'] = summary['successful_processed'] / summary['total_images'] * 100
    
    if output_path:
        save_json(summary, output_path)
    
    return summary


def optimize_batch_processing(file_list: List[str], 
                         n_workers: int = 4,
                         memory_limit_gb: float = 8.0) -> List[List[str]]:
    """优化批处理分配"""
    if not file_list:
        return []
    
    # 按文件大小排序（大文件优先处理）
    file_sizes = []
    for file_path in file_list:
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            file_sizes.append((file_path, size_mb))
        except:
            file_sizes.append((file_path, 0))
    
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # 简单的负载均衡分配
    batches = [[] for _ in range(n_workers)]
    
    for i, (file_path, size_mb) in enumerate(file_sizes):
        # 轮询分配到最小负载的工作器
        current_loads = [sum(f[1] for f in batch) for batch in batches]
        min_load_idx = np.argmin(current_loads)
        batches[min_load_idx].append((file_path, size_mb))
    
    # 转换回文件路径列表
    result_batches = [[file_path for file_path, _ in batch] for batch in batches]
    
    # 记录分配信息
    for i, batch in enumerate(result_batches):
        total_size = sum(Path(f).stat().st_size for f in batch) / (1024 * 1024 * 1024)
        print(f"Worker {i}: {len(batch)} files, {total_size:.2f} GB")
    
    return result_batches


def validate_clustering_parameters(n_clusters: int, 
                              n_samples: int,
                              method: str = 'kmeans') -> Dict[str, Any]:
    """验证聚类参数"""
    validation = {
        'valid': True,
        'warnings': [],
        'recommendations': []
    }
    
    # 检查聚类数量
    if n_clusters <= 1:
        validation['valid'] = False
        validation['warnings'].append("Number of clusters must be > 1")
    
    if n_clusters >= n_samples:
        validation['valid'] = False
        validation['warnings'].append("Number of clusters must be < number of samples")
    
    # 方法特定的验证
    if method == 'kmeans':
        if n_clusters > 20:
            validation['warnings'].append("Large number of clusters may be computationally expensive")
            validation['recommendations'].append("Consider using hierarchical clustering for many clusters")
    
    elif method == 'dbscan':
        if n_samples < 10:
            validation['warnings'].append("DBSCAN requires more samples for reliable clustering")
            validation['recommendations'].append("Consider increasing sample size or using k-means")
    
    elif method == 'hierarchical':
        if n_samples > 1000:
            validation['warnings'].append("Hierarchical clustering is computationally expensive for large datasets")
            validation['recommendations'].append("Consider using k-means for large datasets")
    
    return validation


def create_processing_log(log_file: str, 
                        operation: str,
                        parameters: Dict[str, Any],
                        start_time: str,
                        end_time: Optional[str] = None,
                        status: str = "started",
                        details: Optional[Dict[str, Any]] = None) -> None:
    """创建处理日志"""
    log_entry = {
        'timestamp': start_time,
        'operation': operation,
        'parameters': parameters,
        'status': status,
        'details': details or {}
    }
    
    if end_time:
        log_entry['end_time'] = end_time
        log_entry['duration'] = end_time - start_time
    
    # 读取现有日志
    log_entries = []
    if Path(log_file).exists():
        try:
            log_entries = load_json(log_file)
        except:
            log_entries = []
    
    # 添加新条目
    log_entries.append(log_entry)
    
    # 保存日志
    save_json(log_entries, log_file)


def generate_feature_report(features_dict: Dict[str, Any], 
                        output_path: str) -> None:
    """生成特征报告"""
    report = {
        'summary': {},
        'feature_statistics': {},
        'quality_metrics': {}
    }
    
    if 'features' not in features_dict:
        save_json(report, output_path)
        return
    
    features = features_dict['features']
    coordinates = features_dict.get('coordinates', [])
    metadata = features_dict.get('metadata', {})
    
    # 基本统计
    report['summary'] = {
        'total_voxels': len(coordinates),
        'feature_types': len(features),
        'feature_names': list(features.keys()),
        'image_shape': metadata.get('image_shape', 'Unknown'),
        'kernel_size': metadata.get('kernel_size', 'Unknown')
    }
    
    # 每个特征的统计
    for feature_name, feature_values in features.items():
        if len(feature_values) > 0:
            feature_array = np.array(feature_values)
            valid_mask = np.isfinite(feature_array)
            valid_values = feature_array[valid_mask]
            
            if len(valid_values) > 0:
                report['feature_statistics'][feature_name] = {
                    'count': len(valid_values),
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'q25': float(np.percentile(valid_values, 25)),
                    'median': float(np.median(valid_values)),
                    'q75': float(np.percentile(valid_values, 75)),
                    'missing_count': len(feature_array) - len(valid_values),
                    'missing_percentage': float((len(feature_array) - len(valid_values)) / len(feature_array) * 100)
                }
            else:
                report['feature_statistics'][feature_name] = {
                    'count': 0,
                    'missing_count': len(feature_array),
                    'missing_percentage': 100.0
                }
    
    # 质量指标
    total_features = len(features)
    missing_rates = [stats['missing_percentage'] for stats in report['feature_statistics'].values()]
    
    report['quality_metrics'] = {
        'average_missing_percentage': np.mean(missing_rates) if missing_rates else 0,
        'features_with_high_missing': len([rate for rate in missing_rates if rate > 50]),
        'features_with_low_variance': 0  # 需要计算
    }
    
    # 计算低方差特征
    for feature_name, feature_values in features.items():
        if len(feature_values) > 1:
            valid_values = feature_values[np.isfinite(feature_values)]
            if len(valid_values) > 1:
                variance = np.var(valid_values)
                mean_val = np.mean(valid_values)
                cv = np.sqrt(variance) / abs(mean_val) if mean_val != 0 else float('inf')
                
                if cv < 0.1:  # 低变异系数
                    report['quality_metrics']['features_with_low_variance'] += 1
    
    save_json(report, output_path)