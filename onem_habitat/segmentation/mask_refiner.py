"""
基于聚类的 Mask 重新划分模块
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import ndimage
from skimage import morphology, measure
import warnings

try:
    import nibabel as nib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS = "Install with: pip install nibabel matplotlib scikit-image"

from ..utils import habitat_utils


class MaskRefiner:
    """基于聚类的 Mask 重新划分器"""
    
    def __init__(self,
                 min_cluster_size: int = 50,
                 smoothing_iterations: int = 2,
                 connectivity: int = 1,
                 fill_holes: bool = True,
                 noise_cluster_id: int = -1):
        """
        初始化 Mask 重新划分器
        
        Args:
            min_cluster_size: 最小聚类大小
            smoothing_iterations: 平滑迭代次数
            connectivity: 连通性 (1=6邻域, 2=26邻域)
            fill_holes: 是否填充孔洞
            noise_cluster_id: 噪声聚类标签
        """
        if not HAS_DEPS:
            raise ImportError(f"Missing dependencies: {MISSING_DEPS}")
        
        self.min_cluster_size = min_cluster_size
        self.smoothing_iterations = smoothing_iterations
        self.connectivity = connectivity
        self.fill_holes = fill_holes
        self.noise_cluster_id = noise_cluster_id
        
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
    
    def refine_masks(self,
                   image_path: str,
                   mask_path: str,
                   cluster_labels: np.ndarray,
                   coordinates: List[Tuple[int, int, int]],
                   output_dir: str,
                   save_individual: bool = True,
                   save_combined: bool = True) -> Dict[str, str]:
        """
        基于聚类标签重新划分 Mask
        
        Args:
            image_path: 原始图像路径
            mask_path: 原始掩码路径
            cluster_labels: 聚类标签数组
            coordinates: 对应的坐标列表
            output_dir: 输出目录
            save_individual: 是否保存单独的聚类掩码
            save_combined: 是否保存合并的掩码
            
        Returns:
            保存的文件路径字典
        """
        # 加载原始掩码
        original_mask_data, original_affine, original_header = self._load_nifti(mask_path)
        original_shape = original_mask_data.shape
        
        self.logger.info(f"Original mask shape: {original_shape}")
        self.logger.info(f"Cluster labels shape: {len(cluster_labels)}, coordinates: {len(coordinates)}")
        
        # 创建新的精细掩码
        refined_masks = self._create_refined_masks(
            original_mask_data, cluster_labels, coordinates, original_shape
        )
        
        # 后处理
        refined_masks = self._post_process_masks(refined_masks)
        
        # 保存结果
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存单独的聚类掩码
        if save_individual:
            for cluster_id, mask_data in refined_masks.items():
                if cluster_id == 'background':
                    continue
                
                cluster_mask_file = output_dir / f"cluster_{cluster_id}_mask.nii.gz"
                self._save_nifti(mask_data, original_affine, original_header, str(cluster_mask_file))
                saved_files[f'cluster_{cluster_id}'] = str(cluster_mask_file)
        
        # 保存合并的精细掩码
        if save_individual:
            combined_mask = self._create_combined_mask(refined_masks, original_shape)
            combined_mask_file = output_dir / "refined_combined_mask.nii.gz"
            self._save_nifti(combined_mask, original_affine, original_header, str(combined_mask_file))
            saved_files['combined'] = str(combined_mask_file)
        
        # 保存可视化
        self._visualize_refinement(original_mask_data, refined_masks, output_dir)
        
        # 统计信息
        self._log_refinement_statistics(refined_masks)
        
        return saved_files
    
    def _load_nifti(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """加载 NIfTI 文件"""
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img.affine, img.header
    
    def _save_nifti(self, data: np.ndarray, affine: np.ndarray, header: Any, file_path: str) -> None:
        """保存 NIfTI 文件"""
        img = nib.Nifti1Image(data.astype(np.int16), affine, header)
        nib.save(img, file_path)
    
    def _create_refined_masks(self,
                             original_mask: np.ndarray,
                             cluster_labels: np.ndarray,
                             coordinates: List[Tuple[int, int, int]],
                             shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        """创建精细化的掩码"""
        # 初始化精细化掩码
        refined_masks = {
            'background': original_mask.copy() * 0,  # 背景为0
        }
        
        # 为每个聚类创建掩码
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label == self.noise_cluster_id:
                # 噪声点处理 - 分配到背景或最近的有效聚类
                refined_masks['noise'] = np.zeros_like(original_mask, dtype=np.int16)
                continue
            
            cluster_name = f"cluster_{label}"
            refined_masks[cluster_name] = np.zeros_like(original_mask, dtype=np.int16)
        
        # 将聚类标签映射到空间位置
        for (z, y, x), label in zip(coordinates, cluster_labels):
            if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
                continue
            
            if label == self.noise_cluster_id:
                refined_masks['noise'][z, y, x] = 1
            else:
                cluster_name = f"cluster_{label}"
                refined_masks[cluster_name][z, y, x] = 1
        
        # 处理噪声点
        if 'noise' in refined_masks:
            self._assign_noise_points(refined_masks, original_mask)
        
        # 从背景中移除精细化的区域
        for cluster_name, cluster_mask in refined_masks.items():
            if cluster_name != 'background':
                refined_masks['background'] = np.logical_and(
                    refined_masks['background'] == 0,
                    cluster_mask == 0
                ).astype(np.int16)
        
        return refined_masks
    
    def _assign_noise_points(self, refined_masks: Dict[str, np.ndarray], original_mask: np.ndarray) -> None:
        """分配噪声点到最近的聚类"""
        noise_mask = refined_masks['noise']
        noise_coords = np.where(noise_mask > 0)
        
        if len(noise_coords[0]) == 0:
            return
        
        self.logger.info(f"Assigning {len(noise_coords[0])} noise points")
        
        # 为每个噪声点找到最近的聚类
        for i in range(len(noise_coords[0])):
            z, y, x = noise_coords[0][i], noise_coords[1][i], noise_coords[2][i]
            
            # 搜索周围区域
            search_radius = 3
            min_dist = float('inf')
            nearest_cluster = None
            
            for cluster_name, cluster_mask in refined_masks.items():
                if cluster_name in ['background', 'noise']:
                    continue
                
                # 查找聚类中的点
                cluster_coords = np.where(cluster_mask > 0)
                
                if len(cluster_coords[0]) == 0:
                    continue
                
                # 计算最小距离
                distances = np.sqrt(
                    (cluster_coords[0] - z)**2 + 
                    (cluster_coords[1] - y)**2 + 
                    (cluster_coords[2] - x)**2
                )
                
                min_cluster_dist = np.min(distances)
                if min_cluster_dist < min_dist:
                    min_dist = min_cluster_dist
                    nearest_cluster = cluster_name
            
            if nearest_cluster and min_dist <= search_radius:
                refined_masks[nearest_cluster][z, y, x] = 1
                refined_masks['noise'][z, y, x] = 0
        
        # 剩余噪声点分配到背景
        remaining_noise = refined_masks['noise'] > 0
        refined_masks['background'][remaining_noise] = 1
        refined_masks['noise'][remaining_noise] = 0
    
    def _post_process_masks(self, refined_masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """后处理掩码"""
        processed_masks = {}
        
        for mask_name, mask_data in refined_masks.items():
            if mask_name == 'background':
                processed_masks[mask_name] = mask_data.copy()
                continue
            
            processed_mask = mask_data.copy()
            
            # 移除小连通分量
            if self.min_cluster_size > 0:
                labeled_mask = measure.label(processed_mask, connectivity=self.connectivity)
                component_sizes = np.bincount(labeled_mask.ravel())
                
                for component_id in range(1, len(component_sizes)):
                    if component_sizes[component_id] < self.min_cluster_size:
                        processed_mask[labeled_mask == component_id] = 0
                
                self.logger.info(f"Removed small components from {mask_name}")
            
            # 形态学平滑
            if self.smoothing_iterations > 0:
                for _ in range(self.smoothing_iterations):
                    # 开运算（去除小的突出）
                    processed_mask = morphology.binary_opening(
                        processed_mask, morphology.ball(1)
                    )
                    # 闭运算（填充小孔洞）
                    processed_mask = morphology.binary_closing(
                        processed_mask, morphology.ball(1)
                    )
                
                processed_mask = processed_mask.astype(np.int16)
            
            # 填充孔洞
            if self.fill_holes:
                processed_mask = ndimage.binary_fill_holes(processed_mask)
                processed_mask = processed_mask.astype(np.int16)
            
            processed_masks[mask_name] = processed_mask
        
        return processed_masks
    
    def _create_combined_mask(self, refined_masks: Dict[str, np.ndarray], shape: Tuple[int, int, int]) -> np.ndarray:
        """创建合并的精细掩码"""
        combined_mask = np.zeros(shape, dtype=np.int16)
        cluster_id = 1
        
        for mask_name, mask_data in refined_masks.items():
            if mask_name == 'background':
                continue
            
            # 为每个聚类分配唯一的ID
            combined_mask[mask_data > 0] = cluster_id
            cluster_id += 1
        
        return combined_mask
    
    def _visualize_refinement(self, original_mask: np.ndarray, refined_masks: Dict[str, np.ndarray], output_dir: Path) -> None:
        """可视化精细化结果"""
        try:
            # 创建2D切片可视化
            z_slices = [original_mask.shape[0] // 4, original_mask.shape[0] // 2, original_mask.shape[0] * 3 // 4]
            
            fig, axes = plt.subplots(2, len(z_slices), figsize=(15, 10))
            fig.suptitle('Mask Refinement Visualization', fontsize=16)
            
            for i, z_slice in enumerate(z_slices):
                # 原始掩码
                axes[0, i].imshow(original_mask[z_slice], cmap='gray')
                axes[0, i].set_title(f'Original (z={z_slice})')
                axes[0, i].axis('off')
                
                # 精细化掩码
                combined_mask = self._create_combined_mask(refined_masks, original_mask.shape)
                axes[1, i].imshow(combined_mask[z_slice], cmap='tab20')
                axes[1, i].set_title(f'Refined (z={z_slice})')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'refinement_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # 创建聚类统计图
            self._plot_cluster_statistics(refined_masks, output_dir)
            
            self.logger.info(f"Visualization saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _plot_cluster_statistics(self, refined_masks: Dict[str, np.ndarray], output_dir: Path) -> None:
        """绘制聚类统计图"""
        cluster_sizes = {}
        for mask_name, mask_data in refined_masks.items():
            if mask_name != 'background':
                size = np.sum(mask_data > 0)
                cluster_sizes[mask_name] = size
        
        if not cluster_sizes:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        names = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        ax1.bar(names, sizes, color='skyblue')
        ax1.set_title('Cluster Sizes')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Voxels')
        ax1.tick_params(axis='x', rotation=45)
        
        # 饼图
        ax2.pie(sizes, labels=names, autopct='%1.1f%%')
        ax2.set_title('Cluster Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _log_refinement_statistics(self, refined_masks: Dict[str, np.ndarray]) -> None:
        """记录精细化统计信息"""
        self.logger.info("=== Mask Refinement Statistics ===")
        
        total_voxels = 0
        cluster_info = {}
        
        for mask_name, mask_data in refined_masks.items():
            size = np.sum(mask_data > 0)
            if mask_name != 'background' and size > 0:
                cluster_info[mask_name] = size
                total_voxels += size
        
        # 按大小排序
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1], reverse=True)
        
        for cluster_name, size in sorted_clusters:
            percentage = (size / total_voxels) * 100 if total_voxels > 0 else 0
            self.logger.info(f"  {cluster_name}: {size} voxels ({percentage:.1f}%)")
        
        self.logger.info(f"Total refined voxels: {total_voxels}")
    
    def batch_refine_masks(self,
                          images_dir: str,
                          masks_dir: str,
                          features_dir: str,
                          clustering_results_dir: str,
                          output_dir: str) -> List[Dict[str, Any]]:
        """
        批量精细化掩码
        
        Args:
            images_dir: 图像目录
            masks_dir: 掩码目录
            features_dir: 特征目录
            clustering_results_dir: 聚类结果目录
            output_dir: 输出目录
            
        Returns:
            处理结果列表
        """
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        features_dir = Path(features_dir)
        clustering_dir = Path(clustering_results_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # 查找特征文件
        feature_files = list(features_dir.glob("*_features.npy"))
        
        for feature_file in feature_files:
            try:
                # 加载特征
                features_dict = np.load(feature_file, allow_pickle=True).item()
                coordinates = features_dict.get('coordinates', [])
                
                # 加载聚类标签
                labels_file = clustering_dir / f"clustering_labels_{feature_file.stem.split('_features')[0]}.npy"
                if not labels_file.exists():
                    self.logger.warning(f"Clustering labels not found for {feature_file}")
                    continue
                
                cluster_labels = np.load(labels_file)
                
                # 查找对应的图像和掩码
                image_name = feature_file.stem.replace('_features', '')
                image_file = images_dir / f"{image_name}.nii.gz"
                mask_file = habitat_utils.find_matching_mask(image_file, masks_dir)
                
                if not image_file.exists() or not mask_file:
                    self.logger.warning(f"Image or mask not found for {feature_file}")
                    continue
                
                # 创建图像特定的输出目录
                image_output_dir = output_dir / image_name
                image_output_dir.mkdir(exist_ok=True)
                
                # 精细化掩码
                saved_files = self.refine_masks(
                    str(image_file),
                    str(mask_file),
                    cluster_labels,
                    coordinates,
                    str(image_output_dir)
                )
                
                results.append({
                    'image_name': image_name,
                    'image_file': str(image_file),
                    'mask_file': str(mask_file),
                    'feature_file': str(feature_file),
                    'labels_file': str(labels_file),
                    'output_dir': str(image_output_dir),
                    'saved_files': saved_files,
                    'n_clusters': len([k for k in saved_files.keys() if k.startswith('cluster_')]),
                    'total_voxels': len(coordinates)
                })
                
                self.logger.info(f"Completed refinement for {image_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to refine masks for {feature_file}: {e}")
                results.append({
                    'feature_file': str(feature_file),
                    'error': str(e)
                })
        
        return results
    
    def evaluate_refinement_quality(self,
                                  original_mask: np.ndarray,
                                  refined_masks: Dict[str, np.ndarray],
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        评估精细化质量
        
        Args:
            original_mask: 原始掩码
            refined_masks: 精细化掩码字典
            cluster_labels: 聚类标签
            
        Returns:
            质量评估指标
        """
        metrics = {}
        
        # 计算掩码覆盖度
        total_original = np.sum(original_mask > 0)
        total_refined = sum(np.sum(mask > 0) for mask in refined_masks.values() 
                          if 'background' not in mask)
        
        coverage = total_refined / total_original if total_original > 0 else 0
        metrics['coverage_ratio'] = coverage
        
        # 计算聚类平衡性
        cluster_sizes = [np.sum(mask > 0) for mask in refined_masks.values() 
                        if 'background' not in mask and np.sum(mask > 0) > 0]
        
        if cluster_sizes:
            size_balance = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes)) if np.mean(cluster_sizes) > 0 else 0
            metrics['size_balance'] = size_balance
        
        # 计算聚类连通性
        connectivity_scores = []
        for mask_name, mask_data in refined_masks.items():
            if 'background' in mask_name or np.sum(mask_data) == 0:
                continue
            
            labeled_mask = measure.label(mask_data, connectivity=self.connectivity)
            n_components = len(np.unique(labeled_mask)) - 1  # 减去背景
            
            if n_components > 0:
                connectivity_score = 1.0 / n_components
                connectivity_scores.append(connectivity_score)
        
        if connectivity_scores:
            metrics['mean_connectivity'] = np.mean(connectivity_scores)
        
        # 计算空间紧密度
        tightness_scores = []
        for mask_name, mask_data in refined_masks.items():
            if 'background' in mask_name or np.sum(mask_data) == 0:
                continue
            
            coords = np.where(mask_data > 0)
            if len(coords[0]) > 0:
                center = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
                distances = np.sqrt(sum((coord - center[i])**2 for i, coord in enumerate(coords)))
                tightness_scores.append(np.std(distances) if len(distances) > 0 else 0)
        
        if tightness_scores:
            metrics['mean_tightness'] = np.mean(tightness_scores)
        
        return metrics