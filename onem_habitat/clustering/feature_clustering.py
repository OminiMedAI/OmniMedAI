"""
特征聚类分析模块
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS = "Install with: pip install scikit-learn matplotlib seaborn"

from ..utils import habitat_utils


class FeatureClustering:
    """特征聚类分析器"""
    
    def __init__(self,
                 clustering_method: str = 'kmeans',
                 n_clusters: int = 5,
                 feature_selection: str = 'all',
                 pca_components: Optional[int] = None,
                 standardize: bool = True,
                 random_state: int = 42):
        """
        初始化特征聚类器
        
        Args:
            clustering_method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: 聚类数量（对于 DBSCAN 会自动确定）
            feature_selection: 特征选择方法 ('all', 'variance', 'correlation')
            pca_components: PCA 降维组件数，None 表示不降维
            standardize: 是否标准化特征
            random_state: 随机种子
        """
        if not HAS_DEPS:
            raise ImportError(f"Missing dependencies: {MISSING_DEPS}")
        
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.feature_selection = feature_selection
        self.pca_components = pca_components
        self.standardize = standardize
        self.random_state = random_state
        
        self.logger = self._setup_logger()
        
        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=pca_components, random_state=random_state) if pca_components else None
        self.clusterer = None
        self.feature_names = []
        self.selected_feature_indices = []
    
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
    
    def fit_predict(self, features_dict_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        对特征列表进行聚类预测
        
        Args:
            features_dict_list: 特征字典列表，每个元素包含多个图像的特征
            
        Returns:
            每个图像的聚类标签列表
        """
        self.logger.info(f"Starting clustering with method: {self.clustering_method}")
        
        # 合并所有特征
        all_features, feature_info = self._prepare_features(features_dict_list)
        
        if all_features is None or len(all_features) == 0:
            self.logger.error("No valid features found for clustering")
            return []
        
        self.logger.info(f"Total samples: {len(all_features)}, Features per sample: {len(all_features[0])}")
        
        # 特征选择
        selected_features = self._select_features(all_features, feature_info)
        
        # 数据预处理
        processed_features = self._preprocess_features(selected_features)
        
        # 聚类
        cluster_labels = self._perform_clustering(processed_features)
        
        # 评估聚类质量
        self._evaluate_clustering(processed_features, cluster_labels)
        
        # 分割标签到各个图像
        return self._split_labels_by_images(cluster_labels, features_dict_list)
    
    def _prepare_features(self, features_dict_list: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """准备特征数据"""
        all_features = []
        feature_info = []
        
        for img_idx, features_dict in enumerate(features_dict_list):
            if 'features' not in features_dict:
                self.logger.warning(f"No features found in image {img_idx}")
                continue
            
            features = features_dict['features']
            coordinates = features_dict.get('coordinates', [])
            
            if not features:
                continue
            
            # 获取特征名称（第一次）
            if not self.feature_names:
                self.feature_names = list(features.keys())
                self.logger.info(f"Found {len(self.feature_names)} feature types")
            
            # 构建特征矩阵
            n_voxels = len(coordinates) if coordinates else len(list(features.values())[0])
            
            for voxel_idx in range(n_voxels):
                feature_vector = []
                valid_features = True
                
                for feature_name in self.feature_names:
                    feature_values = features[feature_name]
                    
                    if voxel_idx < len(feature_values) and np.isfinite(feature_values[voxel_idx]):
                        feature_vector.append(feature_values[voxel_idx])
                    else:
                        valid_features = False
                        break
                
                if valid_features:
                    all_features.append(feature_vector)
                    feature_info.append({
                        'image_idx': img_idx,
                        'voxel_idx': voxel_idx,
                        'coordinates': coordinates[voxel_idx] if voxel_idx < len(coordinates) else None
                    })
        
        return np.array(all_features) if all_features else None, feature_info
    
    def _select_features(self, features: np.ndarray, feature_info: List[Dict]) -> np.ndarray:
        """特征选择"""
        if self.feature_selection == 'all':
            return features
        
        self.logger.info(f"Performing feature selection: {self.feature_selection}")
        
        if self.feature_selection == 'variance':
            # 基于方差的特征选择
            variances = np.var(features, axis=0)
            threshold = np.percentile(variances, 50)  # 选择方差前50%的特征
            selected_indices = np.where(variances > threshold)[0]
            
        elif self.feature_selection == 'correlation':
            # 移除高度相关的特征
            correlation_matrix = np.corrcoef(features.T)
            
            # 找到高相关特征对
            high_corr_pairs = np.where(np.abs(correlation_matrix) > 0.9)
            high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i < j]
            
            # 移除一个特征对中的第二个特征
            to_remove = set()
            for i, j in high_corr_pairs:
                to_remove.add(j)
            
            selected_indices = [i for i in range(features.shape[1]) if i not in to_remove]
        else:
            selected_indices = list(range(features.shape[1]))
        
        self.selected_feature_indices = selected_indices
        self.logger.info(f"Selected {len(selected_indices)} features out of {features.shape[1]}")
        
        return features[:, selected_indices]
    
    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """特征预处理"""
        processed_features = features.copy()
        
        # 标准化
        if self.standardize:
            processed_features = self.scaler.fit_transform(processed_features)
            self.logger.info("Features standardized")
        
        # PCA 降维
        if self.pca_components:
            processed_features = self.pca.fit_transform(processed_features)
            self.logger.info(f"PCA applied: {features.shape[1]} -> {self.pca_components} dimensions")
            
            if self.pca_components >= 3:
                explained_variance = np.sum(self.pca.explained_variance_ratio_)
                self.logger.info(f"Explained variance: {explained_variance:.3f}")
        
        return processed_features
    
    def _perform_clustering(self, features: np.ndarray) -> np.ndarray:
        """执行聚类"""
        if self.clustering_method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            
        elif self.clustering_method == 'dbscan':
            # 自动确定 eps 参数
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(features)-1)).fit(features)
            distances, _ = nbrs.kneighbors(features)
            eps = np.percentile(distances[:, -1], 90)
            
            self.clusterer = DBSCAN(eps=eps, min_samples=max(5, len(features)//100))
            
        elif self.clustering_method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # 执行聚类
        cluster_labels = self.clusterer.fit_predict(features)
        
        # 统计聚类结果
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters_found = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
        
        self.logger.info(f"Clustering completed. Found {n_clusters_found} clusters")
        for label, count in zip(unique_labels, counts):
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            self.logger.info(f"  {cluster_name}: {count} points")
        
        return cluster_labels
    
    def _evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """评估聚类质量"""
        metrics = {}
        
        # 过滤噪声点
        valid_mask = labels != -1
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        
        n_clusters = len(np.unique(valid_labels))
        
        if n_clusters > 1 and len(valid_features) > n_clusters:
            try:
                # 轮廓系数
                silhouette = silhouette_score(valid_features, valid_labels)
                metrics['silhouette_score'] = silhouette
                
                # Calinski-Harabasz 指数
                ch_score = calinski_harabasz_score(valid_features, valid_labels)
                metrics['calinski_harabasz_score'] = ch_score
                
                self.logger.info(f"Clustering metrics:")
                self.logger.info(f"  Silhouette Score: {silhouette:.3f}")
                self.logger.info(f"  Calinski-Harabasz Score: {ch_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to compute clustering metrics: {e}")
        
        return metrics
    
    def _split_labels_by_images(self, all_labels: np.ndarray, features_dict_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """将聚类标签按图像分割"""
        image_labels_list = []
        current_idx = 0
        
        for img_idx, features_dict in enumerate(features_dict_list):
            if 'features' not in features_dict:
                image_labels_list.append(np.array([]))
                continue
            
            n_voxels = len(features_dict.get('coordinates', []))
            if not n_voxels:
                n_voxels = len(list(features_dict['features'].values())[0])
            
            # 提取对应图像的标签
            end_idx = min(current_idx + n_voxels, len(all_labels))
            image_labels = all_labels[current_idx:end_idx]
            
            image_labels_list.append(image_labels)
            current_idx = end_idx
        
        return image_labels_list
    
    def visualize_clusters(self, 
                        features: np.ndarray,
                        labels: np.ndarray,
                        output_path: Optional[str] = None,
                        method: str = 'tsne') -> None:
        """
        可视化聚类结果
        
        Args:
            features: 特征矩阵
            labels: 聚类标签
            output_path: 输出路径
            method: 降维方法 ('pca', 'tsne')
        """
        if len(features) > 10000:
            # 大数据集采样
            sample_indices = np.random.choice(len(features), min(5000, len(features)), replace=False)
            features_sample = features[sample_indices]
            labels_sample = labels[sample_indices]
        else:
            features_sample = features
            labels_sample = labels
        
        # 降维到2D
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(features_sample)//4))
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        features_2d = reducer.fit_transform(features_sample)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels_sample)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_sample == label
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=cluster_name, alpha=0.6)
        
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.title(f'Clustering Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_cluster_characteristics(self, 
                                   features_dict_list: List[Dict[str, Any]],
                                   cluster_labels_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        分析各聚类特征
        
        Args:
            features_dict_list: 特征字典列表
            cluster_labels_list: 聚类标签列表
            
        Returns:
            聚类特征分析结果
        """
        analysis = {}
        
        for img_idx, (features_dict, cluster_labels) in enumerate(zip(features_dict_list, cluster_labels_list)):
            if len(cluster_labels) == 0:
                continue
            
            features = features_dict['features']
            coordinates = features_dict.get('coordinates', [])
            
            # 统计每个聚类
            unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
            
            cluster_analysis = {}
            for label, count in zip(unique_labels, label_counts):
                cluster_name = "Noise" if label == -1 else f"Cluster_{label}"
                mask = cluster_labels == label
                
                # 计算该聚类的特征统计
                cluster_stats = {}
                for feature_name in self.feature_names:
                    feature_values = features[feature_name]
                    cluster_feature_values = feature_values[mask]
                    
                    if len(cluster_feature_values) > 0:
                        cluster_stats[feature_name] = {
                            'mean': float(np.mean(cluster_feature_values)),
                            'std': float(np.std(cluster_feature_values)),
                            'min': float(np.min(cluster_feature_values)),
                            'max': float(np.max(cluster_feature_values))
                        }
                
                cluster_analysis[cluster_name] = {
                    'count': int(count),
                    'percentage': float(count / len(cluster_labels) * 100),
                    'coordinates': [coordinates[i] for i, keep in enumerate(mask) if keep 
                                  and i < len(coordinates)],
                    'feature_stats': cluster_stats
                }
            
            analysis[f'image_{img_idx}'] = {
                'total_voxels': len(cluster_labels),
                'n_clusters': len([l for l in unique_labels if l != -1]),
                'cluster_analysis': cluster_analysis
            }
        
        return analysis
    
    def save_clustering_results(self, 
                             features_dict_list: List[Dict[str, Any]],
                             cluster_labels_list: List[np.ndarray],
                             output_dir: str,
                             prefix: str = "clustering") -> Dict[str, str]:
        """
        保存聚类结果
        
        Args:
            features_dict_list: 特征字典列表
            cluster_labels_list: 聚类标签列表
            output_dir: 输出目录
            prefix: 文件前缀
            
        Returns:
            保存的文件路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存聚类标签
        for img_idx, cluster_labels in enumerate(cluster_labels_list):
            if len(cluster_labels) > 0:
                labels_file = output_dir / f"{prefix}_labels_{img_idx}.npy"
                np.save(labels_file, cluster_labels)
                saved_files[f'labels_{img_idx}'] = str(labels_file)
        
        # 保存聚类分析结果
        analysis = self.analyze_cluster_characteristics(features_dict_list, cluster_labels_list)
        analysis_file = output_dir / f"{prefix}_analysis.json"
        habitat_utils.save_json(analysis, analysis_file)
        saved_files['analysis'] = str(analysis_file)
        
        # 保存聚类配置
        config = {
            'clustering_method': self.clustering_method,
            'n_clusters': self.n_clusters,
            'feature_selection': self.feature_selection,
            'pca_components': self.pca_components,
            'standardize': self.standardize,
            'selected_feature_indices': self.selected_feature_indices.tolist(),
            'feature_names': self.feature_names
        }
        config_file = output_dir / f"{prefix}_config.json"
        habitat_utils.save_json(config, config_file)
        saved_files['config'] = str(config_file)
        
        self.logger.info(f"Clustering results saved to {output_dir}")
        
        return saved_files
    
    def load_clustering_results(self, output_dir: str, prefix: str = "clustering") -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        加载聚类结果
        
        Args:
            output_dir: 输出目录
            prefix: 文件前缀
            
        Returns:
            (聚类标签列表, 分析结果)
        """
        output_dir = Path(output_dir)
        
        # 加载分析结果
        analysis_file = output_dir / f"{prefix}_analysis.json"
        if analysis_file.exists():
            analysis = habitat_utils.load_json(analysis_file)
        else:
            analysis = {}
        
        # 加载聚类标签
        labels_files = list(output_dir.glob(f"{prefix}_labels_*.npy"))
        labels_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        cluster_labels_list = []
        for labels_file in labels_files:
            labels = np.load(labels_file)
            cluster_labels_list.append(labels)
        
        return cluster_labels_list, analysis