"""
生态分析配置管理模块
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class HabitatConfig:
    """生态分析配置"""
    
    # 特征提取配置
    kernel_size: Tuple[int, int, int] = (5, 5, 5)
    feature_types: List[str] = None
    bin_width: int = 25
    resampled_pixel_spacing: Optional[Tuple[float, float, float]] = None
    interpolator: str = 'sitkBSpline'
    weight_center: Optional[Tuple[float, float, float]] = None
    weight_radius: Optional[float] = None
    extraction_n_jobs: int = 1
    step_size: int = 1
    
    # 聚类配置
    clustering_method: str = 'kmeans'
    n_clusters: int = 5
    feature_selection: str = 'all'
    pca_components: Optional[int] = None
    standardize: bool = True
    clustering_random_state: int = 42
    
    # 掩码精细化配置
    min_cluster_size: int = 50
    smoothing_iterations: int = 2
    connectivity: int = 1
    fill_holes: bool = True
    noise_cluster_id: int = -1
    
    # 输出配置
    output_format: str = 'npy'
    save_individual_masks: bool = True
    save_combined_masks: bool = True
    save_visualizations: bool = True
    create_reports: bool = True
    
    # 质量控制配置
    min_roi_size: int = 100
    max_missing_ratio: float = 0.3
    min_feature_variance: float = 0.01
    
    # 性能配置
    batch_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.feature_types is None:
            self.feature_types = [
                'firstorder', 'glcm', 'glrlm', 'glszm', 'gltdm', 'ngtdm'
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HabitatConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'HabitatConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def validate(self) -> Dict[str, Any]:
        """验证配置"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 验证核大小
        if any(size <= 0 or size % 2 == 0 for size in self.kernel_size):
            validation_result['errors'].append("Kernel size must be positive odd numbers")
            validation_result['valid'] = False
        
        # 验证聚类参数
        if self.n_clusters <= 1:
            validation_result['errors'].append("Number of clusters must be > 1")
            validation_result['valid'] = False
        
        if self.clustering_method not in ['kmeans', 'dbscan', 'hierarchical']:
            validation_result['errors'].append(f"Unknown clustering method: {self.clustering_method}")
            validation_result['valid'] = False
        
        # 验证特征类型
        valid_feature_types = [
            'firstorder', 'glcm', 'glrlm', 'glszm', 'gltdm', 'ngtdm',
            'shape', 'shape2D'
        ]
        
        for feature_type in self.feature_types:
            if feature_type not in valid_feature_types:
                validation_result['warnings'].append(f"Unknown feature type: {feature_type}")
        
        # 验证性能参数
        if self.max_workers <= 0:
            validation_result['warnings'].append("Max workers should be > 0")
            self.max_workers = max(1, self.max_workers)
        
        if self.memory_limit_gb <= 0:
            validation_result['warnings'].append("Memory limit should be > 0")
            self.memory_limit_gb = max(1.0, self.memory_limit_gb)
        
        # 验证质量控制参数
        if self.min_roi_size <= 0:
            validation_result['warnings'].append("Min ROI size should be > 0")
            self.min_roi_size = max(1, self.min_roi_size)
        
        if not 0 <= self.max_missing_ratio <= 1:
            validation_result['errors'].append("Max missing ratio must be between 0 and 1")
            validation_result['valid'] = False
        
        return validation_result


class HabitatConfigManager:
    """生态分析配置管理器"""
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / 'habitat_config.json'
        self.presets_dir = self.config_dir / 'presets'
        self.presets_dir.mkdir(exist_ok=True)
    
    def get_default_config(self) -> HabitatConfig:
        """获取默认配置"""
        if self.config_file.exists():
            try:
                return HabitatConfig.load(self.config_file)
            except Exception:
                pass
        
        # 创建默认配置
        config = HabitatConfig()
        config.save(self.config_file)
        return config
    
    def save_config(self, config: HabitatConfig, name: Optional[str] = None) -> str:
        """
        保存配置
        
        Args:
            config: 配置对象
            name: 配置名称，如果为None则保存为默认配置
            
        Returns:
            保存的文件路径
        """
        if name is None:
            config_path = self.config_file
        else:
            config_path = self.presets_dir / f"{name}_config.json"
        
        config.save(config_path)
        return str(config_path)
    
    def load_config(self, name: Optional[str] = None) -> HabitatConfig:
        """
        加载配置
        
        Args:
            name: 配置名称，如果为None则加载默认配置
            
        Returns:
            配置对象
        """
        if name is None:
            if self.config_file.exists():
                return HabitatConfig.load(self.config_file)
            else:
                return HabitatConfig()
        else:
            config_path = self.presets_dir / f"{name}_config.json"
            if config_path.exists():
                return HabitatConfig.load(config_path)
            else:
                raise FileNotFoundError(f"Config preset not found: {name}")
    
    def list_presets(self) -> List[str]:
        """列出所有预设配置"""
        presets = []
        for preset_file in self.presets_dir.glob("*_config.json"):
            preset_name = preset_file.stem.replace('_config', '')
            presets.append(preset_name)
        
        return sorted(presets)
    
    def delete_preset(self, name: str) -> bool:
        """
        删除预设配置
        
        Args:
            name: 配置名称
            
        Returns:
            是否删除成功
        """
        preset_path = self.presets_dir / f"{name}_config.json"
        
        if preset_path.exists():
            preset_path.unlink()
            return True
        
        return False
    
    def export_presets(self, export_dir: Union[str, Path]) -> None:
        """
        导出所有预设配置
        
        Args:
            export_dir: 导出目录
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出默认配置
        if self.config_file.exists():
            default_export = export_dir / "default_config.json"
            default_export.write_text(self.config_file.read_text(encoding='utf-8'))
        
        # 导出所有预设
        for preset_file in self.presets_dir.glob("*_config.json"):
            export_path = export_dir / preset_file.name
            export_path.write_text(preset_file.read_text(encoding='utf-8'))
    
    def import_presets(self, import_dir: Union[str, Path]) -> int:
        """
        导入预设配置
        
        Args:
            import_dir: 导入目录
            
        Returns:
            导入的配置数量
        """
        import_dir = Path(import_dir)
        if not import_dir.exists():
            return 0
        
        imported_count = 0
        
        for config_file in import_dir.glob("*_config.json"):
            if config_file.stem == "default_config":
                # 导入默认配置
                default_path = self.config_dir / config_file.name
                config_file.write_text(default_path.read_text(encoding='utf-8'))
            else:
                # 导入预设配置
                preset_path = self.presets_dir / config_file.name
                config_file.write_text(preset_path.read_text(encoding='utf-8'))
            
            imported_count += 1
        
        return imported_count


# 预设配置模板
CT_LUNG_CONFIG = HabitatConfig(
    kernel_size=(7, 7, 7),
    feature_types=['firstorder', 'glcm', 'glrlm', 'glszm'],
    bin_width=25,
    clustering_method='kmeans',
    n_clusters=4,
    feature_selection='variance',
    standardize=True,
    min_cluster_size=100,
    smoothing_iterations=2,
    connectivity=1,
    fill_holes=True
)

MRI_BRAIN_CONFIG = HabitatConfig(
    kernel_size=(5, 5, 5),
    feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'shape'],
    bin_width=16,
    clustering_method='hierarchical',
    n_clusters=3,
    feature_selection='correlation',
    pca_components=10,
    standardize=True,
    min_cluster_size=50,
    smoothing_iterations=1,
    connectivity=2,
    fill_holes=True
)

PET_TUMOR_CONFIG = HabitatConfig(
    kernel_size=(3, 3, 3),
    feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gltdm'],
    bin_width=32,
    clustering_method='dbscan',
    feature_selection='all',
    standardize=False,
    min_cluster_size=30,
    smoothing_iterations=3,
    connectivity=1,
    fill_holes=True,
    weight_center=(0, 0, 0),
    weight_radius=50
)

HIGH_RESOLUTION_CONFIG = HabitatConfig(
    kernel_size=(9, 9, 9),
    feature_types=['firstorder', 'glcm', 'glrlm', 'glszm', 'gltdm', 'ngtdm', 'shape'],
    bin_width=64,
    clustering_method='kmeans',
    n_clusters=6,
    feature_selection='all',
    pca_components=15,
    standardize=True,
    min_cluster_size=200,
    smoothing_iterations=2,
    connectivity=2,
    fill_holes=True,
    extraction_n_jobs=4,
    max_workers=8,
    memory_limit_gb=16.0
)

FAST_PROCESSING_CONFIG = HabitatConfig(
    kernel_size=(3, 3, 3),
    feature_types=['firstorder', 'glcm'],
    bin_width=16,
    clustering_method='kmeans',
    n_clusters=3,
    feature_selection='variance',
    standardize=True,
    min_cluster_size=20,
    smoothing_iterations=1,
    connectivity=1,
    fill_holes=False,
    step_size=2,
    extraction_n_jobs=2,
    max_workers=2,
    memory_limit_gb=4.0,
    save_visualizations=False
)


def create_config_preset(manager: HabitatConfigManager, 
                      name: str,
                      description: str,
                      config: HabitatConfig) -> None:
    """
    创建配置预设
    
    Args:
        manager: 配置管理器
        name: 预设名称
        description: 描述
        config: 配置对象
    """
    # 保存配置
    config_path = manager.save_config(config, name)
    
    # 添加描述信息
    description_file = manager.presets_dir / f"{name}_description.txt"
    description_file.write_text(description, encoding='utf-8')
    
    print(f"Created preset '{name}' at {config_path}")


def load_config_with_validation(manager: HabitatConfigManager,
                           name: Optional[str] = None) -> Tuple[HabitatConfig, Dict[str, Any]]:
    """
    加载配置并验证
    
    Args:
        manager: 配置管理器
        name: 配置名称
        
    Returns:
        (配置对象, 验证结果)
    """
    config = manager.load_config(name)
    validation = config.validate()
    
    if not validation['valid']:
        print(f"⚠️  Configuration validation failed:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['warnings']:
        print(f"⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    return config, validation


# 预设配置字典
PRESET_CONFIGS = {
    'ct_lung': CT_LUNG_CONFIG,
    'mri_brain': MRI_BRAIN_CONFIG,
    'pet_tumor': PET_TUMOR_CONFIG,
    'high_resolution': HIGH_RESOLUTION_CONFIG,
    'fast_processing': FAST_PROCESSING_CONFIG
}

PRESET_DESCRIPTIONS = {
    'ct_lung': 'CT lung cancer habitat analysis with balanced feature extraction and moderate clustering',
    'mri_brain': 'MRI brain tumor habitat analysis with hierarchical clustering and comprehensive features',
    'pet_tumor': 'PET tumor habitat analysis with intensity-weighted features and density-based clustering',
    'high_resolution': 'High-resolution analysis with comprehensive features and advanced processing',
    'fast_processing': 'Fast processing configuration with minimal features and reduced computations'
}