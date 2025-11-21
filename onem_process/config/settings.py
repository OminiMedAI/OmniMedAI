"""
配置管理模块
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ProcessingConfig:
    """图像处理配置"""
    # 预处理配置
    normalize_method: str = 'z_score'  # 'z_score', 'min_max', 'percentile'
    resample_spacing: tuple = (1.0, 1.0, 1.0)  # (z, y, x)
    crop_center_size: Optional[tuple] = None  # (z, y, x)
    pad_to_shape: Optional[tuple] = None  # (z, y, x)
    
    # 窗宽窗位配置
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    
    # ROI 处理配置
    roi_padding: tuple = (10, 10, 10)  # (z, y, x)
    min_roi_size: int = 100  # 最小 ROI 体素数量
    
    # 输出配置
    output_format: str = 'nii.gz'
    compression_level: int = 1
    preserve_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ProcessingConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


@dataclass
class ConversionConfig:
    """格式转换配置"""
    # DICOM 转换配置
    dicom_output_dir: str = 'nifti_output'
    dicom_series_as_volume: bool = True
    dicom_single_slice_handling: str = 'expand'  # 'expand', 'skip'
    
    # 批量转换配置
    batch_skip_existing: bool = True
    batch_output_structure: str = 'flat'  # 'flat', 'preserve'
    
    # 文件匹配配置
    image_mask_patterns: list = None
    file_extensions: list = None
    
    # 输出配置
    overwrite_existing: bool = False
    create_backup: bool = True
    backup_suffix: str = '.backup'
    
    def __post_init__(self):
        """初始化后处理"""
        if self.image_mask_patterns is None:
            self.image_mask_patterns = [
                '{name}_mask{ext}',
                '{name}_seg{ext}',
                '{name}_label{ext}',
                '{name}_roi{ext}'
            ]
        
        if self.file_extensions is None:
            self.file_extensions = ['.nii', '.nii.gz', '.dcm', '.dicom']
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConversionConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ConversionConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.processing_config_file = self.config_dir / 'processing_config.json'
        self.conversion_config_file = self.config_dir / 'conversion_config.json'
    
    def get_processing_config(self) -> ProcessingConfig:
        """获取处理配置"""
        if self.processing_config_file.exists():
            return ProcessingConfig.load(self.processing_config_file)
        else:
            # 创建默认配置
            config = ProcessingConfig()
            config.save(self.processing_config_file)
            return config
    
    def get_conversion_config(self) -> ConversionConfig:
        """获取转换配置"""
        if self.conversion_config_file.exists():
            return ConversionConfig.load(self.conversion_config_file)
        else:
            # 创建默认配置
            config = ConversionConfig()
            config.save(self.conversion_config_file)
            return config
    
    def update_processing_config(self, **kwargs) -> None:
        """更新处理配置"""
        config = self.get_processing_config()
        
        # 更新指定字段
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown processing config field: {key}")
        
        config.save(self.processing_config_file)
    
    def update_conversion_config(self, **kwargs) -> None:
        """更新转换配置"""
        config = self.get_conversion_config()
        
        # 更新指定字段
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown conversion config field: {key}")
        
        config.save(self.conversion_config_file)
    
    def export_configs(self, export_dir: Union[str, Path]) -> None:
        """导出所有配置"""
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出处理配置
        processing_config = self.get_processing_config()
        processing_config.save(export_dir / 'processing_config.json')
        
        # 导出转换配置
        conversion_config = self.get_conversion_config()
        conversion_config.save(export_dir / 'conversion_config.json')
        
        # 创建配置摘要
        summary = {
            'processing_config': processing_config.to_dict(),
            'conversion_config': conversion_config.to_dict(),
            'export_timestamp': str(Path().cwd())
        }
        
        with open(export_dir / 'config_summary.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def import_configs(self, import_dir: Union[str, Path]) -> None:
        """导入配置"""
        import_dir = Path(import_dir)
        
        # 导入处理配置
        processing_file = import_dir / 'processing_config.json'
        if processing_file.exists():
            processing_config = ProcessingConfig.load(processing_file)
            processing_config.save(self.processing_config_file)
        
        # 导入转换配置
        conversion_file = import_dir / 'conversion_config.json'
        if conversion_file.exists():
            conversion_config = ConversionConfig.load(conversion_file)
            conversion_config.save(self.conversion_config_file)
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        # 删除现有配置文件
        if self.processing_config_file.exists():
            self.processing_config_file.unlink()
        
        if self.conversion_config_file.exists():
            self.conversion_config_file.unlink()
        
        # 创建默认配置
        self.get_processing_config()
        self.get_conversion_config()


# 预定义配置模板
CT_CHEST_CONFIG = ProcessingConfig(
    normalize_method='z_score',
    resample_spacing=(1.0, 1.0, 1.0),
    window_center=-400,
    window_width=1500,
    roi_padding=(20, 20, 20)
)

CT_ABDOMEN_CONFIG = ProcessingConfig(
    normalize_method='z_score',
    resample_spacing=(1.0, 1.0, 1.0),
    window_center=40,
    window_width=400,
    roi_padding=(15, 15, 15)
)

MRI_BRAIN_CONFIG = ProcessingConfig(
    normalize_method='percentile',
    resample_spacing=(1.0, 1.0, 1.0),
    window_center=None,
    window_width=None,
    roi_padding=(10, 10, 10)
)

# 转换配置模板
MEDICAL_RESEARCH_CONFIG = ConversionConfig(
    dicom_output_dir='processed_data',
    batch_skip_existing=True,
    batch_output_structure='preserve',
    create_backup=True
)

CLINICAL_WORKFLOW_CONFIG = ConversionConfig(
    dicom_output_dir='clinical_output',
    batch_skip_existing=False,
    batch_output_structure='flat',
    overwrite_existing=True,
    create_backup=False
)