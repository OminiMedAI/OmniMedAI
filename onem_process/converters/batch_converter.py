"""
批量格式转换器
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import shutil

try:
    import nibabel as nib
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from .dicom_to_nifti import DicomToNiftiConverter
from ..utils import file_utils


class BatchConverter:
    """批量格式转换器，支持 images 和 masks 目录的智能转换"""
    
    def __init__(self, base_dir: str, output_base_dir: Optional[str] = None):
        """
        初始化批量转换器
        
        Args:
            base_dir: 基础目录，应包含 images 和 masks 子目录
            output_base_dir: 输出基础目录，如果为None则在base_dir下创建output目录
        """
        self.base_dir = Path(base_dir)
        self.output_base_dir = output_base_dir or str(self.base_dir / "output")
        self.logger = self._setup_logger()
        
        # 初始化 DICOM 转换器
        self.dicom_converter = DicomToNiftiConverter()
        
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
    
    def convert_dataset(self, 
                       images_dir: str = "images", 
                       masks_dir: str = "masks",
                       skip_existing: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        转换整个数据集
        
        Args:
            images_dir: 图像目录名称
            masks_dir: 掩码目录名称
            skip_existing: 是否跳过已存在的文件
            
        Returns:
            (converted_images, converted_masks) 转换后的文件路径字典
        """
        images_path = self.base_dir / images_dir
        masks_path = self.base_dir / masks_dir
        
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        
        # 转换图像
        self.logger.info(f"Converting images from {images_path}")
        converted_images = self._convert_directory(
            images_path, 
            output_subdir="images_nifti",
            skip_existing=skip_existing
        )
        
        # 转换掩码（如果存在）
        converted_masks = {}
        if masks_path.exists():
            self.logger.info(f"Converting masks from {masks_path}")
            converted_masks = self._convert_directory(
                masks_path, 
                output_subdir="masks_nifti",
                skip_existing=skip_existing
            )
        else:
            self.logger.warning(f"Masks directory not found: {masks_path}")
        
        return converted_images, converted_masks
    
    def _convert_directory(self, 
                          input_dir: Path, 
                          output_subdir: str,
                          skip_existing: bool) -> Dict[str, str]:
        """
        转换目录中的所有文件
        
        Args:
            input_dir: 输入目录
            output_subdir: 输出子目录
            skip_existing: 是否跳过已存在文件
            
        Returns:
            转换后的文件路径字典 {input_path: output_path}
        """
        output_dir = Path(self.output_base_dir) / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_files = {}
        
        # 遍历输入目录
        for item in input_dir.iterdir():
            if item.is_file():
                # 单文件转换
                converted_path = self._convert_file_or_directory(item, output_dir, skip_existing)
                if converted_path:
                    converted_files[str(item)] = converted_path
                    
            elif item.is_dir():
                # 目录转换
                converted_path = self._convert_file_or_directory(item, output_dir, skip_existing)
                if converted_path:
                    converted_files[str(item)] = converted_path
        
        return converted_files
    
    def _convert_file_or_directory(self, 
                                   input_path: Path, 
                                   output_dir: Path,
                                   skip_existing: bool) -> Optional[str]:
        """
        转换单个文件或目录
        
        Args:
            input_path: 输入路径
            output_dir: 输出目录
            skip_existing: 是否跳过已存在文件
            
        Returns:
            转换后的文件路径，如果转换失败返回None
        """
        try:
            # 确定输出文件名
            output_filename = input_path.name
            
            # 根据输入类型决定转换方式
            if input_path.is_file() and self._is_dicom_file(input_path):
                # 单个 DICOM 文件
                output_path = output_dir / f"{input_path.stem}.nii.gz"
                
                if skip_existing and output_path.exists():
                    self.logger.info(f"Skipping existing file: {output_path}")
                    return str(output_path)
                
                return self.dicom_converter.convert_single_dicom(str(input_path), str(output_path))
                
            elif input_path.is_dir():
                # DICOM 序列目录或 NIfTI 文件
                output_path = output_dir / f"{input_path.name}.nii.gz"
                
                if skip_existing and output_path.exists():
                    self.logger.info(f"Skipping existing file: {output_path}")
                    return str(output_path)
                
                # 检查是否是 DICOM 序列
                if self._is_dicom_directory(input_path):
                    return self.dicom_converter.convert_dicom_series(str(input_path), str(output_path))
                else:
                    # 可能是包含 NIfTI 文件的目录，直接复制
                    return self._handle_nifti_directory(input_path, output_dir, skip_existing)
            
            elif input_path.suffix.lower() in ['.nii', '.nii.gz']:
                # 已经是 NIfTI 文件，直接复制
                output_path = output_dir / output_filename
                
                if skip_existing and output_path.exists():
                    return str(output_path)
                
                shutil.copy2(input_path, output_path)
                self.logger.info(f"Copied NIfTI file: {input_path} -> {output_path}")
                return str(output_path)
            
            else:
                self.logger.warning(f"Unsupported file type: {input_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting {input_path}: {e}")
            return None
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """检查是否是 DICOM 文件"""
        try:
            import pydicom
            if file_path.suffix.lower() in ['.dcm', '.dicom']:
                return True
            
            # 尝试读取文件头判断
            try:
                pydicom.dcmread(str(file_path), stop_before_pixels=True)
                return True
            except:
                return False
                
        except ImportError:
            # 如果没有 pydicom，根据扩展名判断
            return file_path.suffix.lower() in ['.dcm', '.dicom']
    
    def _is_dicom_directory(self, dir_path: Path) -> bool:
        """检查目录是否包含 DICOM 文件"""
        dicom_extensions = ['.dcm', '.dicom', '.DCM', '.DICOM']
        
        # 查找 DICOM 文件
        for ext in dicom_extensions:
            if list(dir_path.glob(f"*{ext}")):
                return True
        
        # 查找无扩展名的文件（可能是 DICOM）
        no_ext_files = [f for f in dir_path.iterdir() 
                       if f.is_file() and not f.suffix]
        
        if len(no_ext_files) > 1:  # 可能有多个 DICOM 切片
            return True
            
        return False
    
    def _handle_nifti_directory(self, 
                               input_dir: Path, 
                               output_dir: Path,
                               skip_existing: bool) -> Optional[str]:
        """
        处理包含 NIfTI 文件的目录
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            skip_existing: 是否跳过已存在文件
            
        Returns:
            处理后的主文件路径
        """
        # 查找 NIfTI 文件
        nifti_files = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
        
        if not nifti_files:
            self.logger.warning(f"No NIfTI files found in {input_dir}")
            return None
        
        if len(nifti_files) == 1:
            # 单个文件，直接复制
            output_path = output_dir / nifti_files[0].name
            if skip_existing and output_path.exists():
                return str(output_path)
            
            shutil.copy2(nifti_files[0], output_path)
            self.logger.info(f"Copied NIfTI file: {nifti_files[0]} -> {output_path}")
            return str(output_path)
        
        else:
            # 多个文件，创建子目录
            output_subdir = output_dir / input_dir.name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            main_file = None
            for nifti_file in nifti_files:
                output_file = output_subdir / nifti_file.name
                if skip_existing and output_file.exists():
                    continue
                
                shutil.copy2(nifti_file, output_file)
                self.logger.info(f"Copied NIfTI file: {nifti_file} -> {output_file}")
                
                # 假设第一个文件是主文件
                if main_file is None:
                    main_file = str(output_file)
            
            return main_file
    
    def get_conversion_summary(self, 
                            converted_images: Dict[str, str], 
                            converted_masks: Dict[str, str]) -> Dict[str, Any]:
        """
        获取转换摘要
        
        Args:
            converted_images: 转换的图像文件
            converted_masks: 转换的掩码文件
            
        Returns:
            转换摘要信息
        """
        return {
            'total_images': len(converted_images),
            'total_masks': len(converted_masks),
            'successful_conversions': len([v for v in converted_images.values() if v]),
            'image_files': list(converted_images.keys()),
            'mask_files': list(converted_masks.keys()),
            'output_base_dir': self.output_base_dir
        }