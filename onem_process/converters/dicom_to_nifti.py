"""
DICOM 到 NIfTI 格式转换器
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

try:
    import pydicom
    import nibabel as nib
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEPS = str(e)

from ..utils import file_utils, medical_utils


class DicomToNiftiConverter:
    """DICOM 到 NIfTI 格式转换器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            output_dir: 输出目录，如果为None则使用输入目录的子目录
        """
        if not HAS_DEPS:
            raise ImportError(f"Missing dependencies: {MISSING_DEPS}. "
                            "Install with: pip install pydicom nibabel numpy")
        
        self.output_dir = output_dir
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
    
    def convert_single_dicom(self, dicom_path: str, output_path: Optional[str] = None) -> str:
        """
        转换单个 DICOM 文件
        
        Args:
            dicom_path: DICOM 文件路径
            output_path: 输出 NIfTI 文件路径，如果为None则自动生成
            
        Returns:
            输出文件路径
        """
        dicom_path = Path(dicom_path)
        
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        # 生成输出路径
        if output_path is None:
            if self.output_dir:
                output_dir = Path(self.output_dir)
            else:
                output_dir = dicom_path.parent / "nifti"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{dicom_path.stem}.nii.gz"
        
        # 读取 DICOM
        try:
            ds = pydicom.dcmread(str(dicom_path))
            
            # 提取像素数据
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                
                # 处理多帧 DICOM
                if len(pixel_array.shape) == 3:
                    pixel_array = np.transpose(pixel_array, (2, 0, 1))
                elif len(pixel_array.shape) == 2:
                    pixel_array = pixel_array[np.newaxis, ...]
                
                # 获取空间信息
                affine = self._get_affine_matrix(ds)
                
                # 创建 NIfTI 图像
                nifti_img = nib.Nifti1Image(pixel_array.astype(np.float32), affine)
                
                # 保存 NIfTI 文件
                nib.save(nifti_img, str(output_path))
                
                self.logger.info(f"Successfully converted {dicom_path} to {output_path}")
                return str(output_path)
                
            else:
                raise ValueError("DICOM file has no pixel data")
                
        except Exception as e:
            self.logger.error(f"Error converting {dicom_path}: {e}")
            raise
    
    def convert_dicom_series(self, dicom_dir: str, output_path: Optional[str] = None) -> str:
        """
        转换 DICOM 序列（多个切片文件）
        
        Args:
            dicom_dir: DICOM 文件目录
            output_path: 输出 NIfTI 文件路径
            
        Returns:
            输出文件路径
        """
        dicom_dir = Path(dicom_dir)
        
        if not dicom_dir.exists() or not dicom_dir.is_dir():
            raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")
        
        # 查找所有 DICOM 文件
        dicom_files = list(dicom_dir.glob("*.dcm"))
        dicom_files.extend(dicom_dir.glob("*.DCM"))
        
        if not dicom_files:
            # 尝试查找没有扩展名的文件
            dicom_files = [f for f in dicom_dir.iterdir() 
                          if f.is_file() and not f.suffix]
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
        
        # 读取并排序 DICOM 文件
        dicom_files = self._sort_dicom_files(dicom_files)
        
        # 读取第一个文件获取元数据
        first_slice = pydicom.dcmread(str(dicom_files[0]))
        
        # 读取所有切片数据
        slices_data = []
        for dicom_file in dicom_files:
            ds = pydicom.dcmread(str(dicom_file))
            slices_data.append(ds.pixel_array)
        
        # 堆叠成3D体积
        volume = np.stack(slices_data, axis=0)
        
        # 获取仿射矩阵
        affine = self._get_affine_matrix(first_slice)
        
        # 生成输出路径
        if output_path is None:
            if self.output_dir:
                output_dir = Path(self.output_dir)
            else:
                output_dir = dicom_dir.parent / "nifti"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{dicom_dir.name}.nii.gz"
        
        # 创建并保存 NIfTI 图像
        nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
        nib.save(nifti_img, str(output_path))
        
        self.logger.info(f"Successfully converted DICOM series from {dicom_dir} to {output_path}")
        return str(output_path)
    
    def _sort_dicom_files(self, dicom_files: List[Path]) -> List[Path]:
        """根据切片位置排序 DICOM 文件"""
        try:
            # 尝试根据 Instance Number 排序
            sorted_files = []
            for file in dicom_files:
                ds = pydicom.dcmread(str(file))
                instance_num = int(ds.get('InstanceNumber', 0))
                sorted_files.append((instance_num, file))
            
            sorted_files.sort(key=lambda x: x[0])
            return [f[1] for f in sorted_files]
            
        except Exception:
            # 如果失败，使用文件名排序
            return sorted(dicom_files)
    
    def _get_affine_matrix(self, ds) -> np.ndarray:
        """从 DICOM 数据集获取仿射矩阵"""
        try:
            # 获取像素间距
            pixel_spacing = ds.get('PixelSpacing', [1.0, 1.0])
            
            # 获取层厚
            slice_thickness = float(ds.get('SliceThickness', 1.0))
            
            # 获取图像方向
            try:
                image_orientation = ds.get('ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
                image_orientation = [float(x) for x in image_orientation]
            except:
                image_orientation = [1, 0, 0, 0, 1, 0]
            
            # 获取图像位置
            try:
                image_position = ds.get('ImagePositionPatient', [0, 0, 0])
                image_position = [float(x) for x in image_position]
            except:
                image_position = [0, 0, 0]
            
            # 构建仿射矩阵
            affine = np.zeros((4, 4))
            affine[0, 0] = pixel_spacing[0] * image_orientation[0]
            affine[0, 1] = pixel_spacing[0] * image_orientation[1]
            affine[0, 2] = pixel_spacing[0] * image_orientation[2]
            affine[0, 3] = image_position[0]
            
            affine[1, 0] = pixel_spacing[1] * image_orientation[3]
            affine[1, 1] = pixel_spacing[1] * image_orientation[4]
            affine[1, 2] = pixel_spacing[1] * image_orientation[5]
            affine[1, 3] = image_position[1]
            
            affine[2, 0] = 0
            affine[2, 1] = 0
            affine[2, 2] = slice_thickness
            affine[2, 3] = image_position[2]
            
            affine[3, 3] = 1
            
            return affine
            
        except Exception as e:
            self.logger.warning(f"Could not extract affine matrix from DICOM: {e}")
            # 返回单位矩阵
            return np.eye(4)
    
    def batch_convert(self, input_paths: List[str], output_dir: Optional[str] = None) -> List[str]:
        """
        批量转换 DICOM 文件或目录
        
        Args:
            input_paths: 输入文件或目录路径列表
            output_dir: 输出目录
            
        Returns:
            成功转换的文件路径列表
        """
        if output_dir:
            self.output_dir = output_dir
            
        converted_files = []
        
        for path in input_paths:
            path = Path(path)
            
            if path.is_file():
                try:
                    converted_file = self.convert_single_dicom(str(path))
                    converted_files.append(converted_file)
                except Exception as e:
                    self.logger.error(f"Failed to convert {path}: {e}")
                    
            elif path.is_dir():
                try:
                    converted_file = self.convert_dicom_series(str(path))
                    converted_files.append(converted_file)
                except Exception as e:
                    self.logger.error(f"Failed to convert DICOM series {path}: {e}")
        
        return converted_files