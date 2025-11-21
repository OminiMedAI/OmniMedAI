"""
onem_process - 医学图像处理模块

主要功能：
1. DICOM 到 NIfTI 格式转换
2. 批量图像和掩码格式转换
3. 3D ROI 区域切分
"""

from .converters import *
from .processors import *
from .utils import *

__VERSION__ = '1.0.0'

__all__ = [
    'DicomToNiftiConverter',
    'BatchConverter', 
    'ROIProcessor',
    'ImageProcessor',
    'file_utils',
    'medical_utils'
]