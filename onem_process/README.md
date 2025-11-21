# onem_process

医学图像处理模块，主要功能包括 DICOM 到 NIfTI 格式转换、批量格式转换和 3D ROI 区域切分。

## 功能特性

### 1. 格式转换
- **DICOM 到 NIfTI 转换**: 支持单个 DICOM 文件和 DICOM 序列转换
- **批量格式转换**: 自动识别和转换 images/masks 目录中的多种格式
- **智能文件匹配**: 自动匹配图像和对应的掩码文件

### 2. 3D ROI 处理
- **ROI 提取**: 基于掩码提取 3D 感兴趣区域
- **批量处理**: 支持批量 ROI 提取
- **ROI 统计**: 计算体积、质心、强度统计等信息
- **边界框扩展**: 可配置的 ROI 边界扩展

### 3. 图像处理
- **重采样**: 调整图像分辨率
- **归一化**: Z-score、Min-Max、百分位数归一化
- **裁剪/填充**: 中心裁剪和边界填充
- **窗宽窗位**: CT 图像窗宽窗位调整

### 4. 配置管理
- **灵活配置**: 支持处理参数和转换参数配置
- **配置导入导出**: 保存和加载配置文件
- **预设模板**: 提供常用配置模板

## 安装依赖

```bash
# 基础依赖
pip install numpy nibabel pydicom

# 图像处理依赖
pip install opencv-python scipy

# 高级功能依赖（可选）
pip install SimpleITK

# 开发和测试依赖
pip install unittest
```

## 快速开始

### 1. 单个 DICOM 文件转换

```python
from onem_process.converters import DicomToNiftiConverter

# 创建转换器
converter = DicomToNiftiConverter(output_dir="output/nifti")

# 转换单个文件
nifti_file = converter.convert_single_dicom("data/patient001.dcm")
print(f"转换完成: {nifti_file}")
```

### 2. DICOM 序列转换

```python
# 转换 DICOM 序列
nifti_file = converter.convert_dicom_series("data/dicom_series/")
print(f"序列转换完成: {nifti_file}")
```

### 3. 批量数据集转换

```python
from onem_process.converters import BatchConverter

# 创建批量转换器
converter = BatchConverter(
    base_dir="data/dataset",
    output_base_dir="output/converted"
)

# 转换整个数据集
converted_images, converted_masks = converter.convert_dataset(
    images_dir="images",
    masks_dir="masks"
)

# 获取转换摘要
summary = converter.get_conversion_summary(converted_images, converted_masks)
print(f"转换了 {summary['total_images']} 个图像文件")
print(f"转换了 {summary['total_masks']} 个掩码文件")
```

### 4. ROI 提取

```python
from onem_process.processors import ROIProcessor

# 创建 ROI 处理器
roi_processor = ROIProcessor(padding=(10, 10, 10))

# 提取单个 ROI
roi_image, roi_mask = roi_processor.extract_roi_from_mask(
    image_path="output/converted/images_nifti/patient001.nii.gz",
    mask_path="output/converted/masks_nifti/patient001_mask.nii.gz",
    output_dir="output/rois",
    roi_name="patient001_roi"
)

print(f"ROI 图像: {roi_image}")
print(f"ROI 掩码: {roi_mask}")

# 获取 ROI 统计信息
stats = roi_processor.get_roi_statistics(roi_mask)
print(f"体积: {stats['actual_volume_mm3']:.2f} mm³")
print(f"质心: {stats['centroid']}")
```

### 5. 批量 ROI 提取

```python
# 批量提取 ROI
results = roi_processor.batch_extract_rois(
    image_dir="output/converted/images_nifti",
    mask_dir="output/converted/masks_nifti", 
    output_dir="output/batch_rois",
    file_pattern="*.nii.gz"
)

print(f"成功处理了 {len(results)} 个文件")
```

### 6. 图像处理

```python
from onem_process.processors import ImageProcessor

# 创建图像处理器
processor = ImageProcessor()

# 重采样
resampled = processor.resample_image(
    "input.nii.gz",
    target_spacing=(1.0, 1.0, 1.0),
    output_path="resampled.nii.gz"
)

# 归一化
normalized = processor.normalize_image(
    "resampled.nii.gz",
    method='z_score',
    output_path="normalized.nii.gz"
)

# 中心裁剪
cropped = processor.crop_image(
    "normalized.nii.gz",
    crop_size=(128, 128, 128),
    output_path="cropped.nii.gz"
)
```

### 7. 配置管理

```python
from onem_process.config import ConfigManager, ProcessingConfig

# 创建配置管理器
config_manager = ConfigManager("config")

# 获取配置
processing_config = config_manager.get_processing_config()
print(f"当前归一化方法: {processing_config.normalize_method}")

# 更新配置
config_manager.update_processing_config(
    normalize_method='percentile',
    resample_spacing=(0.5, 0.5, 0.5),
    roi_padding=(20, 20, 20)
)

# 导出配置
config_manager.export_configs("output/configs")
```

## 目录结构

```
onem_process/
├── __init__.py              # 主入口
├── converters/               # 格式转换器
│   ├── __init__.py
│   ├── dicom_to_nifti.py   # DICOM 转换器
│   └── batch_converter.py   # 批量转换器
├── processors/               # 图像处理器
│   ├── __init__.py
│   ├── roi_processor.py     # ROI 处理器
│   └── image_processor.py   # 图像处理器
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── file_utils.py        # 文件工具
│   └── medical_utils.py     # 医学工具
├── config/                   # 配置模块
│   ├── __init__.py
│   └── settings.py          # 配置管理
├── example_usage.py         # 使用示例
├── test_basic.py           # 基础测试
└── README.md               # 说明文档
```

## API 文档

### DicomToNiftiConverter

```python
class DicomToNiftiConverter:
    def __init__(self, output_dir: Optional[str] = None)
    def convert_single_dicom(self, dicom_path: str, output_path: Optional[str] = None) -> str
    def convert_dicom_series(self, dicom_dir: str, output_path: Optional[str] = None) -> str
    def batch_convert(self, input_paths: List[str], output_dir: Optional[str] = None) -> List[str]
```

### BatchConverter

```python
class BatchConverter:
    def __init__(self, base_dir: str, output_base_dir: Optional[str] = None)
    def convert_dataset(self, images_dir: str = "images", masks_dir: str = "masks", 
                       skip_existing: bool = True) -> Tuple[Dict, Dict]
    def get_conversion_summary(self, converted_images: Dict, converted_masks: Dict) -> Dict
```

### ROIProcessor

```python
class ROIProcessor:
    def __init__(self, padding: Tuple[int, int, int] = (10, 10, 10))
    def extract_roi_from_mask(self, image_path: str, mask_path: str, 
                             output_dir: str, roi_name: Optional[str] = None) -> Tuple[str, str]
    def batch_extract_rois(self, image_dir: str, mask_dir: str, 
                          output_dir: str, file_pattern: str = "*.nii.gz") -> List[Dict]
    def get_roi_statistics(self, roi_path: str) -> Dict
```

### ImageProcessor

```python
class ImageProcessor:
    def resample_image(self, image_path: str, target_spacing: Tuple[float, float, float],
                      output_path: str, interpolation: str = 'linear') -> str
    def normalize_image(self, image_path: str, method: str = 'z_score',
                       output_path: Optional[str] = None) -> str
    def crop_center(self, image_path: str, crop_size: Tuple[int, int, int],
                   output_path: str) -> str
    def pad_image(self, image_path: str, target_shape: Tuple[int, int, int],
                  output_path: str, pad_value: Union[float, str] = 'min') -> str
```

## 配置选项

### ProcessingConfig

- `normalize_method`: 归一化方法 ('z_score', 'min_max', 'percentile')
- `resample_spacing`: 重采样间距 (z, y, x)
- `roi_padding`: ROI 边界扩展 (z, y, x)
- `window_center/window_width`: 窗宽窗位设置

### ConversionConfig

- `dicom_output_dir`: DICOM 转换输出目录
- `batch_skip_existing`: 是否跳过已存在文件
- `image_mask_patterns`: 图像掩码匹配模式
- `file_extensions`: 支持的文件扩展名

## 预设配置模板

```python
from onem_process.config.settings import CT_CHEST_CONFIG, MRI_BRAIN_CONFIG

# 使用预设配置
config_manager.update_processing_config(
    normalize_method=CT_CHEST_CONFIG.normalize_method,
    window_center=CT_CHEST_CONFIG.window_center,
    window_width=CT_CHEST_CONFIG.window_width
)
```

## 错误处理

模块包含完善的错误处理机制：

- **依赖检查**: 自动检测所需依赖是否安装
- **文件验证**: 验证输入文件格式和完整性
- **异常捕获**: 提供详细的错误信息和日志
- **可选导入**: 某些功能在缺少依赖时会优雅降级

## 性能优化

- **批量处理**: 支持并行处理大量文件
- **内存优化**: 大文件采用流式处理
- **缓存机制**: 避免重复处理相同文件
- **进度反馈**: 提供处理进度信息

## 测试

运行测试：

```bash
cd onem_process
python test_basic.py
```

运行示例：

```bash
python example_usage.py
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个模块。

## 更新日志

### v1.0.0
- 初始版本发布
- 实现 DICOM 到 NIfTI 转换功能
- 实现 3D ROI 提取功能
- 添加批量处理和配置管理功能