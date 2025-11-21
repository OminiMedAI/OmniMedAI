# onem_segment

医学影像自动ROI分割模块 - 对 images 目录下的 NIfTI 文件进行自动ROI勾画，根据z轴判断选择2D或3D模型，将分割结果保存为NIfTI文件到指定目录。

## 功能特性

### 1. 智能模型选择
- **自动维度分析**: 基于z轴特征（切片数量、间距、内容变化）自动判断2D/3D处理
- **多维度评估**: 综合考虑切片数量、层间距、内容变化等因素
- **置信度评估**: 提供选择建议的置信度评分
- **批量分析**: 支持批量图像分析和统一推荐

### 2. 多模型支持
- **2D模型**: 2D U-Net、2D Attention U-Net等
- **3D模型**: 3D U-Net、3D V-Net等
- **质量分级**: 默认质量和高质量处理模式
- **模型管理**: 统一的模型加载和推理接口

### 3. 灵活的预处理
- **多种归一化**: Z-score、Min-Max、Robust归一化
- **图像重采样**: 支持不同间距的重采样
- **尺寸调整**: 自动调整到模型输入尺寸
- **对比度增强**: 直方图均衡化、CLAHE等

### 4. 智能后处理
- **连通域分析**: 移除小的连通分量
- **形态学操作**: 开闭运算、孔洞填充
- **边界优化**: ROI边界平滑处理
- **质量控制**: 自动质量评估和报告

### 5. 配置管理
- **预设配置**: 针对不同模态和需求的预设参数
- **自定义配置**: 灵活的参数定制和验证
- **配置保存**: 配置的保存和加载
- **批量配置**: 支持批量处理的统一配置

## 安装依赖

```bash
# 核心依赖
pip install numpy pandas

# 医学图像处理
pip install nibabel SimpleITK

# 深度学习框架
pip install torch torchvision

# 图像处理
pip install scikit-image scipy

# 可视化（可选）
pip install matplotlib seaborn

# 开发和测试
pip install unittest
```

## 快速开始

### 1. 基础ROI分割

```python
from onem_segment import ROISegmenter

# 创建分割器（默认配置）
segmenter = ROISegmenter()

# 对单个图像进行分割
result = segmenter.segment_image(
    image_path="images/patient001.nii.gz",
    output_path=None,  # 自动生成输出路径
    model_type='auto',  # 自动选择2D/3D
    quality='default'
)

print(f"分割完成: {result['output_path']}")
print(f"使用模型: {result['model_used']}")
print(f"处理类型: {result['model_type']}")

# 获取分割统计信息
stats = segmenter.get_segmentation_statistics(result)
print(f"ROI体积: {stats['roi_volume_voxels']} 个体素")
print(f"ROI占比: {stats['roi_percentage']:.2f}%")
```

### 2. 批量分割处理

```python
# 批量分割
results = segmenter.segment_batch(
    image_dir="data/images",
    output_dir="output/segmentations",
    file_pattern="*.nii.gz",
    model_type='auto',
    quality='high_quality'
)

successful = [r for r in results if 'error' not in r]
print(f"成功分割: {len(successful)} 个图像")

# 分析模型使用情况
model_usage = {}
for result in successful:
    model = result.get('model_used', 'unknown')
    model_usage[model] = model_usage.get(model, 0) + 1

print("模型使用统计:")
for model, count in model_usage.items():
    print(f"  {model}: {count} 个图像")
```

### 3. 图像维度分析

```python
from onem_segment import ImageDimensionAnalyzer

# 创建图像分析器
analyzer = ImageDimensionAnalyzer(
    min_3d_slices=30,
    max_slice_spacing=5.0,
    min_content_variation=0.1
)

# 分析单个图像
analysis = analyzer.analyze_image("images/patient001.nii.gz")

print(f"图像形状: {analysis['shape']}")
print(f"切片数量: {analysis['n_slices']}")
print(f"层间距: {analysis['slice_spacing']:.2f} mm")
print(f"推荐处理: {'3D' if analysis['is_3d'] else '2D'}")

print("内容分析:")
content = analysis['content_analysis']
print(f"  平均强度: {content['mean']:.2f}")
print(f"  标准差: {content['std']:.2f}")
print(f"  偏度: {content['skewness']:.2f}")

print("质量指标:")
quality = analysis['quality_metrics']
print(f"  信噪比: {quality['snr']:.2f}")
print(f"  对比噪声比: {quality['cnr']:.2f}")
```

### 4. 使用预设配置

```python
from onem_segment import PRESET_CONFIGS

# 使用高质量预设
config = PRESET_CONFIGS['high_quality']
segmenter = ROISegmenter(config)

print(f"使用高质量预设:")
print(f"  质量级别: {config.quality_level}")
print(f"  归一化: {config.normalization}")
print(f"  形态学操作: {config.morphological_operations}")
print(f"  返回概率图: {config.return_probabilities}")
```

### 5. 自定义配置

```python
from onem_segment import SegmentationConfig, create_custom_config

# 创建自定义配置
custom_config = create_custom_config(
    model_type='auto',
    quality_level='high_quality',
    input_size=(512, 512),  # 2D模型输入尺寸
    normalization='z_score',
    min_roi_volume=20,
    morphological_operations=True,
    return_probabilities=True,
    generate_statistics=True
)

segmenter = ROISegmenter(custom_config)

# 保存配置
custom_config.save("custom_segmentation_config.json")
```

## 目录结构

```
onem_segment/
├── __init__.py                      # 主入口文件
├── segmenters/                       # 分割器模块
│   ├── __init__.py
│   └── roi_segmenter.py             # ROI分割器
├── models/                          # 模型管理
│   ├── __init__.py
│   └── model_manager.py             # 模型管理器
├── config/                          # 配置管理
│   ├── __init__.py
│   └── settings.py                 # 配置类和预设
├── utils/                           # 工具函数
│   ├── __init__.py
│   ├── image_analyzer.py           # 图像维度分析
│   ├── file_utils.py               # 文件操作工具
│   └── preprocessing.py           # 图像预处理
├── example_usage.py                  # 使用示例
├── test_basic.py                    # 基础测试
└── README.md                       # 说明文档
```

## API 文档

### ROISegmenter

```python
class ROISegmenter:
    def __init__(self, config=None)
    
    def segment_image(self, image_path: str, output_path: str = None,
                    model_type: str = 'auto', quality: str = 'default',
                    return_probabilities: bool = False) -> Dict
    
    def segment_batch(self, image_dir: str, output_dir: str = None,
                    file_pattern: str = "*.nii.gz", model_type: str = 'auto',
                    quality: str = 'default', n_jobs: int = 1) -> List[Dict]
    
    def get_segmentation_statistics(self, result: Dict) -> Dict
```

### ImageDimensionAnalyzer

```python
class ImageDimensionAnalyzer:
    def __init__(self, min_3d_slices: int = 30, max_slice_spacing: float = 5.0,
                 min_content_variation: float = 0.1)
    
    def analyze_image(self, image_path: str) -> Dict
    
    def batch_analyze(self, image_paths: list) -> Dict
```

### SegmentationConfig

```python
class SegmentationConfig:
    # 模型选择参数
    model_type: str = 'auto'          # 'auto', '2d', '3d'
    quality_level: str = 'default'     # 'default', 'high_quality'
    device: str = 'auto'             # 'cpu', 'cuda', 'auto'
    
    # 图像分析参数
    min_3d_slices: int = 30
    max_slice_spacing: float = 5.0
    min_content_variation: float = 0.1
    
    # 预处理参数
    normalization: str = 'z_score'
    input_size_2d: Tuple[int, int] = (256, 256)
    input_size_3d: Tuple[int, int, int] = (64, 64, 64)
    
    # 后处理参数
    min_roi_volume: int = 10
    morphological_operations: bool = True
    morph_iterations: int = 2
```

## 预设配置

### 可用预设配置

```python
from onem_segment import PRESET_CONFIGS

# 默认配置
config = PRESET_CONFIGS['default']

# 快速处理
config = PRESET_CONFIGS['fast']

# 高质量处理
config = PRESET_CONFIGS['high_quality']

# 仅2D处理
config = PRESET_CONFIGS['2d_only']

# 仅3D处理
config = PRESET_CONFIGS['3d_only']

# CT器官分割
config = PRESET_CONFIGS['ct_organ']

# MRI脑部分割
config = PRESET_CONFIGS['mri_brain']

# PET肿瘤分割
config = PRESET_CONFIGS['pet_tumor']

# 研究级配置
config = PRESET_CONFIGS['research']

# 生产环境配置
config = PRESET_CONFIGS['production']

# 内存高效配置
config = PRESET_CONFIGS['memory_efficient']
```

### 预设配置对比

| 配置名称 | 模型选择 | 质量级别 | 归一化 | 输入尺寸 | 适用场景 |
|---------|---------|---------|---------|---------|---------|
| default | auto | default | z_score | 256×256 / 64³ | 日常使用 |
| fast | auto | default | z_score | 128×128 / 32³ | 快速处理 |
| high_quality | auto | high_quality | z_score | 512×512 / 128³ | 高质量输出 |
| 2d_only | 2d | default | z_score | 256×256 | 仅2D处理 |
| 3d_only | 3d | default | z_score | 64×64×64 | 仅3D处理 |
| ct_organ | auto | high_quality | z_score | 256×256 / 64³ | CT器官分割 |
| mri_brain | auto | high_quality | z_score | 256×256 / 96³ | MRI脑部分割 |
| pet_tumor | auto | default | min_max | 256×256 / 64³ | PET肿瘤分割 |
| research | auto | high_quality | z_score | 512×512 / 256³ | 研究用途 |
| production | auto | default | z_score | 256×256 / 64³ | 生产环境 |

## 2D/3D模型选择逻辑

### 判断标准

系统基于以下标准自动选择2D或3D模型：

1. **切片数量**: 
   - 少于 `min_3d_slices` (默认30) → 选择2D
   - 多于等于 `min_3d_slices` → 候选3D

2. **层间距**:
   - 大于 `max_slice_spacing` (默认5.0mm) → 选择2D
   - 小于等于 `max_slice_spacing` → 候选3D

3. **内容变化**:
   - 计算相邻切片间的内容差异
   - 差异小于 `min_content_variation` → 选择2D
   - 差异大于等于 `min_content_variation` → 候选3D

4. **综合判断**:
   - 需要2个或3个条件满足才选择3D处理
   - 否则选择2D处理

### 示例场景

```python
# 典型CT扫描 (64层, 1.25mm间距, 高变化)
analysis = {
    'n_slices': 64,           # ✅ 满足切片数量
    'slice_spacing': 1.25,     # ✅ 满足层间距
    'content_variation': 0.3,   # ✅ 满足内容变化
    'is_3d': True,            # 推荐使用3D模型
    'confidence': 'high'
}

# 薄层扫描 (15层, 0.5mm间距, 低变化)
analysis = {
    'n_slices': 15,           # ❌ 不满足切片数量
    'slice_spacing': 0.5,     # ✅ 满足层间距
    'content_variation': 0.05,  # ❌ 不满足内容变化
    'is_3d': False,           # 推荐使用2D模型
    'confidence': 'medium'
}
```

## 输出格式

### 分割结果文件

分割结果保存为NIfTI格式，包含：

- **二值掩码**: ROI区域的二值分割 (0=背景, 1=前景)
- **概率图**: (可选)ROI概率值 (0.0-1.0)
- **原始头信息**: 保持输入图像的空间信息

### 文件命名规则

```bash
# 自动输出命名
patient001.nii.gz → patient001_3d_seg.nii.gz
patient002.nii.gz → patient002_2d_seg.nii.gz

# 概率图
patient001.nii.gz → patient001_3d_prob.nii.gz
```

### 统计报告

生成详细的分割统计报告：

```json
{
  "roi_volume_voxels": 1250,
  "roi_volume_mm3": 1562.5,
  "roi_percentage": 2.45,
  "roi_shape": [512, 512, 64],
  "roi_slices": 23,
  "roi_centroid": [256.3, 247.8, 31.2],
  "bounding_box": [[200, 300], [180, 320], [15, 45]]
}
```

## 性能优化

### 1. 内存优化

```python
# 使用内存高效配置
config = PRESET_CONFIGS['memory_efficient']
segmenter = ROISegmenter(config)

# 或自定义内存配置
config = create_custom_config(
    model_type='2d',              # 强制使用2D
    input_size=(128, 128),        # 减小输入尺寸
    batch_inference=True,          # 批量推理
    batch_size=2,                 # 小批量大小
    memory_limit_gb=4.0,          # 内存限制
    use_mixed_precision=True        # 混合精度
)
```

### 2. 处理速度优化

```python
# 快速处理配置
config = PRESET_CONFIGS['fast']
segmenter = ROISegmenter(config)

# 禁用后处理加速
config.morphological_operations = False
config.remove_small_components = False
config.generate_statistics = False
```

### 3. GPU加速

```python
# 指定GPU设备
config.device = 'cuda'

# 使用混合精度
config.use_mixed_precision = True

# 批量推理
config.batch_inference = True
config.batch_size = 4
```

## 测试

运行基础测试：

```bash
cd onem_segment
python test_basic.py
```

运行示例：

```bash
python example_usage.py
```

## 常见问题

### Q: 如何调整2D/3D选择阈值？
A: 可以通过修改ImageDimensionAnalyzer参数：
```python
analyzer = ImageDimensionAnalyzer(
    min_3d_slices=25,      # 降低切片数要求
    max_slice_spacing=6.0,   # 放宽间距要求
    min_content_variation=0.05  # 降低变化要求
)
```

### Q: 如何处理不同模态的图像？
A: 使用对应的预设配置：
```python
# CT图像
config = PRESET_CONFIGS['ct_organ']

# MRI图像  
config = PRESET_CONFIGS['mri_brain']

# PET图像
config = PRESET_CONFIGS['pet_tumor']
```

### Q: 如何提高分割质量？
A: 使用高质量配置：
```python
config = PRESET_CONFIGS['high_quality']
# 或者研究级配置
config = PRESET_CONFIGS['research']
```

### Q: 如何处理大内存图像？
A: 使用内存优化策略：
```python
config = PRESET_CONFIGS['memory_efficient']
# 或强制使用2D处理
config.model_type = '2d'
config.input_size_2d = (128, 128)
```

### Q: 如何批量处理不同尺寸的图像？
A: 使用自动尺寸调整：
```python
config.resize_input = True  # 启用自动重采样
config.interpolation_method = 'bilinear'  # 选择插值方法
```

## 致谢

专业版，请联系：
- **微信**: AcePwn
- **邮箱**: acezqy@gmail.com

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个模块。

## 更新日志

### v1.0.0
- 初始版本发布
- 实现自动2D/3D模型选择功能
- 实现多模型支持和统一接口
- 实现灵活的图像预处理和后处理
- 实现完整的配置管理系统
- 提供多种预设配置模板
- 实现图像维度分析和批量处理
- 提供完整的示例和测试