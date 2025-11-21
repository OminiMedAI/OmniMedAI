# onem_path

病理图像特征提取模块 - 实现对病理图像目录的基于 CellProfiler 的病理组学特征提取和 TITAN 模型的深度迁移学习特征提取。

## 功能特性

### 1. 传统病理组学特征提取
- **形态学特征**: 面积、周长、圆度、形状描述子等
- **纹理特征**: GLCM、LBP、Gabor 滤波器等纹理特征
- **强度特征**: 统计特征、百分位数、偏度、峰度等
- **颜色特征**: RGB、HSV、Lab 颜色空间特征
- **细胞核特征**: 细胞核特定的形态学特征
- **细胞特征**: 完整细胞的形态学特征
- **细胞质特征**: 细胞质区域特征
- **膜特征**: 细胞膜特征

### 2. 深度迁移学习特征提取
- **TITAN 模型**: 专为病理图像设计的迁移学习模型
- **多骨干网络**: ResNet、EfficientNet、DenseNet、VGG 等
- **注意力机制**: 多头注意力增强特征表达能力
- **层级特征**: 支持提取不同层级的深度特征
- **预训练权重**: 支持使用 ImageNet 预训练权重
- **检查点加载**: 支持加载训练好的模型检查点

### 3. 图像预处理
- **染色归一化**: Reinhard、Macenko、Vahadane 等染色归一化
- **对比度增强**: CLAHE、直方图均衡化、伽马校正
- **噪声抑制**: 高斯滤波、双边滤波、中值滤波
- **尺寸标准化**: 多种插值方法的重采样
- **颜色空间转换**: RGB、HSV、Lab 等颜色空间

### 4. 配置管理
- **预设配置**: 针对不同用途的预设参数
- **自定义配置**: 灵活的参数定制和验证
- **模块化提取**: 可选择性提取特定类型的特征
- **质量控制**: 输入验证和图像质量检查
- **性能优化**: 批处理、并行计算、内存管理

## 安装依赖

```bash
# 核心依赖
pip install numpy pandas

# 图像处理
pip install Pillow scikit-image

# 深度学习框架
pip install torch torchvision

# 可视化（可选）
pip install matplotlib seaborn

# 传统特征提取（可选，如果需要完整的 CellProfiler 功能）
pip install cellprofiler

# OpenCV（可选，用于额外的图像处理）
pip install opencv-python

# 开发和测试
pip install unittest
```

## 快速开始

### 1. 传统病理组学特征提取

```python
from onem_path import CellProfilerExtractor

# 创建提取器
extractor = CellProfilerExtractor()

# 提取单个图像的特征
result = extractor.extract_features(
    image_path="pathology/patient001.tif",
    output_path="features/patient001_features.json",
    modules=['morphological', 'texture', 'intensity', 'color', 'nuclei']
)

print(f"成功提取 {len(result['features'])} 个传统特征")
print("模块结果:", result['module_results'])
```

### 2. 批量传统特征提取

```python
# 批量提取
df = extractor.extract_batch_features(
    image_dir="data/pathology",
    output_dir="output/cellprofiler_features",
    file_pattern="*.tif *.tiff",
    modules=['morphological', 'texture', 'intensity', 'nuclei'],
    n_jobs=4
)

print(f"成功处理 {len(df)} 个图像")
print(f"特征列数: {len([col for col in df.columns if col.startswith('morphological_') or col.startswith('texture_')])}")
```

### 3. TITAN 深度特征提取

```python
from onem_path import TITANExtractor, PathologyConfig, PRESET_CONFIGS

# 使用预设配置
config = PRESET_CONFIGS['high_quality']
extractor = TITANExtractor(config)

# 提取单个图像的深度特征
result = extractor.extract_features(
    image_path="pathology/patient001.tif",
    output_path="features/patient001_titan_features.json"
)

print(f"成功提取 {len(result['features'])} 个深度特征")
print("模型信息:", result['model_info'])
```

### 4. TITAN 批量深度特征提取

```python
# GPU 加速批量提取
gpu_config = PRESET_CONFIGS['gpu_accelerated']
extractor = TITANExtractor(gpu_config)

df = extractor.extract_batch_features(
    image_dir="data/pathology",
    output_dir="output/titan_features",
    batch_size=32,
    num_workers=4
)

print(f"GPU 批量处理完成，特征维度: {len([col for col in df.columns if col.startswith('titan_feature_')])}")
```

### 5. 联合特征提取

```python
# 同时提取传统和深度特征
cellprofiler_extractor = CellProfilerExtractor()
titan_extractor = TITANExtractor()

# 创建输出结构
from onem_path.utils.file_utils import create_output_structure
output_structure = create_output_structure("output/combined_features")

image_files = get_pathology_files("data/pathology")[:5]  # 处理前5个图像
combined_results = []

for image_path in image_files:
    # 提取传统特征
    cp_result = cellprofiler_extractor.extract_features(
        image_path=image_path,
        modules=['morphological', 'texture', 'nuclei']
    )
    
    # 提取深度特征
    titan_result = titan_extractor.extract_features(image_path=image_path)
    
    # 合并结果
    combined = {
        'image_path': image_path,
        'cellprofiler_features': cp_result['features'],
        'titan_features': titan_result['features'],
        'total_features': len(cp_result['features']) + len(titan_result['features'])
    }
    combined_results.append(combined)

print(f"联合特征提取完成: {len(combined_results)} 个图像")
```

### 6. 图像预处理

```python
from onem_path.utils.image_utils import preprocess_pathology_image

# 染色归一化预处理
preprocessed = preprocess_pathology_image(
    image_path="pathology/h&e_stained.tif",
    config={
        'normalize_staining': True,
        'staining_method': 'reinhard',
        'enhance_contrast': True,
        'contrast_method': 'clahe',
        'target_size': (512, 512),
        'color_conversion': 'hsv'
    }
)
```

### 7. 模型性能基准测试

```python
from onem_path.utils.titan_utils import benchmark_titan_extraction, get_available_backbones

# 获取可用的骨干网络
backbones = get_available_backbones()[:3]  # 测试前3个
print(f"可用骨干网络: {backbones}")

# 基准测试
for backbone in backbones:
    from onem_path.models.titan_model import create_titan_model
    
    model = create_titan_model(backbone_name=backbone)
    benchmark_result = benchmark_titan_extraction(
        model=model,
        num_iterations=100
    )
    
    print(f"{backbone}:")
    print(f"  FPS: {benchmark_result['fps']:.1f}")
    print(f"  内存使用: {benchmark_result['memory_used_gb']:.2f} GB")
    print(f"  特征维度: {benchmark_result['feature_dim']}")
```

## 目录结构

```
onem_path/
├── __init__.py                          # 主入口文件
├── extractors/                           # 特征提取器
│   ├── __init__.py
│   ├── cellprofiler_extractor.py     # CellProfiler 特征提取器
│   └── titan_extractor.py             # TITAN 深度特征提取器
├── models/                               # 模型管理
│   ├── __init__.py
│   └── titan_model.py                 # TITAN 模型实现
├── config/                              # 配置管理
│   ├── __init__.py
│   └── settings.py                     # 配置类和预设
├── utils/                               # 工具函数
│   ├── __init__.py
│   ├── file_utils.py                   # 文件操作工具
│   ├── image_utils.py                  # 图像处理工具
│   ├── cellprofiler_utils.py            # CellProfiler 工具
│   └── titan_utils.py                  # TITAN 模型工具
├── example_usage.py                      # 使用示例
├── test_basic.py                        # 基础测试
└── README.md                           # 说明文档
```

## API 文档

### CellProfilerExtractor

```python
class CellProfilerExtractor:
    def __init__(self, config=None)
    
    def extract_features(self, image_path: str, output_path: str = None,
                        modules: List[str] = None) -> Dict
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             file_pattern: str = "*.jpg *.png *.tif *.tiff",
                             modules: List[str] = None, n_jobs: int = 1) -> pd.DataFrame
    
    def get_module_results(self, result: Dict) -> Dict
```

### TITANExtractor

```python
class TITANExtractor:
    def __init__(self, config=None)
    
    def extract_features(self, image_path: str, output_path: str = None) -> Dict
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame
    
    def extract_layer_features(self, image_path: str, layers: List[str] = None) -> Dict
    
    def benchmark_model(self, input_shape: Tuple = (3, 224, 224),
                       num_iterations: int = 100) -> Dict
```

### TITANModel

```python
class TITANModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 1024,
                 use_attention: bool = True)
    
    def extract_features(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor
    
    def get_model_info(self) -> Dict
```

### PathologyConfig

```python
class PathologyConfig:
    # 图像预处理参数
    resize_images: bool = True
    image_size: Tuple[int, int] = (224, 224)
    normalize_images: bool = True
    color_conversion: str = 'rgb'
    
    # CellProfiler 参数
    extract_cellprofiler_features: bool = True
    cellprofiler_modules: List[str] = ['morphological', 'texture', 'intensity', 'nuclei']
    min_object_size: int = 50
    
    # TITAN 参数
    extract_titan_features: bool = True
    titan_backbone: str = 'resnet50'
    titan_feature_dim: int = 1024
    titan_use_attention: bool = True
    titan_pretrained: bool = True
    
    # 性能参数
    device: str = 'auto'
    batch_size_titan: int = 32
    use_mixed_precision: bool = False
```

## 预设配置

### 可用预设配置

```python
from onem_path import PRESET_CONFIGS

# 默认配置 - 传统+深度特征平衡
config = PRESET_CONFIGS['default']

# 仅传统特征
config = PRESET_CONFIGS['cellprofiler_only']

# 仅深度特征
config = PRESET_CONFIGS['titan_only']

# 高质量配置
config = PRESET_CONFIGS['high_quality']

# 快速处理配置
config = PRESET_CONFIGS['fast_processing']

# 研究级配置
config = PRESET_CONFIGS['research_grade']

# GPU 加速配置
config = PRESET_CONFIGS['gpu_accelerated']

# 内存高效配置
config = PRESET_CONFIGS['memory_efficient']

# 综合配置
config = PRESET_CONFIGS['comprehensive']
```

### 预设配置对比

| 配置名称 | 传统特征 | 深度特征 | 骨干网络 | 特征维度 | 适用场景 |
|---------|---------|---------|---------|---------|---------|
| default | ✅ | ✅ | resnet50 | 1024 | 日常使用 |
| cellprofiler_only | ✅ | ❌ | - | - | 传统分析 |
| titan_only | ❌ | ✅ | resnet50 | 1024 | 深度学习 |
| high_quality | ✅ | ✅ | resnet101 | 2048 | 高质量分析 |
| fast_processing | ✅ | ❌ | - | - | 快速处理 |
| research_grade | ✅ | ✅ | resnet152 | 2048 | 研究用途 |
| gpu_accelerated | ❌ | ✅ | resnet50 | 1024 | GPU 批处理 |
| memory_efficient | ❌ | ✅ | efficientnet_b0 | 512 | 内存限制 |
| comprehensive | ✅ | ✅ | resnet101 | 2048 | 全面分析 |

## 支持的图像格式

### 输入格式
- **TIFF/TIFF**: 病理学常用的无损格式
- **PNG**: 支持透明通道
- **JPEG/JPG**: 有损压缩格式
- **BMP**: 位图格式
- **SVS**: Aperio 扫描仪格式（需要额外库）

### 输出格式
- **JSON**: 结构化数据，包含元数据
- **CSV**: 表格格式，便于数据分析
- **HDF5**: 大规模数据存储（需要 h5py）

## 特征类型详解

### 形态学特征
```python
# 细胞核形态学特征示例
morphological_features = {
    'num_objects': 125,                    # 对象数量
    'avg_object_area': 156.3,              # 平均面积
    'max_object_area': 892.1,              # 最大面积
    'avg_circularity': 0.73,               # 平均圆度
    'avg_solidity': 0.85,                 # 平均实体性
    'avg_eccentricity': 0.42,             # 平均离心率
    'binary_fill_ratio': 0.15               # 填充比率
}
```

### 纹理特征
```python
# GLCM 纹理特征示例
texture_features = {
    'glcm_contrast_mean': 0.234,           # 对比度均值
    'glcm_correlation_mean': 0.567,         # 相关性均值
    'glcm_homogeneity_mean': 0.891,         # 同质性均值
    'glcm_energy_mean': 0.145,              # 能量均值
    'lbp_energy': 0.067,                    # LBP 能量
    'lbp_entropy': 2.345,                    # LBP 熵
    'gabor_mean': 0.128,                     # Gabor 滤波响应均值
}
```

### 深度特征
```python
# TITAN 深度特征示例（1024维）
deep_features = {
    'titan_feature_0': 0.234,               # 第一个特征值
    'titan_feature_1': -0.567,              # 第二个特征值
    # ... 共 1024 个特征值
    'titan_feature_1023': 0.891,            # 最后一个特征值
}
```

## 性能优化

### 1. GPU 加速
```python
# GPU 配置
gpu_config = {
    'device': 'cuda',
    'batch_size_titan': 64,
    'num_workers_titan': 8,
    'use_mixed_precision': True
}

extractor = TITANExtractor(gpu_config)
```

### 2. 内存优化
```python
# 内存高效配置
memory_config = {
    'titan_backbone': 'efficientnet_b0',    # 轻量级骨干网络
    'titan_feature_dim': 512,              # 减少特征维度
    'batch_size_titan': 16,                  # 小批量大小
    'memory_limit_gb': 4.0                   # 内存限制
}
```

### 3. 批处理优化
```python
# 并行处理
df = extractor.extract_batch_features(
    image_dir="data/pathology",
    n_jobs=8,                              # 使用8个进程
    batch_size_titan=64                      # TITAN批量大小
)
```

## 测试

运行基础测试：

```bash
cd onem_path
python test_basic.py
```

运行示例：

```bash
python example_usage.py
```

## 常见问题

### Q: 如何选择合适的特征提取方法？
A: 根据研究目的选择：
- **传统方法**: 适合需要可解释性的研究
- **深度学习方法**: 适合需要高准确性的研究
- **联合方法**: 结合两者优势，获得最全面的信息

### Q: 如何处理不同染色方法？
A: 使用染色归一化：
```python
config = {
    'normalize_staining': True,
    'staining_method': 'reinhard'  # H&E 染色
}
preprocessed = preprocess_pathology_image(image_path, config)
```

### Q: 如何优化大规模数据处理？
A: 使用以下策略：
- GPU 加速：使用 `gpu_accelerated` 预设
- 内存管理：使用 `memory_efficient` 预设
- 批处理：调整 `batch_size_titan` 和 `n_jobs`
- 特征选择：减少特征数量

### Q: 如何自定义特征提取？
A: 可以通过配置参数：
```python
# 自定义 CellProfiler 模块
modules = ['morphological', 'texture', 'intensity', 'nuclei', 'cells']

# 自定义 TITAN 模型
config = {
    'titan_backbone': 'resnet101',    # 使用 ResNet-101
    'titan_feature_dim': 2048,     # 增加特征维度
    'titan_use_attention': True       # 启用注意力机制
}
```

### Q: 如何处理不同尺寸的病理图像？
A: 自动尺寸调整：
```python
config = {
    'resize_images': True,
    'image_size': (512, 512),     # 统一尺寸
    'normalize_intensity': True       # 强度归一化
}
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
- 实现基于 CellProfiler 的传统病理组学特征提取
- 实现 TITAN 深度迁移学习特征提取模型
- 支持多种骨干网络和注意力机制
- 实现完整的图像预处理流程
- 实现灵活的配置管理系统
- 提供多种预设配置模板
- 实现染色归一化和图像增强
- 支持批处理和并行计算
- 提供完整的示例和测试