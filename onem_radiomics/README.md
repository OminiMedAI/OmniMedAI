# onem_radiomics

医学影像放射组学特征提取模块 - 从 NIfTI 格式的图像和掩码中提取全面的放射组学特征，并保存到 CSV 文件。

## 功能特性

### 1. 全面的特征提取
- **一阶统计特征**: 均值、方差、偏度、峰度等描述性统计量
- **纹理特征**: GLCM、GLRLM、GLSZM、GLTDM、NGTDM 等纹理特征
- **形态特征**: 体积、表面积、紧致度等形状描述子
- **灵活配置**: 可选择提取的特征类型和参数设置

### 2. 图像预处理
- **重采样**: 支持自定义像素间距重采样
- **插值方法**: 多种插值方法（最近邻、线性、B样条等）
- **归一化**: 可选的强度值归一化处理
- **分箱**: 可配置的直方图分箱宽度

### 3. 批量处理能力
- **并行处理**: 支持多线程并行特征提取
- **批量匹配**: 自动匹配图像和掩码文件
- **进度跟踪**: 详细的处理进度和日志记录
- **错误处理**: 健壮的错误处理和恢复机制

### 4. 配置管理
- **预设配置**: 针对不同影像模态的预设配置（CT、MRI、PET等）
- **自定义配置**: 灵活的参数配置和验证
- **配置保存**: 配置参数的保存和加载
- **参数验证**: 自动参数有效性验证

## 安装依赖

```bash
# 核心依赖
pip install numpy pandas

# 医学图像处理
pip install nibabel SimpleITK

# 放射组学特征提取
pip install pyradiomics

# 可视化和分析（可选）
pip install matplotlib seaborn

# 开发和测试
pip install unittest
```

## 快速开始

### 1. 基础特征提取

```python
from onem_radiomics import RadiomicsExtractor, RadiomicsConfig

# 创建默认配置的提取器
extractor = RadiomicsExtractor()

# 提取单个图像的特征
result = extractor.extract_features(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    patient_id="patient001"
)

print(f"成功提取 {len(result['features'])} 个特征")
print("示例特征:")
for feature_name, value in list(result['features'].items())[:5]:
    print(f"  {feature_name}: {value:.4f}")
```

### 2. 批量特征提取

```python
# 批量提取并保存到 CSV
df = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks", 
    output_csv_path="output/radiomics_features.csv",
    file_pattern="*.nii.gz",
    n_jobs=4  # 使用4个并行进程
)

print(f"成功处理 {len(df)} 个患者的数据")
print(f"提取的特征数量: {len(df.columns) - 3}")  # 减去ID和路径列
```

### 3. 使用预设配置

```python
from onem_radiomics import PRESET_CONFIGS

# 使用肺癌CT预设配置
config = PRESET_CONFIGS['ct_lung']
extractor = RadiomicsExtractor(config)

print(f"特征类型: {config.feature_types}")
print(f"分箱宽度: {config.bin_width}")
print(f"重采样像素间距: {config.resampled_pixel_spacing}")
```

### 4. 自定义配置

```python
from onem_radiomics import RadiomicsConfig, create_custom_config

# 创建自定义配置
custom_config = create_custom_config(
    feature_types=['firstorder', 'glcm', 'glrlm'],
    bin_width=16,
    resampled_pixel_spacing=(1.5, 1.5, 3.0),
    interpolator='sitkLinear',
    normalize=True,
    n_jobs=8
)

extractor = RadiomicsExtractor(custom_config)

# 保存配置
custom_config.save("custom_config.json")

# 从文件加载配置
loaded_config = RadiomicsConfig.load("custom_config.json")
```

### 5. 特征分析和报告

```python
from onem_radiomics.utils.radiomics_utils import (
    find_constant_features, 
    find_highly_correlated_features,
    create_feature_selection_report
)

# 查找常量特征（零方差）
constant_features = find_constant_features(df)
print(f"常量特征数量: {len(constant_features)}")

# 查找高度相关的特征对
high_corr_pairs = find_highly_correlated_features(df, threshold=0.95)
print(f"高度相关特征对数量: {len(high_corr_pairs)}")

# 生成特征选择报告
report = create_feature_selection_report(df)
with open("feature_analysis_report.txt", "w") as f:
    f.write(report)
```

## 目录结构

```
onem_radiomics/
├── __init__.py                    # 主入口文件
├── extractors/                     # 特征提取器
│   ├── __init__.py
│   └── radiomics_extractor.py     # 放射组学特征提取器
├── config/                         # 配置管理
│   ├── __init__.py
│   └── settings.py                # 配置类和预设
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── file_utils.py              # 文件操作工具
│   └── radiomics_utils.py        # 放射组学工具
├── example_usage.py                # 使用示例
├── test_basic.py                   # 基础测试
└── README.md                      # 说明文档
```

## API 文档

### RadiomicsExtractor

```python
class RadiomicsExtractor:
    def __init__(self, config: Optional[RadiomicsConfig] = None)
    
    def extract_features(self, image_path: str, mask_path: str, 
                        patient_id: Optional[str] = None) -> Dict
    
    def batch_extract_features(self, images_dir: str, masks_dir: str,
                            output_csv_path: str, file_pattern: str = "*.nii.gz",
                            n_jobs: int = 1) -> pd.DataFrame
    
    def get_feature_descriptions(self) -> Dict[str, str]
    
    def validate_extraction_setup(self, test_image_path: str, 
                                 test_mask_path: str) -> Dict
```

### RadiomicsConfig

```python
class RadiomicsConfig:
    # 图像预处理参数
    resampled_pixel_spacing: Optional[Tuple[float, float, float]]
    interpolator: str = 'sitkBSpline'
    bin_width: int = 25
    normalize: bool = False
    normalize_scale: int = 100
    
    # 特征选择
    feature_types: List[str] = [
        'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm'
    ]
    
    # 处理参数
    n_jobs: int = 1
    verbose: bool = True
    
    def validate(self)
    def update(self, **kwargs)
    def save(self, file_path: str)
    @classmethod
    def load(cls, file_path: str)
```

## 预设配置

### 可用预设配置

```python
from onem_radiomics import PRESET_CONFIGS

# 标准配置
config = PRESET_CONFIGS['standard']

# 全面配置（包含所有特征类型）
config = PRESET_CONFIGS['comprehensive']

# CT 肺癌
config = PRESET_CONFIGS['ct_lung']

# MRI 脑肿瘤
config = PRESET_CONFIGS['mri_brain']

# PET 肿瘤
config = PRESET_CONFIGS['pet_tumor']

# 快速处理（较少特征）
config = PRESET_CONFIGS['fast_processing']

# 高质量处理（精细参数）
config = PRESET_CONFIGS['high_quality']

# 纹理特征专注
config = PRESET_CONFIGS['texture_focused']

# 形状特征专注
config = PRESET_CONFIGS['shape_focused']

# 研究级配置
config = PRESET_CONFIGS['research']
```

### 预设配置参数对比

| 配置名称 | 特征类型 | 分箱宽度 | 重采样间距 | 归一化 | 用途 |
|---------|---------|---------|-----------|--------|------|
| standard | 4类基础 | 25 | - | 否 | 日常使用 |
| comprehensive | 7类全部 | 16 | - | 是 | 研究分析 |
| ct_lung | 5类纹理 | 25 | (1,1,1) | 否 | CT肺癌 |
| mri_brain | 4类基础 | 16 | (1,1,1) | 是 | MRI脑瘤 |
| pet_tumor | 6类纹理 | 32 | (4,4,4) | 否 | PET肿瘤 |
| fast_processing | 2类基础 | 50 | (2,2,2) | 否 | 快速处理 |
| high_quality | 7类全部 | 8 | (0.5,0.5,0.5) | 是 | 高质量 |

## 文件命名约定

### 图像-掩码匹配规则

模块支持多种图像和掩码文件的命名约定：

1. **后缀匹配**: `patient001.nii.gz` ↔ `patient001_mask.nii.gz`
2. **分割后缀**: `patient001.nii.gz` ↔ `patient001_seg.nii.gz`
3. **相同名称**: `patient001.nii.gz` ↔ `patient001.nii.gz`（不同目录）
4. **模态后缀**: `patient001_CT.nii.gz` ↔ `patient001_mask.nii.gz`
5. **通用后缀**: `patient001.nii.gz` ↔ `patient001_{mask,seg,roi,label}.nii.gz`

### 目录结构示例

```
data/
├── images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── patient003.nii.gz
└── masks/
    ├── patient001_mask.nii.gz
    ├── patient002_mask.nii.gz
    └── patient003_mask.nii.gz
```

## 输出格式

### CSV 文件结构

生成的 CSV 文件包含以下列：

- **patient_id**: 患者标识符
- **image_path**: 图像文件路径
- **mask_path**: 掩码文件路径
- **特征列**: 所有提取的放射组学特征
- **元数据列**: 以 `meta_` 开头的元数据信息

### 示例 CSV 输出

```csv
patient_id,image_path,mask_path,Original_Firstorder_Mean,Original_GLCM_Correlation,...,meta_feature_types
patient001,/path/to/img1.nii.gz,/path/to/mask1.nii.gz,45.23,0.78,...,"['firstorder','glcm']"
patient002,/path/to/img2.nii.gz,/path/to/mask2.nii.gz,38.91,0.65,...,"['firstorder','glcm']"
```

### 特征摘要文件

还会生成一个同名的 `_summary.txt` 文件，包含：

- 处理的患者总数
- 提取的特征总数
- 使用的特征类型
- 提取参数设置

## 性能优化

### 1. 并行处理

```python
# 使用多进程加速处理
extractor = RadiomicsExtractor(config)
df = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks",
    output_csv_path="output/features.csv",
    n_jobs=8  # 使用8个CPU核心
)
```

### 2. 内存优化

```python
# 减少特征类型降低内存使用
config = RadiomicsConfig(feature_types=['firstorder'])

# 使用较大的分箱宽度减少特征数量
config.bin_width = 50

# 降低重采样分辨率
config.resampled_pixel_spacing = (2.0, 2.0, 2.0)
```

### 3. 处理时间优化

```python
# 快速处理预设
config = PRESET_CONFIGS['fast_processing']

# 或者自定义快速配置
fast_config = create_custom_config(
    feature_types=['firstorder'],  # 只使用一阶特征
    bin_width=100,                 # 大分箱宽度
    resampled_pixel_spacing=(3, 3, 3),  # 低分辨率
    n_jobs=8                      # 高并行度
)
```

## 测试

运行基础测试：

```bash
cd onem_radiomics
python test_basic.py
```

运行示例：

```bash
python example_usage.py
```

## 常见问题

### Q: 如何选择合适的分箱宽度？
A: 分箱宽度的选择取决于：
- **数据动态范围**: 范围大时使用较大值（25-50）
- **噪声水平**: 噪声大时使用较大值减少噪声影响
- **应用需求**: 需要精细纹理时使用较小值（8-16）

### Q: 哪些特征类型最常用？
A: 根据应用场景：
- **基础研究**: `firstorder` + `glcm` + `glrlm`
- **全面分析**: 添加 `glszm` + `gldm`
- **形态学研究**: 添加 `shape` 特征
- **高级纹理**: 添加 `ngtdm`

### Q: 如何处理数据缺失？
A: 模块会：
- 自动跳过无法加载的图像-掩码对
- 将 NaN/Inf 值转换为 None
- 在报告中记录缺失数据的统计信息

### Q: 如何减少特征冗余？
A: 可以使用以下策略：
```python
# 查找常量特征
constant_features = find_constant_features(df)

# 查找高度相关特征
high_corr_pairs = find_highly_correlated_features(df, threshold=0.95)

# 生成详细的特征分析报告
report = create_feature_selection_report(df)
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
- 实现全面的放射组学特征提取功能
- 支持多种特征类型和预处理选项
- 实现批量处理和并行计算
- 添加完整的配置管理系统
- 提供多种预设配置模板
- 实现文件工具和特征分析工具
- 提供完整的示例和测试