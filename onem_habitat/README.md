# onem_habitat

医学影像放射组学生态分析模块，实现 ROI 区域局部特征计算和基于聚类的 mask 重新划分。

## 功能特性

### 1. 局部放射组学特征提取
- **5x5x5 3D 区域特征提取**: 为每个 ROI 体素计算局部放射组学特征
- **多种特征类型**: 支持一阶统计、GLCM、GLRLM、GLSZM、GLTDM、NGTDM 等特征
- **灵活配置**: 可配置核大小、分箱宽度、重采样参数等
- **并行处理**: 支持多线程并行计算

### 2. 特征聚类分析
- **多种聚类算法**: K-means、DBSCAN、层次聚类
- **特征选择**: 方差筛选、相关性分析、PCA 降维
- **聚类评估**: 轮廓系数、Calinski-Harabasz 指数等质量指标
- **可视化支持**: 2D/3D 聚类结果可视化

### 3. 基于 Mask 重新划分
- **聚类驱动**: 基于特征聚类结果重新划分 mask 区域
- **形态学优化**: 连通分量分析、孔洞填充、边界平滑
- **多级处理**: 小聚类过滤、噪声点分配、空间紧密度优化
- **质量控制**: 覆盖度、平衡性、连通性评估

### 4. 配置管理
- **灵活配置**: 完整的参数配置系统
- **预设模板**: CT 肺癌、MRI 脑瘤、PET 肿瘤等预设配置
- **验证机制**: 自动配置验证和警告提示
- **导入导出**: 配置模板的保存和加载

## 安装依赖

```bash
# 核心依赖
pip install numpy nibabel

# 放射组学特征提取
pip install pyradiomics SimpleITK

# 聚类和可视化
pip install scikit-learn matplotlib seaborn

# 图像处理和形态学
pip install scikit-image

# 开发和测试
pip install unittest
```

## 快速开始

### 1. 局部放射组学特征提取

```python
from onem_habitat.radiomics import LocalRadiomicsExtractor

# 创建特征提取器
extractor = LocalRadiomicsExtractor(
    kernel_size=(5, 5, 5),           # 5x5x5 核
    feature_types=['firstorder', 'glcm'],  # 特征类型
    bin_width=25,                      # 直方图分箱
    n_jobs=4                            # 并行处理
)

# 提取单个图像的特征
result = extractor.extract_local_features(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    output_path="features/patient001_features.npy",
    step_size=2  # 每2个体素提取一次，减少计算量
)

print(f"提取了 {result['metadata']['n_voxels']} 个体素的特征")
print(f"特征类型: {result['metadata']['feature_types']}")
```

### 2. 批量特征提取

```python
# 批量提取
results = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks",
    output_dir="output/features",
    file_pattern="*.nii.gz",
    step_size=3
)

print(f"成功处理 {len([r for r in results if 'error' not in r])} 个图像")
```

### 3. 特征聚类分析

```python
from onem_habitat.clustering import FeatureClustering

# 加载特征数据
features_dict_list = []
for result_file in Path("features").glob("*_features.npy"):
    features_dict = np.load(result_file, allow_pickle=True).item()
    features_dict_list.append(features_dict)

# 创建聚类器
clusterer = FeatureClustering(
    clustering_method='kmeans',
    n_clusters=4,
    feature_selection='variance',    # 基于方差的特征选择
    pca_components=10,               # PCA 降维
    standardize=True
)

# 执行聚类
cluster_labels_list = clusterer.fit_predict(features_dict_list)

# 聚类可视化
clusterer.visualize_clusters(
    features=clusterer._prepare_features(features_dict_list)[0],
    labels=cluster_labels_list[0],
    output_path="output/cluster_visualization.png"
)
```

### 4. 基于 Mask 重新划分

```python
from onem_habitat.segmentation import MaskRefiner

# 创建精细器
refiner = MaskRefiner(
    min_cluster_size=50,      # 最小聚类大小
    smoothing_iterations=2,     # 平滑迭代
    connectivity=1,           # 6邻域连通性
    fill_holes=True          # 填充孔洞
)

# 重新划分 mask
saved_files = refiner.refine_masks(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    cluster_labels=cluster_labels_list[0],
    coordinates=features_dict_list[0]['coordinates'],
    output_dir="output/refined_masks",
    save_individual=True,      # 保存单独的聚类掩码
    save_combined=True         # 保存合并的精细掩码
)

print(f"保存了 {len([k for k in saved_files.keys() if k.startswith('cluster_')])} 个聚类掩码")
```

### 5. 完整工作流程

```python
from onem_habitat.config import PRESET_CONFIGS

# 使用预设配置
config = PRESET_CONFIGS['ct_lung']

# 步骤 1: 特征提取
extractor = LocalRadiomicsExtractor(
    kernel_size=config.kernel_size,
    feature_types=config.feature_types,
    n_jobs=config.extraction_n_jobs
)
extraction_results = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks", 
    output_dir="output/features"
)

# 步骤 2: 特征聚类
clusterer = FeatureClustering(
    clustering_method=config.clustering_method,
    n_clusters=config.n_clusters,
    feature_selection=config.feature_selection
)
cluster_labels_list = clusterer.fit_predict([
    np.load(r['output_file'], allow_pickle=True).item()
    for r in extraction_results if 'error' not in r
])

# 步骤 3: 掩码精细划分
refiner = MaskRefiner(
    min_cluster_size=config.min_cluster_size,
    smoothing_iterations=config.smoothing_iterations
)
refinement_results = refiner.batch_refine_masks(
    images_dir="data/images",
    masks_dir="data/masks",
    features_dir="output/features",
    clustering_results_dir="output/clustering", 
    output_dir="output/refined_masks"
)

print("✅ 完整生态分析工作流程完成!")
```

## 目录结构

```
onem_habitat/
├── __init__.py              # 主入口文件
├── radiomics/                # 放射组学特征提取
│   ├── __init__.py
│   └── local_extractor.py   # 局部特征提取器
├── clustering/               # 特征聚类分析
│   ├── __init__.py
│   └── feature_clustering.py # 特征聚类器
├── segmentation/             # 掩码精细划分
│   ├── __init__.py
│   └── mask_refiner.py     # 掩码精细器
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── habitat_utils.py    # 生态分析工具
│   └── radiomics_utils.py  # 放射组学工具
├── config/                  # 配置管理
│   ├── __init__.py
│   └── settings.py         # 配置类和预设
├── example_usage.py          # 使用示例
├── test_basic.py           # 基础测试
└── README.md               # 说明文档
```

## API 文档

### LocalRadiomicsExtractor

```python
class LocalRadiomicsExtractor:
    def __init__(self, kernel_size=(5,5,5), feature_types=None, 
                 bin_width=25, resampled_pixel_spacing=None, 
                 interpolator='sitkBSpline', weight_center=None,
                 weight_radius=None, n_jobs=1)
    
    def extract_local_features(self, image_path, mask_path, output_path=None,
                           coordinates=None, step_size=1) -> Dict
    def batch_extract_features(self, images_dir, masks_dir, output_dir,
                           file_pattern="*.nii.gz", step_size=1) -> List[Dict]
    def get_feature_summary(self, features_dict) -> Dict
    def load_features(self, features_path) -> Dict
```

### FeatureClustering

```python
class FeatureClustering:
    def __init__(self, clustering_method='kmeans', n_clusters=5,
                 feature_selection='all', pca_components=None,
                 standardize=True, random_state=42)
    
    def fit_predict(self, features_dict_list) -> List[np.ndarray]
    def visualize_clusters(self, features, labels, output_path=None,
                        method='tsne') -> None
    def analyze_cluster_characteristics(self, features_dict_list, 
                                   cluster_labels_list) -> Dict
    def save_clustering_results(self, features_dict_list, cluster_labels_list,
                             output_dir, prefix="clustering") -> Dict
```

### MaskRefiner

```python
class MaskRefiner:
    def __init__(self, min_cluster_size=50, smoothing_iterations=2,
                 connectivity=1, fill_holes=True, noise_cluster_id=-1)
    
    def refine_masks(self, image_path, mask_path, cluster_labels,
                   coordinates, output_dir, save_individual=True,
                   save_combined=True) -> Dict
    def batch_refine_masks(self, images_dir, masks_dir, features_dir,
                           clustering_results_dir, output_dir) -> List[Dict]
    def evaluate_refinement_quality(self, original_mask, refined_masks,
                                  cluster_labels) -> Dict
```

## 配置选项

### HabitatConfig 主要参数

```python
# 特征提取配置
kernel_size: Tuple[int, int, int] = (5, 5, 5)     # 3D 核大小
feature_types: List[str] = ['firstorder', 'glcm']    # 特征类型
bin_width: int = 25                                   # 直方图分箱宽度
extraction_n_jobs: int = 1                              # 并行处理数
step_size: int = 1                                    # 提取步长

# 聚类配置
clustering_method: str = 'kmeans'                     # 聚类算法
n_clusters: int = 5                                   # 聚类数量
feature_selection: str = 'all'                           # 特征选择
pca_components: Optional[int] = None                     # PCA 降维
standardize: bool = True                                # 特征标准化

# 掩码精细化配置
min_cluster_size: int = 50                             # 最小聚类大小
smoothing_iterations: int = 2                           # 平滑迭代次数
connectivity: int = 1                                   # 连通性
fill_holes: bool = True                                # 填充孔洞
```

## 预设配置模板

```python
from onem_habitat.config import PRESET_CONFIGS

# CT 肺癌分析
ct_lung_config = PRESET_CONFIGS['ct_lung']

# MRI 脑瘤分析  
mri_brain_config = PRESET_CONFIGS['mri_brain']

# PET 肿瘤分析
pet_tumor_config = PRESET_CONFIGS['pet_tumor']

# 高分辨率分析
high_res_config = PRESET_CONFIGS['high_resolution']

# 快速处理
fast_config = PRESET_CONFIGS['fast_processing']
```

## 性能优化

### 1. 并行处理
```python
# 特征提取并行化
extractor = LocalRadiomicsExtractor(n_jobs=4)

# 批量处理优化
results = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks", 
    step_size=2  # 减少计算量
)
```

### 2. 内存优化
```python
# 使用步长减少特征计算
extractor.extract_local_features(
    image_path="image.nii.gz",
    mask_path="mask.nii.gz",
    step_size=3  # 每3个体素计算一次
)

# 特征降维减少内存占用
clusterer = FeatureClustering(pca_components=10)
```

### 3. 特征选择
```python
# 方差筛选减少特征数量
clusterer = FeatureClustering(
    feature_selection='variance',
    n_clusters=5
)

# 相关性分析去除冗余特征
clusterer = FeatureClustering(
    feature_selection='correlation'
)
```

## 测试

运行基础测试：

```bash
cd onem_habitat
python test_basic.py
```

运行示例：

```bash
python example_usage.py
```

## 输出文件格式

### 1. 特征文件 (.npy)
```python
{
    'features': {
        'feature1': [value1, value2, ...],
        'feature2': [value1, value2, ...],
        ...
    },
    'coordinates': [(z, y, x), (z, y, x), ...],
    'metadata': {
        'image_path': 'path/to/image.nii.gz',
        'mask_path': 'path/to/mask.nii.gz',
        'n_voxels': 1000,
        'feature_types': ['firstorder', 'glcm']
    }
}
```

### 2. 聚类文件 (.npy)
```python
# clustering_labels_patient001.npy
[cluster_id, cluster_id, cluster_id, ...]  # 每个体素的聚类标签
```

### 3. 精细掩码文件 (.nii.gz)
```
cluster_0_mask.nii.gz  # 聚类0的掩码
cluster_1_mask.nii.gz  # 聚类1的掩码
...
refined_combined_mask.nii.gz  # 合并的精细掩码
```

## 常见问题

### Q: 如何选择合适的核大小？
A: 核大小取决于图像分辨率和目标特征尺度：
- 高分辨率图像：3x3x3 或 5x5x5
- 中等分辨率：5x5x5 或 7x7x7  
- 低分辨率：7x7x7 或 9x9x9

### Q: 如何确定最佳聚类数量？
A: 可以使用以下方法：
1. 肘部法则：观察轮廓系数随聚类数量的变化
2. 领域知识：根据医学先验知识确定
3. 轮廓系数：选择最大轮廓系数对应的聚类数

### Q: 处理大数据集时内存不足怎么办？
A: 可以采用以下策略：
1. 增加步长：step_size=2 或 3
2. 特征选择：减少特征类型数量
3. PCA 降维：pca_components=10-20
4. 分批处理：按区域或切片分批处理

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个模块。

## 更新日志

### v1.0.0
- 初始版本发布
- 实现局部放射组学特征提取功能
- 实现多种聚类算法
- 实现基于聚类的掩码重新划分
- 添加完整的配置管理系统
- 提供多种预设配置模板