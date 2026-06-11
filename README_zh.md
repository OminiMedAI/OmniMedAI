# OmniMedAI：全模态医学 AI 建模分析平台

![OmniMedAI logo](./logo.png)

OmniMedAI 是一个面向异构医学数据分析的模块化 AI 平台，目标是帮助科研人员和临床 AI 团队完成从医学数据处理、特征提取、多模态融合、模型训练、模型评估到报告输出的可复现工作流。

当前仓库更接近一个 Python SDK 形态的平台底座。README 中涉及的联邦学习、大规模临床部署、基因组学流水线、完整模型治理等能力属于平台路线图或集成方向，并非全部已经开箱即用。

![OmniMedAI platform overview](./platform-m-fusion-p.png)

## 平台工作流

OmniMedAI 的整体流程可以概括为：

```text
数据接入
  -> 医学预处理
  -> 分割与 ROI 准备
  -> 特征工程
  -> 多模态融合
  -> 模型构建
  -> 评估与解释
  -> 报告与协作
```

## 流程图

建议图片命名：

- `docs/assets/omnimedai-platform-workflow.png`：整体平台功能流程图。
- `docs/assets/omnimedai-module-workflow-map.png`：模块工作流矩阵图。

![OmniMedAI 整体平台功能流程图](./docs/assets/omnimedai-platform-workflow.png)

![OmniMedAI 模块工作流矩阵图](./docs/assets/omnimedai-module-workflow-map.png)

## 当前模块

| 平台能力 | 代码模块 | 当前状态 |
|---|---|---|
| 医学影像预处理、DICOM/NIfTI 转换、ROI 裁剪 | `onem_process` | 已有基础能力 |
| 基于图像和 mask 的影像组学特征提取 | `onem_radiomics` | 已有基础能力 |
| 2D/3D 自动 ROI 分割框架 | `onem_segment` | 框架已具备；真实推理需要预训练权重 |
| 数字病理图像特征提取 | `onem_path` | 已有基础能力；部分功能依赖 CellProfiler/TITAN |
| Habitat 与肿瘤异质性分析 | `onem_habitat` | 已有基础能力 |
| 深度学习模型组件 | `onem_torch` | 已有基础能力 |
| 多模态特征融合 | `onem_fusion` | 新增基础模块 |
| 特征表建模 | `onem_modeling` | 新增基础模块 |
| 模型评估与指标汇总 | `onem_eval` | 新增基础模块 |
| Notebook 教程与示例流程 | `onem_start` | 已有基础能力 |

## 核心能力

### 1. 数据接入与预处理

- 支持 DICOM、NIfTI 等医学影像数据的读取与转换。
- 支持 Z-score、Min-Max、百分位归一化、窗宽窗位等预处理方式。
- 支持重采样、裁剪、填充和 ROI 提取。
- 支持图像和掩码数据集的批量转换。

主要模块：`onem_process`

### 2. 分割与 ROI 准备

- 分析图像维度、切片数量、层间距和内容变化。
- 根据图像结构自动选择 2D 或 3D 分割流程。
- 提供模型加载、推理、后处理和 NIfTI mask 导出的统一接口。

主要模块：`onem_segment`

说明：该模块目前提供分割框架。要完成真实自动分割，需要兼容的模型定义和预训练权重。

### 3. 特征工程

- 基于 PyRadiomics 提取一阶统计、纹理、形态等影像组学特征。
- 从病理图像中提取传统图像特征和深度特征。
- 计算局部影像组学特征，并通过聚类进行 habitat / 瘤内异质性分析。
- 使用 `onem_torch` 中的深度学习模型组件进行分类或深度特征研究。

主要模块：`onem_radiomics`、`onem_path`、`onem_habitat`、`onem_torch`

### 4. 多模态融合

- 按患者或样本 ID 对齐多张特征表。
- 融合影像组学、病理组学、临床、基因、habitat 和深度特征。
- 自动添加模态前缀，避免特征名冲突。
- 生成可直接用于建模的融合特征表。

主要模块：`onem_fusion`

### 5. 模型构建

- 从提取或融合后的特征表训练基线模型。
- 支持常见分类和回归建模流程。
- 为影像组学、临床特征和多模态预测建模提供标准入口。

主要模块：`onem_modeling`

### 6. 评估与解释

- 计算准确率、F1、召回率、精确率、混淆矩阵和 AUC。
- 支持回归任务的 MAE、MSE 和 R2。
- 为外部验证、论文结果表和临床报告准备指标输出。

主要模块：`onem_eval`

后续计划包括校准曲线、决策曲线分析、置信区间、SHAP 汇总、Grad-CAM 接口和自动报告生成。

## 安装说明

当前项目以本地 Python 包方式组织，统一依赖文件后续会补齐。现阶段建议按使用场景安装依赖。

### 最小依赖

```bash
pip install numpy pandas scikit-learn
```

### 医学影像处理

```bash
pip install nibabel pydicom SimpleITK scipy opencv-python scikit-image
```

### 影像组学

```bash
pip install pyradiomics
```

### 数字病理

```bash
pip install Pillow scikit-image matplotlib
# 如果需要完整 CellProfiler 流程，可选安装：
pip install cellprofiler
```

### 深度学习

```bash
pip install torch torchvision monai
```

### Notebook 教程

```bash
pip install jupyter seaborn
```

## 快速开始

### 影像组学特征提取

```python
from onem_radiomics import RadiomicsExtractor, PRESET_CONFIGS

extractor = RadiomicsExtractor(PRESET_CONFIGS["ct_lung"])
result = extractor.extract_features(
    image_path="data/images/patient001.nii.gz",
    mask_path="data/masks/patient001_mask.nii.gz",
    patient_id="patient001"
)

print(len(result["features"]))
```

### 多模态特征融合

```python
from onem_fusion import FeatureFusion, FusionConfig

fusion = FeatureFusion(FusionConfig(
    id_column="patient_id",
    join_strategy="inner",
    modality_prefixes={
        "radiomics": "rad",
        "clinical": "clin",
        "pathology": "path"
    }
))

fused = fusion.fit_transform({
    "radiomics": "output/radiomics_features.csv",
    "clinical": "data/clinical.csv",
    "pathology": "output/pathology_features.csv"
})

fusion.save(fused, "output/fused_features.csv")
```

### 模型训练与评估

```python
from onem_modeling import ModelingConfig, train_tabular_model
from onem_eval import EvaluationConfig, ModelEvaluator

result = train_tabular_model(
    csv_path="output/fused_features.csv",
    label_column="label",
    config=ModelingConfig(task="classification", model_type="random_forest")
)

metrics = ModelEvaluator(EvaluationConfig(task="classification")).evaluate(
    y_true=result["y_test"],
    y_pred=result["predictions"],
    y_proba=result["probabilities"]
)

print(metrics["accuracy"])
```

## Notebook 教程

`onem_start` 目录提供了入门与示例工作流：

- `00_Quick_Start_Tutorial.ipynb`：平台快速上手。
- `01_Radiomics_Feature_Extraction.ipynb`：影像组学特征提取。
- `02_ROI_Segmentation.ipynb`：ROI 分割流程。
- `03_Pathology_Analysis.ipynb`：病理图像分析。
- `04_Comprehensive_Workflow.ipynb`：多模态综合流程。

## 路线图

### 近期计划

- 补充统一的 `requirements.txt` 或 `pyproject.toml`。
- 将根目录原型脚本整理为稳定 CLI 或模块入口。
- 增加轻量测试数据和可复现 demo 命令。
- 扩展 `onem_modeling`，加入交叉验证和特征选择。
- 扩展 `onem_eval`，加入校准曲线、决策曲线分析和置信区间。

### 平台扩展方向

- 临床表格处理和数据字典校验。
- 基因组学和分子特征接入。
- 生存分析、Cox 模型和时间结局验证。
- 高级多模态融合：早期融合、后期融合、多实例学习、注意力融合。
- 实验追踪和模型版本管理。
- 联邦学习与隐私保护的多中心建模。
- 自动化临床研究报告生成。

## 临床与科研使用说明

OmniMedAI 面向科研和转化开发。若用于临床场景，需要完成本地验证、伦理与合规审查、隐私治理和机构级质量控制。

建议实践：

- 使用脱敏数据。
- 严格区分训练集、验证集、内部测试集和外部测试集。
- 记录预处理参数和特征提取配置。
- 在不同扫描仪、不同中心和不同患者亚组上验证模型。
- 面向临床决策支持时，应报告校准度、决策曲线收益和置信区间。

## 开发团队与合作机构

OmniMedAI 是一个围绕多模态医学数据分析、精准医疗、影像生物标志物、病理 AI 和隐私保护学习展开的合作项目。

### 核心开发与临床转化

- 复旦大学：中国，多模态数据融合与计算病理。
- 复旦大学附属中山医院：中国，临床研究设计、队列验证与转化评估。
- 蚌埠医科大学：中国，医学 AI 工作流开发与教学。
- 蚌埠医科大学第一附属医院：中国，临床场景验证与数据集建设。
- 东南大学：中国，医学影像分析与 AI 算法优化。
- 浙江省人民医院：中国，肿瘤数据集整理与临床验证。
- 西京医院：中国，高级影像生物标志物研究。
- 上海交通大学：中国，深度学习模型开发。
- 哈尔滨工业大学：中国，隐私保护与联邦学习研究。
- 中北大学：中国，信号处理与硬件集成。
- 山西大学：中国，基因组学与生物信息学分析。
- 安徽科技学院：中国，边缘计算与应用 AI 工作流。
- 中国电子科技集团公司第四十一研究所：中国，精准医疗设备研发。
- 江苏省人民医院：中国，临床数据验证与慢病研究。
- 西安电子科技大学：中国，通信技术与嵌入式系统集成。
- 武汉同济医院：中国，大规模临床队列研究与疗效评估。

### 国际合作伙伴

- 都柏林大学学院，爱尔兰：放射基因组学与转化研究。
- 阿德莱德大学，澳大利亚：可解释 AI 与临床决策支持。

### 合作重点

- 联合临床医学、医学影像、病理学、工程学和生物信息学进行跨学科开发。
- 基于多中心队列开展外部验证与泛化能力评估。
- 推动从科研原型到规范化临床 AI 工作流的转化。

## 关键参考文献

以下文献和研究方向为平台架构、应用场景和路线图提供了参考。

### 影像组学与深度学习

1. Chen et al. (2022). MRI radiomics for mucosal healing prediction in Crohn's
   disease. *Journal of Gastroenterology*.
2. Shen et al. (2025). Correlation of MRI characteristics with KRAS mutation
   status in pancreatic ductal adenocarcinoma. *Abdominal Radiology (NY)*.
3. Shen et al. (2025). Contrast-enhanced MRI-based intratumoral heterogeneity
   assessment for predicting lymph node metastasis in pancreatic ductal
   adenocarcinoma. *Insights into Imaging*.
4. Li et al. (2024). SMAD4-mutated pancreatic ductal adenocarcinoma
   identification using preoperative MRI and clinical data. *BMC Medical
   Imaging*.
5. Li et al. (2024). Preoperative prediction of vasculogenic mimicry in lung
   adenocarcinoma using CT radiomics. *Clinical Radiology*.

### 病理组学与全切片图像

1. Liu et al. (2020). Automated glioma subtyping via MRI radiomics.
   *Neuro-Oncology*.
2. Yang et al. (2020). Preoperative cervical cancer radiomics.
   *European Radiology*.

### 技术方向

1. Liu et al. (2024). Deep learning-reconstructed ultra-fast
   respiratory-triggered T2-weighted liver MRI. *Magnetic Resonance Imaging*.
2. Huang et al. (2021). Federated learning for privacy-preserving medical AI.
   *Gut*.
3. Zhang et al. (2022). Multi-instance learning for robust diagnostic modeling.
   *IEEE Journal of Biomedical and Health Informatics*.
4. Guo et al. (2021). MRI radiomics and VAE latent-space analysis for HCC immune
   subtyping.

说明：以上参考文献保留并整理自项目原始说明。正式论文、产品文档或申报材料中引用前，建议进一步核对作者、题名、期刊、年份和 DOI。

## 联系方式

专业版或合作咨询请联系：

- Email: <acezqy@gmail.com>
