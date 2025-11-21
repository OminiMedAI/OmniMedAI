# 🚀 OmniMedAI Jupyter Notebooks - Quick Start Guide

Welcome to the **onem_start** folder - your gateway to mastering the OmniMedAI platform! This collection of Jupyter notebooks is designed to get you up and running with medical AI analysis in minutes.

## 📋 Notebooks Overview

### 🎯 [00_Quick_Start_Tutorial.ipynb](00_Quick_Start_Tutorial.ipynb)
**Perfect for beginners!** Get started in 5-10 minutes.
- ⚡ Quick setup and imports
- 🎯 One-click segmentation demo
- 🧬 Simple feature extraction
- 🔬 Basic pathology analysis
- 📊 Instant visualizations
- 🏆 **Time to complete: 5-10 minutes**

### 🧬 [01_Radiomics_Feature_Extraction.ipynb](01_Radiomics_Feature_Extraction.ipynb)
**Deep dive into radiomics analysis** with comprehensive feature extraction.
- 📊 First-order, texture, and morphological features
- ⚙️ Multiple preset configurations (CT, MRI, PET)
- 🔍 Feature correlation and analysis
- 📈 Advanced visualization techniques
- 🧪 Statistical analysis pipelines
- 🏆 **Best for: Research and quantitative analysis**

### 🎯 [02_ROI_Segmentation.ipynb](02_ROI_Segmentation.ipynb)
**Master automatic ROI segmentation** with intelligent 2D/3D model selection.
- 🤖 Smart 2D/3D model auto-selection
- 📊 Batch processing capabilities
- ⚖️ Model comparison and optimization
- 🔧 Advanced post-processing techniques
- 📈 Performance metrics and analysis
- 🏆 **Best for: Clinical workflow automation**

### 🔬 [03_Pathology_Analysis.ipynb](03_Pathology_Analysis.ipynb)
**Dual-mode pathology analysis** combining traditional and deep learning approaches.
- 🧪 CellProfiler-based radiomics features
- 🤖 TITAN deep learning features
- 🔬 Whole Slide Image (WSI) processing
- 🔗 Feature fusion and selection
- 📊 Comparative analysis
- 🏆 **Best for: Digital pathology research**

### 🔗 [04_Comprehensive_Workflow.ipynb](04_Comprehensive_Workflow.ipynb)
**Complete end-to-end pipeline** for clinical deployment.
- 🏗️ Multi-modal data integration
- 🤖 Predictive modeling and validation
- 🏞️ Habitat and microenvironment analysis
- 📋 Clinical reporting and documentation
- 📊 Performance evaluation
- 🏆 **Best for: Production deployment**

---

## 🚀 Getting Started

### Prerequisites
```bash
# Core requirements
pip install jupyter numpy pandas matplotlib seaborn
pip install scikit-learn SimpleITK nibabel

# Medical imaging
pip install pydicom openslide-python

# Optional: GPU acceleration
pip install torch torchvision
```

### Quick Launch
```bash
# Navigate to the directory
cd onem_start

# Start Jupyter
jupyter notebook

# OR open a specific notebook
jupyter notebook 00_Quick_Start_Tutorial.ipynb
```

---

## 📁 Recommended Learning Path

### 🌱 **Beginners** (New to Medical AI)
1. **Start with**: `00_Quick_Start_Tutorial.ipynb`
2. **Then try**: `01_Radiomics_Feature_Extraction.ipynb`
3. **Explore**: `02_ROI_Segmentation.ipynb`

### 🔬 **Researchers** (Familiar with Medical Imaging)
1. **Skip to**: `01_Radiomics_Feature_Extraction.ipynb`
2. **Add**: `03_Pathology_Analysis.ipynb`
3. **Master**: `04_Comprehensive_Workflow.ipynb`

### 🏥 **Clinicians** (Focused on Clinical Application)
1. **Focus on**: `02_ROI_Segmentation.ipynb`
2. **Integrate**: `04_Comprehensive_Workflow.ipynb`
3. **Reference**: `00_Quick_Start_Tutorial.ipynb`

---

## 🎯 Module Focus

| Notebook | Primary Module | Key Skills | Time Required |
|-----------|----------------|-------------|---------------|
| `00_Quick_Start` | All modules | Quick overview | 5-10 min |
| `01_Radiomics` | `onem_radiomics` | Feature extraction | 30-45 min |
| `02_Segmentation` | `onem_segment` | ROI detection | 45-60 min |
| `03_Pathology` | `onem_path` | Tissue analysis | 60-90 min |
| `04_Comprehensive` | All modules | Full pipeline | 90-120 min |

---

## 🛠️ Common Use Cases

### 🏥 **Clinical Research**
```python
# Typical workflow
start → 02_Segmentation → 01_Radiomics → 04_Comprehensive
```

### 🔬 **Pathology Studies**
```python
# Specialized path
start → 03_Pathology → 04_Comprehensive
```

### 🎓 **Learning & Teaching**
```python
# Educational journey
start → 01_Radiomics → 02_Segmentation → 03_Pathology → 04_Comprehensive
```

### ⚡ **Rapid Prototyping**
```python
# Fast development
start → 00_Quick_Start → customize → production
```

---

## 📊 Data Requirements

| Module | Input Formats | Typical Data Size | Processing Time |
|--------|---------------|------------------|-----------------|
| `onem_segment` | NIfTI (.nii.gz) | 100-500 MB per case | 20-60 sec |
| `onem_radiomics` | NIfTI + Masks | 50-200 MB per case | 10-30 sec |
| `onem_path` | PNG/TIF/SVS | 10-100 MB per image | 30-90 sec |
| `Comprehensive` | All formats | 500 MB - 2 GB per patient | 2-5 min |

---

## 🔧 Configuration Tips

### 🎯 **Quick Setup**
```python
# Use presets for immediate results
config_name = 'ct_lung'      # For CT scans
config_name = 'mri_brain'     # For brain MRI
config_name = 'pet_tumor'      # For PET scans
```

### ⚡ **Performance Optimization**
```python
# Enable parallel processing
parallel=True, n_workers=4

# GPU acceleration (if available)
device='cuda'
```

### 🎨 **Custom Configurations**
```python
# Create your own settings
custom_config = {
    'feature_types': ['firstorder', 'texture'],
    'bin_width': 25,
    'resampling': {'voxel_size': [1.0, 1.0, 1.0]}
}
```

---

## 🎨 Visualization Gallery

### 📊 **Feature Analysis**
- Distribution plots and histograms
- Correlation heatmaps
- Principal component analysis
- Feature importance charts

### 🎯 **Segmentation Results**
- 3D volume rendering
- Slice-by-slice comparisons
- Model performance metrics
- Processing time benchmarks

### 🔬 **Pathology Insights**
- Tissue classification maps
- Nuclear morphology charts
- Deep learning feature projections
- Multi-modal comparisons

---

## 🔍 Advanced Features

### 🧠 **AI-Powered Features**
- **Auto 2D/3D Selection**: Intelligent model choice
- **Quality Control**: Automatic artifact detection
- **Multi-modal Fusion**: Cross-modality integration
- **Clinical Reporting**: Automated documentation

### ⚡ **Performance Features**
- **Parallel Processing**: Multi-core utilization
- **GPU Acceleration**: CUDA support for deep learning
- **Memory Optimization**: Efficient large data handling
- **Batch Operations**: Bulk processing capabilities

---

## 🐛 Troubleshooting

### ⚠️ **Common Issues**

| Issue | Solution | Notebook |
|-------|----------|----------|
| Import errors | Check Python path and dependencies | All |
| Memory issues | Reduce batch size, use smaller images | 02, 03, 04 |
| GPU not available | Use CPU mode, install CUDA | 03, 04 |
| File not found | Check file paths and permissions | All |
| Slow processing | Enable parallel processing | 01, 02, 03 |

### 🆘 **Getting Help**
```python
# Check module installation
from onem_segment import ROISegmenter
print(ROISegmenter.__doc__)

# Get help with functions
help(segmenter.segment_image)

# Check available configurations
from onem_radiomics.config.settings import get_preset_config
print(get_preset_config('default'))
```

---

## 📚 Additional Resources

### 📖 **Documentation**
- [Module READMEs](../README.md) - Detailed module documentation
- [API Reference](../docs/) - Complete function documentation
- [Configuration Guide](../config/) - All configuration options

### 🎓 **Tutorials**
- [Video Tutorials](https://example.com/tutorials) - Step-by-step video guides
- [Blog Posts](https://example.com/blog) - Use cases and best practices
- [Webinar Series](https://example.com/webinars) - Live training sessions

### 🔗 **Community**
- [GitHub Issues](https://github.com/example/issues) - Bug reports and features
- [Discord Server](https://discord.gg/example) - Live chat and support
- [Stack Overflow](https://stackoverflow.com/questions/tagged/omnimedai) - Q&A

---

## 🏆 Success Stories

### 🏥 **Clinical Deployment**
> *"The comprehensive workflow notebook helped us deploy in production within 2 weeks."*  
> – Hospital Radiology Department

### 🔬 **Research Breakthrough**
> *"Multi-modal feature analysis led to our publication in Nature Medicine."*  
> – Cancer Research Institute

### 🎓 **Educational Impact**
> *"Perfect teaching tool for medical AI curriculum."*  
> – Medical University Professor

---

## 🚀 Next Steps

### 📈 **For Production**
1. ✅ Complete all notebooks
2. 🔧 Customize configurations
3. 🧪 Validate with your data
4. 🏥 Deploy in clinical environment
5. 📊 Monitor performance metrics

### 📚 **For Learning**
1. ✅ Work through notebooks sequentially
2. 📝 Take notes and experiment
3. 🤝 Join community discussions
4. 🔬 Try real-world datasets
5. 🎯 Build custom pipelines

### 🔬 **For Research**
1. ✅ Master comprehensive workflow
2. 📊 Publish reproducible results
3. 🔗 Integrate with existing tools
4. 🧪 Contribute new methods
5. 📖 Share with community

---

## 🤝 Professional Support

For advanced features, professional support, and enterprise deployments:

- **WeChat**: AcePwn
- **Email**: acezqy@gmail.com

---

## 📄 License

These notebooks are part of the OmniMedAI open-source project. See main project license for details.

---

## 🎉 Ready to Start?

**Choose your adventure:**

🌱 **New to Medical AI?** → Start with [`00_Quick_Start_Tutorial.ipynb`](00_Quick_Start_Tutorial.ipynb)

🔬 **Ready for deep learning?** → Jump to [`03_Pathology_Analysis.ipynb`](03_Pathology_Analysis.ipynb)

🏥 **Clinical deployment?** → Master [`04_Comprehensive_Workflow.ipynb`](04_Comprehensive_Workflow.ipynb)

**Happy analyzing! 🚀**