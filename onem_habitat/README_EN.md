# onem_habitat

Medical Imaging Radiomics Habitat Analysis Module - Implements local feature computation for ROI regions and clustering-based mask refinement.

## Features

### 1. Local Radiomics Feature Extraction
- **5x5x5 3D Regional Feature Extraction**: Compute local radiomics features for each ROI voxel
- **Multiple Feature Types**: Support first-order statistics, GLCM, GLRLM, GLSZM, GLTDM, NGTDM features
- **Flexible Configuration**: Configurable kernel size, bin width, resampling parameters, etc.
- **Parallel Processing**: Support multi-threaded parallel computation

### 2. Feature Clustering Analysis
- **Multiple Clustering Algorithms**: K-means, DBSCAN, Hierarchical clustering
- **Feature Selection**: Variance filtering, correlation analysis, PCA dimensionality reduction
- **Clustering Evaluation**: Silhouette coefficient, Calinski-Harabasz index and other quality metrics
- **Visualization Support**: 2D/3D clustering result visualization

### 3. Clustering-based Mask Refinement
- **Clustering-Driven**: Re-partition mask regions based on feature clustering results
- **Morphological Optimization**: Connected component analysis, hole filling, boundary smoothing
- **Multi-level Processing**: Small cluster filtering, noise point assignment, spatial compactness optimization
- **Quality Control**: Coverage, balance, and connectivity assessment

### 4. Configuration Management
- **Flexible Configuration**: Complete parameter configuration system
- **Preset Templates**: CT lung cancer, MRI brain tumor, PET tumor and other preset configurations
- **Validation Mechanism**: Automatic configuration validation and warning prompts
- **Import/Export**: Save and load configuration templates

## Installation

```bash
# Core dependencies
pip install numpy nibabel

# Radiomics feature extraction
pip install pyradiomics SimpleITK

# Clustering and visualization
pip install scikit-learn matplotlib seaborn

# Image processing and morphology
pip install scikit-image

# Development and testing
pip install unittest
```

## Quick Start

### 1. Local Radiomics Feature Extraction

```python
from onem_habitat.radiomics import LocalRadiomicsExtractor

# Create feature extractor
extractor = LocalRadiomicsExtractor(
    kernel_size=(5, 5, 5),           # 5x5x5 kernel
    feature_types=['firstorder', 'glcm'],  # Feature types
    bin_width=25,                      # Histogram binning
    n_jobs=4                            # Parallel processing
)

# Extract features for a single image
result = extractor.extract_local_features(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    output_path="features/patient001_features.npy",
    step_size=2  # Extract every 2 voxels to reduce computation
)

print(f"Extracted features for {result['metadata']['n_voxels']} voxels")
print(f"Feature types: {result['metadata']['feature_types']}")
```

### 2. Batch Feature Extraction

```python
# Batch extraction
results = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks",
    output_dir="output/features",
    file_pattern="*.nii.gz",
    step_size=3
)

print(f"Successfully processed {len([r for r in results if 'error' not in r])} images")
```

### 3. Feature Clustering Analysis

```python
from onem_habitat.clustering import FeatureClustering

# Load feature data
features_dict_list = []
for result_file in Path("features").glob("*_features.npy"):
    features_dict = np.load(result_file, allow_pickle=True).item()
    features_dict_list.append(features_dict)

# Create clusterer
clusterer = FeatureClustering(
    clustering_method='kmeans',
    n_clusters=4,
    feature_selection='variance',    # Variance-based feature selection
    pca_components=10,               # PCA dimensionality reduction
    standardize=True
)

# Perform clustering
cluster_labels_list = clusterer.fit_predict(features_dict_list)

# Clustering visualization
clusterer.visualize_clusters(
    features=clusterer._prepare_features(features_dict_list)[0],
    labels=cluster_labels_list[0],
    output_path="output/cluster_visualization.png"
)
```

### 4. Clustering-based Mask Refinement

```python
from onem_habitat.segmentation import MaskRefiner

# Create refiner
refiner = MaskRefiner(
    min_cluster_size=50,      # Minimum cluster size
    smoothing_iterations=2,     # Smoothing iterations
    connectivity=1,           # 6-connectivity
    fill_holes=True          # Fill holes
)

# Refine mask
saved_files = refiner.refine_masks(
    image_path="images/patient001.nii.gz",
    mask_path="masks/patient001_mask.nii.gz",
    cluster_labels=cluster_labels_list[0],
    coordinates=features_dict_list[0]['coordinates'],
    output_dir="output/refined_masks",
    save_individual=True,      # Save individual cluster masks
    save_combined=True         # Save combined refined mask
)

print(f"Saved {len([k for k in saved_files.keys() if k.startswith('cluster_')])} cluster masks")
```

### 5. Complete Workflow

```python
from onem_habitat.config import PRESET_CONFIGS

# Use preset configuration
config = PRESET_CONFIGS['ct_lung']

# Step 1: Feature extraction
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

# Step 2: Feature clustering
clusterer = FeatureClustering(
    clustering_method=config.clustering_method,
    n_clusters=config.n_clusters,
    feature_selection=config.feature_selection
)
cluster_labels_list = clusterer.fit_predict([
    np.load(r['output_file'], allow_pickle=True).item()
    for r in extraction_results if 'error' not in r
])

# Step 3: Mask refinement
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

print("✅ Complete habitat analysis workflow finished!")
```

## Directory Structure

```
onem_habitat/
├── __init__.py              # Main entry file
├── radiomics/                # Radiomics feature extraction
│   ├── __init__.py
│   └── local_extractor.py   # Local feature extractor
├── clustering/               # Feature clustering analysis
│   ├── __init__.py
│   └── feature_clustering.py # Feature clusterer
├── segmentation/             # Mask refinement
│   ├── __init__.py
│   └── mask_refiner.py     # Mask refiner
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── habitat_utils.py    # Habitat analysis tools
│   └── radiomics_utils.py  # Radiomics tools
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py         # Configuration classes and presets
├── example_usage.py          # Usage examples
├── test_basic.py           # Basic tests
└── README.md               # Documentation
```

## API Documentation

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

## Configuration Options

### HabitatConfig Main Parameters

```python
# Feature extraction configuration
kernel_size: Tuple[int, int, int] = (5, 5, 5)     # 3D kernel size
feature_types: List[str] = ['firstorder', 'glcm']    # Feature types
bin_width: int = 25                                   # Histogram bin width
extraction_n_jobs: int = 1                              # Number of parallel processes
step_size: int = 1                                    # Extraction step size

# Clustering configuration
clustering_method: str = 'kmeans'                     # Clustering algorithm
n_clusters: int = 5                                   # Number of clusters
feature_selection: str = 'all'                           # Feature selection
pca_components: Optional[int] = None                     # PCA dimensionality reduction
standardize: bool = True                                # Feature standardization

# Mask refinement configuration
min_cluster_size: int = 50                             # Minimum cluster size
smoothing_iterations: int = 2                           # Smoothing iteration count
connectivity: int = 1                                   # Connectivity
fill_holes: bool = True                                # Fill holes
```

## Preset Configuration Templates

```python
from onem_habitat.config import PRESET_CONFIGS

# CT lung cancer analysis
ct_lung_config = PRESET_CONFIGS['ct_lung']

# MRI brain tumor analysis  
mri_brain_config = PRESET_CONFIGS['mri_brain']

# PET tumor analysis
pet_tumor_config = PRESET_CONFIGS['pet_tumor']

# High resolution analysis
high_res_config = PRESET_CONFIGS['high_resolution']

# Fast processing
fast_config = PRESET_CONFIGS['fast_processing']
```

## Performance Optimization

### 1. Parallel Processing
```python
# Feature extraction parallelization
extractor = LocalRadiomicsExtractor(n_jobs=4)

# Batch processing optimization
results = extractor.batch_extract_features(
    images_dir="data/images",
    masks_dir="data/masks", 
    step_size=2  # Reduce computation
)
```

### 2. Memory Optimization
```python
# Use step size to reduce feature computation
extractor.extract_local_features(
    image_path="image.nii.gz",
    mask_path="mask.nii.gz",
    step_size=3  # Compute every 3 voxels
)

# Feature dimensionality reduction to reduce memory usage
clusterer = FeatureClustering(pca_components=10)
```

### 3. Feature Selection
```python
# Variance filtering to reduce feature count
clusterer = FeatureClustering(
    feature_selection='variance',
    n_clusters=5
)

# Correlation analysis to remove redundant features
clusterer = FeatureClustering(
    feature_selection='correlation'
)
```

## Testing

Run basic tests:

```bash
cd onem_habitat
python test_basic.py
```

Run examples:

```bash
python example_usage.py
```

## Output File Formats

### 1. Feature Files (.npy)
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

### 2. Clustering Files (.npy)
```python
# clustering_labels_patient001.npy
[cluster_id, cluster_id, cluster_id, ...]  # Cluster label for each voxel
```

### 3. Refined Mask Files (.nii.gz)
```
cluster_0_mask.nii.gz  # Cluster 0 mask
cluster_1_mask.nii.gz  # Cluster 1 mask
...
refined_combined_mask.nii.gz  # Combined refined mask
```

## Frequently Asked Questions

### Q: How to choose appropriate kernel size?
A: Kernel size depends on image resolution and target feature scale:
- High-resolution images: 3x3x3 or 5x5x5
- Medium resolution: 5x5x5 or 7x7x7  
- Low resolution: 7x7x7 or 9x9x9

### Q: How to determine optimal number of clusters?
A: You can use the following methods:
1. Elbow method: Observe changes in silhouette coefficient with cluster count
2. Domain knowledge: Determine based on medical prior knowledge
3. Silhouette coefficient: Choose cluster count with maximum silhouette coefficient

### Q: What to do when memory is insufficient for large datasets?
A: You can adopt the following strategies:
1. Increase step size: step_size=2 or 3
2. Feature selection: Reduce number of feature types
3. PCA dimensionality reduction: pca_components=10-20
4. Batch processing: Process in batches by region or slice

## License

This project is licensed under the MIT License.

## Contributing

Issues and Pull Requests are welcome to improve this module.

## Changelog

### v1.0.0
- Initial release
- Implemented local radiomics feature extraction
- Implemented multiple clustering algorithms
- Implemented clustering-based mask refinement
- Added complete configuration management system
- Provided multiple preset configuration templates