"""
CellProfiler-based Pathology Radiomics Feature Extractor

This module provides pathology radiomics feature extraction capabilities using CellProfiler.
It extracts comprehensive morphological, texture, and intensity features from pathology images.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import subprocess
import tempfile

# Try to import required dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from skimage import io, color, filters, measure, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from ..utils.cellprofiler_utils import (
        create_cellprofiler_pipeline, run_cellprofiler_pipeline
    )
    CELLPROFILER_UTILS_AVAILABLE = True
except ImportError:
    CELLPROFILER_UTILS_AVAILABLE = False


class CellProfilerExtractor:
    """
    Extract pathology radiomics features using CellProfiler.
    
    This class provides a comprehensive interface for extracting traditional
    pathology features including morphological, texture, and intensity features.
    """
    
    def __init__(self, config=None):
        """
        Initialize CellProfiler extractor.
        
        Args:
            config: Configuration object for feature extraction parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.temp_dir = None
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize feature extraction parameters
        self.feature_modules = self._initialize_feature_modules()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available. Install with: pip install Pillow")
        
        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image not available. Install with: pip install scikit-image")
        
        if not CELLPROFILER_UTILS_AVAILABLE:
            self.logger.warning("CellProfiler utils not available")
    
    def _initialize_feature_modules(self):
        """Initialize available feature extraction modules."""
        modules = {
            'morphological': True,
            'texture': True,
            'intensity': True,
            'color': True,
            'nuclei': True,
            'cells': True,
            'cytoplasm': True,
            'membrane': True
        }
        
        # Disable modules based on dependencies
        if not SKIMAGE_AVAILABLE:
            modules['texture'] = False
            modules['nuclei'] = False
            modules['cells'] = False
            modules['cytoplasm'] = False
            modules['membrane'] = False
        
        return modules
    
    def extract_features(self, image_path: str, output_path: str = None,
                        modules: List[str] = None) -> Dict:
        """
        Extract pathology radiomics features from a single image.
        
        Args:
            image_path: Path to pathology image file
            output_path: Path to save extracted features
            modules: List of feature modules to extract
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Use default modules if not specified
            if modules is None:
                modules = [mod for mod, enabled in self.feature_modules.items() if enabled]
            
            self.logger.info(f"Extracting features from {Path(image_path).name}")
            
            # Load image
            image_data = self._load_image(image_path)
            if image_data is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Extract features from different modules
            all_features = {}
            module_results = {}
            
            for module in modules:
                if module in self.feature_modules and self.feature_modules[module]:
                    try:
                        features = self._extract_module_features(module, image_data)
                        if features:
                            all_features.update(features)
                            module_results[module] = 'success'
                        else:
                            module_results[module] = 'empty'
                    except Exception as e:
                        self.logger.error(f"Failed to extract {module} features: {e}")
                        module_results[module] = f'error: {str(e)}'
                else:
                    module_results[module] = 'disabled'
            
            # Prepare result
            result = {
                'image_path': image_path,
                'features': all_features,
                'module_results': module_results,
                'image_metadata': self._extract_image_metadata(image_data, image_path),
                'extraction_metadata': {
                    'modules_requested': modules,
                    'modules_successful': [mod for mod, status in module_results.items() 
                                         if status == 'success'],
                    'total_features': len(all_features)
                }
            }
            
            # Save features if output path provided
            if output_path:
                self._save_features(result, output_path)
            
            self.logger.info(f"Successfully extracted {len(all_features)} features "
                           f"from {len([r for r in module_results.values() if r == 'success'])} modules")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            raise
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             file_pattern: str = "*.jpg *.png *.tif *.tiff",
                             modules: List[str] = None, n_jobs: int = 1) -> pd.DataFrame:
        """
        Extract features from multiple images in a directory.
        
        Args:
            image_dir: Directory containing pathology images
            output_dir: Directory to save feature results
            file_pattern: File pattern to match images
            modules: List of feature modules to extract
            n_jobs: Number of parallel jobs
            
        Returns:
            DataFrame containing features for all images
        """
        try:
            # Find image files
            image_files = self._find_image_files(image_dir, file_pattern)
            
            if not image_files:
                raise ValueError(f"No images found in {image_dir}")
            
            self.logger.info(f"Found {len(image_files)} images for batch processing")
            
            # Create output structure
            if output_dir:
                output_structure = self._create_output_structure(output_dir)
            else:
                output_structure = None
            
            # Process images
            all_results = []
            
            for image_path in image_files:
                try:
                    # Generate output path
                    if output_structure:
                        output_path = self._generate_batch_output_path(
                            image_path, output_structure['features']
                        )
                    else:
                        output_path = None
                    
                    # Extract features
                    result = self.extract_features(image_path, output_path, modules)
                    all_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    all_results.append({
                        'image_path': image_path,
                        'error': str(e)
                    })
            
            # Convert to DataFrame
            df = self._results_to_dataframe(all_results)
            
            # Save batch results
            if output_structure:
                self._save_batch_results(df, output_structure)
            
            successful = len([r for r in all_results if 'error' not in r])
            self.logger.info(f"Batch processing completed: {successful}/{len(image_files)} successful")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Batch feature extraction failed: {e}")
            raise
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load pathology image."""
        try:
            if PIL_AVAILABLE:
                # Use PIL for better image format support
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image_array = np.array(img)
                    return image_array
            else:
                # Fallback to skimage
                return io.imread(image_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _extract_module_features(self, module: str, image_data: np.ndarray) -> Dict:
        """Extract features from a specific module."""
        if module == 'morphological':
            return self._extract_morphological_features(image_data)
        elif module == 'texture':
            return self._extract_texture_features(image_data)
        elif module == 'intensity':
            return self._extract_intensity_features(image_data)
        elif module == 'color':
            return self._extract_color_features(image_data)
        elif module == 'nuclei':
            return self._extract_nuclei_features(image_data)
        elif module == 'cells':
            return self._extract_cell_features(image_data)
        elif module == 'cytoplasm':
            return self._extract_cytoplasm_features(image_data)
        elif module == 'membrane':
            return self._extract_membrane_features(image_data)
        else:
            raise ValueError(f"Unknown feature module: {module}")
    
    def _extract_morphological_features(self, image_data: np.ndarray) -> Dict:
        """Extract morphological features from image."""
        if not SKIMAGE_AVAILABLE:
            return {}
        
        features = {}
        
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray = color.rgb2gray(image_data)
        else:
            gray = image_data
        
        # Basic morphological features
        try:
            # Image size
            features['image_width'] = gray.shape[1]
            features['image_height'] = gray.shape[0]
            features['image_aspect_ratio'] = gray.shape[1] / gray.shape[0]
            
            # Area and perimeter (assuming binary thresholding)
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Connected components
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            
            if regions:
                # Number of objects
                features['num_objects'] = len(regions)
                
                # Average object properties
                areas = [r.area for r in regions]
                perimeters = [r.perimeter for r in regions]
                
                features['avg_object_area'] = np.mean(areas)
                features['std_object_area'] = np.std(areas)
                features['max_object_area'] = np.max(areas)
                features['min_object_area'] = np.min(areas)
                
                features['avg_object_perimeter'] = np.mean(perimeters)
                features['std_object_perimeter'] = np.std(perimeters)
                
                # Circularity features
                circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0 
                               for r in regions]
                features['avg_circularity'] = np.mean(circularities)
                features['std_circularity'] = np.std(circularities)
                
                # Solidity
                solidities = [r.solidity for r in regions]
                features['avg_solidity'] = np.mean(solidities)
                features['std_solidity'] = np.std(solidities)
                
                # Eccentricity
                eccentricities = [r.eccentricity for r in regions]
                features['avg_eccentricity'] = np.mean(eccentricities)
                features['std_eccentricity'] = np.std(eccentricities)
            
            # Texture-based morphological features
            features['binary_fill_ratio'] = np.sum(binary) / binary.size
            
        except Exception as e:
            self.logger.error(f"Failed to extract morphological features: {e}")
        
        return features
    
    def _extract_texture_features(self, image_data: np.ndarray) -> Dict:
        """Extract texture features using GLCM and other methods."""
        if not SKIMAGE_AVAILABLE:
            return {}
        
        features = {}
        
        try:
            # Convert to grayscale
            if len(image_data.shape) == 3:
                gray = color.rgb2gray(image_data)
            else:
                gray = image_data
            
            # GLCM features
            from skimage.feature import graycomatrix, graycoprops
            
            # Calculate GLCM
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray.astype(np.uint8), distances=distances, angles=angles)
            
            # GLCM properties
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            
            for prop in props:
                glcm_values = graycoprops(glcm, prop)
                features[f'glcm_{prop}_mean'] = np.mean(glcm_values)
                features[f'glcm_{prop}_std'] = np.std(glcm_values)
                features[f'glcm_{prop}_min'] = np.min(glcm_values)
                features[f'glcm_{prop}_max'] = np.max(glcm_values)
            
            # LBP (Local Binary Pattern) features
            try:
                from skimage.feature import local_binary_pattern
                
                # LBP histogram
                radius = 3
                n_points = 8 * radius
                lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                
                # LBP histogram
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)
                lbp_hist = lbp_hist.astype(float)
                lbp_hist /= (lbp_hist.sum() + 1e-7)
                
                # LBP features
                features['lbp_energy'] = np.sum(lbp_hist ** 2)
                features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
                
            except Exception as e:
                self.logger.warning(f"Failed to extract LBP features: {e}")
            
            # Gabor filter responses
            try:
                frequencies = [0.1, 0.3, 0.5]
                thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                
                gabor_responses = []
                for freq in frequencies:
                    for theta in thetas:
                        real, _ = filters.gabor(gray, frequency=freq, theta=theta)
                        gabor_responses.append(np.mean(real))
                
                features['gabor_mean'] = np.mean(gabor_responses)
                features['gabor_std'] = np.std(gabor_responses)
                features['gabor_max'] = np.max(gabor_responses)
                features['gabor_min'] = np.min(gabor_responses)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract Gabor features: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract texture features: {e}")
        
        return features
    
    def _extract_intensity_features(self, image_data: np.ndarray) -> Dict:
        """Extract intensity-based features."""
        features = {}
        
        try:
            # Convert to grayscale if needed
            if len(image_data.shape) == 3:
                gray = color.rgb2gray(image_data) if SKIMAGE_AVAILABLE else np.mean(image_data, axis=2)
            else:
                gray = image_data
            
            # Basic statistics
            features['intensity_mean'] = np.mean(gray)
            features['intensity_std'] = np.std(gray)
            features['intensity_min'] = np.min(gray)
            features['intensity_max'] = np.max(gray)
            features['intensity_range'] = np.max(gray) - np.min(gray)
            features['intensity_median'] = np.median(gray)
            features['intensity_mode'] = self._mode(gray)
            
            # Percentiles
            features['intensity_p10'] = np.percentile(gray, 10)
            features['intensity_p25'] = np.percentile(gray, 25)
            features['intensity_p75'] = np.percentile(gray, 75)
            features['intensity_p90'] = np.percentile(gray, 90)
            
            # Shape statistics
            features['intensity_skewness'] = self._skewness(gray)
            features['intensity_kurtosis'] = self._kurtosis(gray)
            
            # Histogram-based features
            hist, bin_edges = np.histogram(gray, bins=32, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            features['histogram_peak_height'] = np.max(hist)
            features['histogram_peak_location'] = bin_centers[np.argmax(hist)]
            features['histogram_spread'] = np.sqrt(np.sum(((bin_centers - np.mean(gray)) ** 2) * hist))
            
            # Color channel features (if color image)
            if len(image_data.shape) == 3:
                for i, channel in enumerate(['R', 'G', 'B']):
                    channel_data = image_data[:, :, i]
                    features[f'{channel}_mean'] = np.mean(channel_data)
                    features[f'{channel}_std'] = np.std(channel_data)
                    features[f'{channel}_range'] = np.max(channel_data) - np.min(channel_data)
        
        except Exception as e:
            self.logger.error(f"Failed to extract intensity features: {e}")
        
        return features
    
    def _extract_color_features(self, image_data: np.ndarray) -> Dict:
        """Extract color-based features."""
        features = {}
        
        try:
            if len(image_data.shape) != 3:
                return features  # No color features for grayscale images
            
            # Color space conversions
            if SKIMAGE_AVAILABLE:
                # HSV features
                hsv = color.rgb2hsv(image_data)
                features['hsv_hue_mean'] = np.mean(hsv[:, :, 0])
                features['hsv_saturation_mean'] = np.mean(hsv[:, :, 1])
                features['hsv_value_mean'] = np.mean(hsv[:, :, 2])
                features['hsv_saturation_std'] = np.std(hsv[:, :, 1])
                
                # Lab features
                lab = color.rgb2lab(image_data)
                features['lab_l_mean'] = np.mean(lab[:, :, 0])
                features['lab_a_mean'] = np.mean(lab[:, :, 1])
                features['lab_b_mean'] = np.mean(lab[:, :, 2])
                features['lab_l_std'] = np.std(lab[:, :, 0])
            
            # RGB color ratios
            r_mean = np.mean(image_data[:, :, 0])
            g_mean = np.mean(image_data[:, :, 1])
            b_mean = np.mean(image_data[:, :, 2])
            
            sum_rgb = r_mean + g_mean + b_mean
            if sum_rgb > 0:
                features['r_ratio'] = r_mean / sum_rgb
                features['g_ratio'] = g_mean / sum_rgb
                features['b_ratio'] = b_mean / sum_rgb
            
            # Colorfulness metric
            rg = image_data[:, :, 0] - image_data[:, :, 1]
            yb = 0.5 * (image_data[:, :, 0] + image_data[:, :, 1]) - image_data[:, :, 2]
            
            features['colorfulness'] = (np.std(rg) + np.std(yb)) / 255.0
            
        except Exception as e:
            self.logger.error(f"Failed to extract color features: {e}")
        
        return features
    
    def _extract_nuclei_features(self, image_data: np.ndarray) -> Dict:
        """Extract nuclei-specific features."""
        features = {}
        
        try:
            if not SKIMAGE_AVAILABLE:
                return features
            
            # Convert to grayscale
            if len(image_data.shape) == 3:
                gray = color.rgb2gray(image_data)
            else:
                gray = image_data
            
            # Nuclei detection using thresholding and morphological operations
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Remove small objects
            binary = morphology.remove_small_objects(binary, min_size=50)
            
            # Fill holes
            binary = ndimage.binary_fill_holes(binary)
            
            # Label nuclei
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            
            if regions:
                features['nuclei_count'] = len(regions)
                
                # Nuclei size features
                areas = [r.area for r in regions]
                features['nuclei_avg_area'] = np.mean(areas)
                features['nuclei_std_area'] = np.std(areas)
                features['nuclei_max_area'] = np.max(areas)
                features['nuclei_min_area'] = np.min(areas)
                
                # Nuclei shape features
                eccentricities = [r.eccentricity for r in regions]
                solidities = [r.solidity for r in regions]
                circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0 
                               for r in regions]
                
                features['nuclei_avg_eccentricity'] = np.mean(eccentricities)
                features['nuclei_avg_solidity'] = np.mean(solidities)
                features['nuclei_avg_circularity'] = np.mean(circularities)
                
                # Nuclei intensity features
                mean_intensities = [r.intensity_mean for r in regions]
                features['nuclei_avg_intensity'] = np.mean(mean_intensities)
                features['nuclei_std_intensity'] = np.std(mean_intensities)
                
                # Nuclei density
                total_area = np.sum([r.area for r in regions])
                features['nuclei_density'] = total_area / gray.size
        
        except Exception as e:
            self.logger.error(f"Failed to extract nuclei features: {e}")
        
        return features
    
    def _extract_cell_features(self, image_data: np.ndarray) -> Dict:
        """Extract cell features (similar to nuclei but for entire cells)."""
        # For now, this is similar to nuclei extraction
        # In practice, this would use different staining markers
        return self._extract_nuclei_features(image_data)
    
    def _extract_cytoplasm_features(self, image_data: np.ndarray) -> Dict:
        """Extract cytoplasm features."""
        # Placeholder for cytoplasm feature extraction
        # This would require membrane staining and cell segmentation
        return {}
    
    def _extract_membrane_features(self, image_data: np.ndarray) -> Dict:
        """Extract membrane features."""
        # Placeholder for membrane feature extraction
        # This would require membrane-specific staining
        return {}
    
    def _extract_image_metadata(self, image_data: np.ndarray, image_path: str) -> Dict:
        """Extract metadata from image."""
        metadata = {
            'file_name': Path(image_path).name,
            'file_size_bytes': os.path.getsize(image_path),
            'image_shape': image_data.shape,
            'image_dtype': str(image_data.dtype),
            'image_channels': len(image_data.shape) if len(image_data.shape) < 3 else image_data.shape[2]
        }
        
        if len(image_data.shape) == 3:
            metadata['aspect_ratio'] = image_data.shape[1] / image_data.shape[0]
        else:
            metadata['aspect_ratio'] = image_data.shape[1] / image_data.shape[0]
        
        return metadata
    
    def _save_features(self, result: Dict, output_path: str):
        """Save extracted features to file."""
        try:
            # Prepare feature data
            feature_data = {}
            feature_data.update(result['features'])
            feature_data.update(result['image_metadata'])
            feature_data.update({'image_path': result['image_path']})
            
            # Save as JSON
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(feature_data, f, indent=2, default=str)
            else:
                # Save as CSV
                df = pd.DataFrame([feature_data])
                df.to_csv(output_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to save features to {output_path}: {e}")
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert extraction results to DataFrame."""
        rows = []
        
        for result in results:
            if 'error' in result:
                continue
            
            # Flatten features and metadata
            row = {
                'image_path': result['image_path'],
                'extraction_success': True,
                'total_features': result['extraction_metadata']['total_features']
            }
            
            # Add all features
            row.update(result['features'])
            row.update(result['image_metadata'])
            
            # Add module status
            for module, status in result['module_results'].items():
                row[f'module_{module}'] = status
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _find_image_files(self, directory: str, pattern: str) -> List[str]:
        """Find image files in directory."""
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(Path(directory).glob(ext))
        
        # Convert to strings and sort
        image_files = sorted([str(f) for f in image_files])
        
        return image_files
    
    def _generate_batch_output_path(self, image_path: str, output_dir: str) -> str:
        """Generate output path for batch processing."""
        input_file = Path(image_path)
        output_name = f"{input_file.stem}_features.json"
        return str(Path(output_dir) / output_name)
    
    def _create_output_structure(self, base_dir: str) -> Dict:
        """Create output directory structure."""
        base_path = Path(base_dir)
        
        structure = {
            'base': str(base_path),
            'features': str(base_path / 'features'),
            'reports': str(base_path / 'reports'),
            'logs': str(base_path / 'logs')
        }
        
        for path in structure.values():
            os.makedirs(path, exist_ok=True)
        
        return structure
    
    def _save_batch_results(self, df: pd.DataFrame, structure: Dict):
        """Save batch results."""
        try:
            # Save combined CSV
            csv_path = Path(structure['features']) / 'batch_features.csv'
            df.to_csv(csv_path, index=False)
            
            # Save summary report
            report = {
                'total_images': len(df),
                'successful_extractions': len(df[df['extraction_success'] == True]),
                'feature_modules': list(set([col.replace('module_', '') for col in df.columns 
                                         if col.startswith('module_') and df[col][0] != 'disabled'])),
                'total_feature_columns': len([col for col in df.columns if not col.startswith('module_') 
                                          and col not in ['image_path', 'extraction_success', 'total_features']])
            }
            
            report_path = Path(structure['reports']) / 'batch_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save batch results: {e}")
    
    def _mode(self, data: np.ndarray) -> float:
        """Calculate mode of array."""
        try:
            values, counts = np.unique(data, return_counts=True)
            return values[np.argmax(counts)]
        except:
            return np.mean(data)
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis