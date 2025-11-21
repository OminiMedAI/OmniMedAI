"""
Image Dimension Analyzer

This module analyzes medical image dimensions to determine whether to use
2D or 3D segmentation models based on z-axis characteristics.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

# Try to import nibabel
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Install with: pip install nibabel")


class ImageDimensionAnalyzer:
    """
    Analyzes medical image dimensions to determine optimal segmentation approach.
    
    This class determines whether to use 2D or 3D segmentation models by analyzing
    the z-axis characteristics of NIfTI images, including slice count, spacing,
    and content variation.
    """
    
    def __init__(self, 
                 min_3d_slices: int = 30,
                 max_slice_spacing: float = 5.0,
                 min_content_variation: float = 0.1,
                 slice_sampling_ratio: float = 0.1):
        """
        Initialize the image dimension analyzer.
        
        Args:
            min_3d_slices: Minimum number of slices to consider 3D processing
            max_slice_spacing: Maximum slice spacing (mm) for 3D processing
            min_content_variation: Minimum content variation threshold
            slice_sampling_ratio: Ratio of slices to sample for content analysis
        """
        self.min_3d_slices = min_3d_slices
        self.max_slice_spacing = max_slice_spacing
        self.min_content_variation = min_content_variation
        self.slice_sampling_ratio = slice_sampling_ratio
        self.logger = logging.getLogger(__name__)
        
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a NIfTI image to determine dimension characteristics.
        
        Args:
            image_path: Path to the NIfTI image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load image
            img = nib.load(image_path)
            data = img.get_fdata()
            header = img.header
            zooms = header.get_zooms()
            
            # Extract basic information
            shape = data.shape
            n_slices = shape[2] if len(shape) >= 3 else 1
            slice_spacing = zooms[2] if len(zooms) >= 3 else 1.0
            
            # Determine if 2D or 3D
            is_3d = self._determine_dimension_type(data, n_slices, slice_spacing)
            
            # Analyze content characteristics
            content_analysis = self._analyze_content(data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data)
            
            analysis_result = {
                'image_path': image_path,
                'shape': shape,
                'n_slices': n_slices,
                'slice_spacing': slice_spacing,
                'voxel_size_mm': zooms,
                'is_3d': is_3d,
                'content_analysis': content_analysis,
                'quality_metrics': quality_metrics,
                'recommendations': self._generate_recommendations(
                    is_3d, n_slices, slice_spacing, content_analysis
                )
            }
            
            self.logger.info(f"Analyzed {Path(image_path).name}: "
                           f"{'3D' if is_3d else '2D'} processing recommended")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze image {image_path}: {e}")
            raise
    
    def _determine_dimension_type(self, data: np.ndarray, 
                                n_slices: int, slice_spacing: float) -> bool:
        """
        Determine whether to use 2D or 3D processing based on image characteristics.
        
        Args:
            data: Image data array
            n_slices: Number of slices
            slice_spacing: Slice spacing in mm
            
        Returns:
            True if 3D processing is recommended, False for 2D
        """
        # Criteria for 3D processing
        criteria_3d = []
        
        # 1. Minimum number of slices
        criteria_3d.append(n_slices >= self.min_3d_slices)
        
        # 2. Reasonable slice spacing
        criteria_3d.append(slice_spacing <= self.max_slice_spacing)
        
        # 3. Content variation across slices
        if n_slices > 1:
            # Sample slices for efficiency
            sample_indices = np.linspace(0, n_slices-1, 
                                       max(2, int(n_slices * self.slice_sampling_ratio)), 
                                       dtype=int)
            
            content_variations = []
            for i in range(len(sample_indices) - 1):
                idx1, idx2 = sample_indices[i], sample_indices[i + 1]
                slice1 = data[:, :, idx1].flatten()
                slice2 = data[:, :, idx2].flatten()
                
                # Calculate correlation between consecutive slices
                if np.std(slice1) > 0 and np.std(slice2) > 0:
                    correlation = np.corrcoef(slice1, slice2)[0, 1]
                    content_variations.append(1 - abs(correlation))  # Dissimilarity
            
            if content_variations:
                avg_variation = np.mean(content_variations)
                criteria_3d.append(avg_variation >= self.min_content_variation)
            else:
                criteria_3d.append(True)  # Default to 3D if can't determine
        else:
            criteria_3d.append(False)  # Single slice = 2D
        
        # Final decision: need at least 2 out of 3 criteria
        return sum(criteria_3d) >= 2
    
    def _analyze_content(self, data: np.ndarray) -> Dict:
        """
        Analyze content characteristics of the image.
        
        Args:
            data: Image data array
            
        Returns:
            Dictionary with content analysis results
        """
        # Flatten data for analysis
        flat_data = data.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaN values
        
        if len(flat_data) == 0:
            return {'error': 'No valid data found'}
        
        # Basic statistics
        content_stats = {
            'mean': float(np.mean(flat_data)),
            'std': float(np.std(flat_data)),
            'min': float(np.min(flat_data)),
            'max': float(np.max(flat_data)),
            'range': float(np.max(flat_data) - np.min(flat_data)),
            'median': float(np.median(flat_data)),
            'skewness': self._calculate_skewness(flat_data),
            'kurtosis': self._calculate_kurtosis(flat_data)
        }
        
        # Analyze signal distribution
        hist, bin_edges = np.histogram(flat_data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        content_stats.update({
            'histogram_peaks': int(self._count_histogram_peaks(hist)),
            'dominant_intensity': float(bin_centers[np.argmax(hist)]),
            'background_threshold': float(self._estimate_background_threshold(flat_data))
        })
        
        # Slice-wise analysis for 3D images
        if len(data.shape) >= 3 and data.shape[2] > 1:
            slice_analysis = self._analyze_slices(data)
            content_stats['slice_analysis'] = slice_analysis
        
        return content_stats
    
    def _analyze_slices(self, data: np.ndarray) -> Dict:
        """
        Analyze individual slices for 3D images.
        
        Args:
            data: 3D image data array
            
        Returns:
            Dictionary with slice analysis results
        """
        n_slices = data.shape[2]
        slice_stats = []
        
        for i in range(n_slices):
            slice_data = data[:, :, i].flatten()
            slice_data = slice_data[~np.isnan(slice_data)]
            
            if len(slice_data) > 0:
                slice_stats.append({
                    'mean': float(np.mean(slice_data)),
                    'std': float(np.std(slice_data)),
                    'signal_range': float(np.max(slice_data) - np.min(slice_data))
                })
        
        if not slice_stats:
            return {'error': 'No valid slices found'}
        
        # Calculate inter-slice statistics
        means = [s['mean'] for s in slice_stats]
        stds = [s['std'] for s in slice_stats]
        ranges = [s['signal_range'] for s in slice_stats]
        
        return {
            'mean_variation': float(np.std(means)),
            'std_variation': float(np.std(stds)),
            'range_variation': float(np.std(ranges)),
            'high_variance_slices': int(sum(1 for s in slice_stats if s['std'] > np.mean(stds) + np.std(stds))),
            'low_signal_slices': int(sum(1 for s in slice_stats if s['signal_range'] < np.mean(ranges) * 0.5))
        }
    
    def _calculate_quality_metrics(self, data: np.ndarray) -> Dict:
        """
        Calculate image quality metrics.
        
        Args:
            data: Image data array
            
        Returns:
            Dictionary with quality metrics
        """
        # Signal-to-noise ratio (SNR)
        flat_data = data.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]
        
        if len(flat_data) == 0:
            return {'error': 'No valid data for quality assessment'}
        
        signal_power = np.mean(flat_data ** 2)
        noise_power = np.var(flat_data)
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        
        # Contrast-to-noise ratio (CNR) approximation
        background_threshold = self._estimate_background_threshold(flat_data)
        foreground = flat_data[flat_data > background_threshold]
        background = flat_data[flat_data <= background_threshold]
        
        if len(foreground) > 0 and len(background) > 0:
            cnr = (np.mean(foreground) - np.mean(background)) / np.std(background)
        else:
            cnr = 0.0
        
        return {
            'snr': float(snr),
            'cnr': float(cnr),
            'dynamic_range': float(np.max(flat_data) - np.min(flat_data)),
            'effective_bits_used': int(np.log2(np.max(flat_data) - np.min(flat_data) + 1)),
            'sparsity': float(np.sum(flat_data == 0) / len(flat_data))
        }
    
    def _generate_recommendations(self, is_3d: bool, n_slices: int, 
                                slice_spacing: float, content_analysis: Dict) -> Dict:
        """
        Generate processing recommendations based on analysis.
        
        Args:
            is_3d: Whether 3D processing is recommended
            n_slices: Number of slices
            slice_spacing: Slice spacing
            content_analysis: Content analysis results
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'processing_type': '3D' if is_3d else '2D',
            'confidence': 'high' if abs(n_slices - self.min_3d_slices) > 10 else 'medium',
            'preprocessing_suggestions': [],
            'model_suggestions': []
        }
        
        # Preprocessing suggestions
        if content_analysis.get('skewness', 0) > 2:
            recommendations['preprocessing_suggestions'].append('log_transform')
        
        if content_analysis.get('std', 0) / (content_analysis.get('mean', 1) + 1e-8) < 0.1:
            recommendations['preprocessing_suggestions'].append('contrast_enhancement')
        
        if slice_spacing > 3.0:
            recommendations['preprocessing_suggestions'].append('interpolation')
        
        # Model suggestions
        if is_3d:
            recommendations['model_suggestions'].append('3d_unet')
            recommendations['model_suggestions'].append('3d_vnet')
            if n_slices > 64:
                recommendations['model_suggestions'].append('cascade_3d')
        else:
            recommendations['model_suggestions'].append('2d_unet')
            recommendations['model_suggestions'].append('2d_attention_unet')
            recommendations['model_suggestions'].append('slice_ensemble')
        
        return recommendations
    
    def _estimate_background_threshold(self, data: np.ndarray) -> float:
        """Estimate background threshold using Otsu's method."""
        try:
            from skimage.filters import threshold_otsu
            return threshold_otsu(data)
        except ImportError:
            # Fallback: use 10th percentile
            return np.percentile(data, 10)
    
    def _count_histogram_peaks(self, hist: np.ndarray) -> int:
        """Count the number of peaks in histogram."""
        # Simple peak detection
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks += 1
        return peaks
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def batch_analyze(self, image_paths: list) -> Dict:
        """
        Analyze multiple images and provide batch recommendations.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary with batch analysis results
        """
        results = []
        summary = {
            'total_images': len(image_paths),
            'recommend_3d': 0,
            'recommend_2d': 0,
            'processing_confidence': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
                
                # Update summary
                if result['is_3d']:
                    summary['recommend_3d'] += 1
                else:
                    summary['recommend_2d'] += 1
                
                confidence = result['recommendations']['confidence']
                if confidence in summary['processing_confidence']:
                    summary['processing_confidence'][confidence] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze {image_path}: {e}")
                results.append({'error': str(e), 'image_path': image_path})
        
        return {
            'individual_results': results,
            'summary': summary,
            'batch_recommendations': self._generate_batch_recommendations(summary)
        }
    
    def _generate_batch_recommendations(self, summary: Dict) -> Dict:
        """Generate recommendations for batch processing."""
        total = summary['total_images']
        if total == 0:
            return {'error': 'No valid images analyzed'}
        
        recommendations = {}
        
        # Primary processing approach
        if summary['recommend_3d'] > summary['recommend_2d']:
            recommendations['primary_approach'] = '3D'
            recommendations['confidence'] = summary['recommend_3d'] / total
        else:
            recommendations['primary_approach'] = '2D'
            recommendations['confidence'] = summary['recommend_2d'] / total
        
        # Mixed processing strategy
        if 0.3 < summary['recommend_3d'] / total < 0.7:
            recommendations['mixed_processing'] = True
            recommendations['strategy'] = 'adaptive_selection'
        else:
            recommendations['mixed_processing'] = False
        
        return recommendations