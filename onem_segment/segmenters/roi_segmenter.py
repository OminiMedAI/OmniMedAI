"""
ROI Segmenter

This module provides automatic ROI (Region of Interest) segmentation for medical images.
It automatically determines whether to use 2D or 3D segmentation models based on
z-axis characteristics and saves segmented ROI as NIfTI files.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Try to import required dependencies
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from ..models.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

try:
    from ..utils.image_analyzer import ImageDimensionAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

try:
    from ..utils.file_utils import load_nifti_image, save_nifti_image, create_output_structure
    FILE_UTILS_AVAILABLE = True
except ImportError:
    FILE_UTILS_AVAILABLE = False


class ROISegmenter:
    """
    Main class for automatic ROI segmentation of medical images.
    
    This class provides comprehensive ROI segmentation capabilities including:
    - Automatic 2D/3D model selection based on image characteristics
    - Model management and inference
    - Post-processing and refinement
    - Batch processing capabilities
    """
    
    def __init__(self, config=None):
        """
        Initialize the ROI segmenter.
        
        Args:
            config: Configuration object for segmentation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required. Install with: pip install nibabel")
        if not MODEL_MANAGER_AVAILABLE:
            raise ImportError("Model manager not available")
        if not ANALYZER_AVAILABLE:
            raise ImportError("Image analyzer not available")
        if not FILE_UTILS_AVAILABLE:
            raise ImportError("File utilities not available")
        
        # Initialize components
        self.model_manager = ModelManager()
        self.image_analyzer = ImageDimensionAnalyzer()
        
        # Model mapping for different scenarios
        self.model_mapping = {
            '2d_default': '2d_unet',
            '3d_default': '3d_unet',
            '2d_high_quality': '2d_attention_unet',
            '3d_high_quality': '3d_vnet'
        }
    
    def segment_image(self, image_path: str, output_path: str = None,
                    model_type: str = 'auto', quality: str = 'default',
                    return_probabilities: bool = False) -> Dict:
        """
        Segment ROI from a single medical image.
        
        Args:
            image_path: Path to the input NIfTI image
            output_path: Path to save the output segmentation
            model_type: Model type to use ('auto', '2d', '3d')
            quality: Quality level ('default', 'high_quality')
            return_probabilities: Whether to return probability maps
            
        Returns:
            Dictionary containing segmentation results and metadata
        """
        try:
            # Load image
            image_data, header, affine = load_nifti_image(image_path)
            
            # Analyze image characteristics
            analysis = self.image_analyzer.analyze_image(image_path)
            self.logger.info(f"Image analysis: {analysis['shape']}, "
                           f"{'3D' if analysis['is_3d'] else '2D'} recommended")
            
            # Determine model type
            if model_type == 'auto':
                use_3d = analysis['is_3d']
            else:
                use_3d = model_type.startswith('3d')
            
            # Select model
            model_key = f"{'3d' if use_3d else '2d'}_{quality}"
            model_name = self.model_mapping.get(model_key, self.model_mapping['2d_default'])
            
            # Load model if not already loaded
            if not self._ensure_model_loaded(model_name, model_name):
                raise RuntimeError(f"Failed to load model: {model_name}")
            
            # Perform segmentation
            segmentation = self.model_manager.predict(
                model_name, 
                image_data, 
                return_probabilities=return_probabilities
            )
            
            # Post-processing
            if not return_probabilities:
                segmentation = self._postprocess_segmentation(segmentation, analysis)
            
            # Save result
            if output_path is None:
                output_path = self._generate_output_path(image_path, analysis['is_3d'])
            
            save_nifti_image(segmentation, affine, header, output_path)
            
            # Prepare result
            result = {
                'input_path': image_path,
                'output_path': output_path,
                'segmentation': segmentation,
                'analysis': analysis,
                'model_used': model_name,
                'model_type': '3D' if use_3d else '2D',
                'quality': quality,
                'return_probabilities': return_probabilities,
                'metadata': {
                    'original_shape': image_data.shape,
                    'segmentation_shape': segmentation.shape,
                    'processing_time': 'Not tracked',  # Could add timing
                    'memory_usage': 'Not tracked'
                }
            }
            
            self.logger.info(f"Successfully segmented {Path(image_path).name} "
                           f"using {model_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation failed for {image_path}: {e}")
            raise
    
    def segment_batch(self, image_dir: str, output_dir: str = None,
                    file_pattern: str = "*.nii.gz", model_type: str = 'auto',
                    quality: str = 'default', n_jobs: int = 1) -> List[Dict]:
        """
        Segment ROI from multiple images in a directory.
        
        Args:
            image_dir: Directory containing input NIfTI images
            output_dir: Directory to save output segmentations
            file_pattern: File pattern to match (e.g., "*.nii.gz")
            model_type: Model type to use ('auto', '2d', '3d')
            quality: Quality level ('default', 'high_quality')
            n_jobs: Number of parallel jobs (not implemented yet)
            
        Returns:
            List of segmentation results for each image
        """
        # Create output structure
        if output_dir is None:
            output_dir = str(Path(image_dir).parent / "segmentations")
        
        output_structure = create_output_structure(output_dir)
        
        # Find input images
        input_files = list(Path(image_dir).glob(file_pattern))
        
        if not input_files:
            raise ValueError(f"No files found matching pattern {file_pattern} in {image_dir}")
        
        self.logger.info(f"Found {len(input_files)} images for batch segmentation")
        
        # Process images
        results = []
        
        # For now, process sequentially (can be parallelized later)
        for image_file in input_files:
            try:
                # Generate output path
                output_filename = f"{image_file.stem}_seg.nii.gz"
                output_path = Path(output_structure['base']) / output_filename
                
                # Segment image
                result = self.segment_image(
                    str(image_file),
                    str(output_path),
                    model_type=model_type,
                    quality=quality
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to segment {image_file}: {e}")
                results.append({
                    'input_path': str(image_file),
                    'error': str(e)
                })
        
        # Generate batch summary
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        self.logger.info(f"Batch segmentation completed: "
                       f"{len(successful)} successful, {len(failed)} failed")
        
        # Save batch report
        batch_report = self._generate_batch_report(results, analysis=self.image_analyzer.batch_analyze([str(f) for f in input_files]))
        report_path = Path(output_structure['reports']) / "batch_report.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump(batch_report, f, indent=2, default=str)
        
        return results
    
    def _ensure_model_loaded(self, model_name: str, model_type: str) -> bool:
        """Ensure that the required model is loaded."""
        if model_name in self.model_manager.list_loaded_models():
            return True
        
        # Try to load model (assuming models are in a default location)
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'pretrained')
        
        # Look for model weights
        model_path = self._find_model_weights(model_type, model_dir)
        
        return self.model_manager.load_model(
            model_name=model_name,
            model_type=model_type,
            model_path=model_path
        )
    
    def _find_model_weights(self, model_type: str, search_dir: str) -> Optional[str]:
        """Find model weights file for the given model type."""
        if not os.path.exists(search_dir):
            return None
        
        # Look for common weight file patterns
        patterns = [
            f"{model_type}.pth",
            f"{model_type}.pt",
            f"{model_type}_best.pth",
            f"{model_type}_final.pth",
            f"model_{model_type}.pth"
        ]
        
        for pattern in patterns:
            weight_path = os.path.join(search_dir, pattern)
            if os.path.exists(weight_path):
                return weight_path
        
        # Search for any file containing model_type
        for file in os.listdir(search_dir):
            if model_type in file and file.endswith(('.pth', '.pt')):
                return os.path.join(search_dir, file)
        
        return None
    
    def _generate_output_path(self, input_path: str, is_3d: bool) -> str:
        """Generate output path for segmentation result."""
        input_file = Path(input_path)
        output_dir = input_file.parent / "segmentations"
        output_dir.mkdir(exist_ok=True)
        
        suffix = "_3d_seg" if is_3d else "_2d_seg"
        output_filename = f"{input_file.stem}{suffix}.nii.gz"
        
        return str(output_dir / output_filename)
    
    def _postprocess_segmentation(self, segmentation: np.ndarray, 
                                analysis: Dict) -> np.ndarray:
        """Post-process segmentation result."""
        # Convert to binary if not already
        if segmentation.max() > 1:
            segmentation = (segmentation > 0.5).astype(np.uint8)
        
        # Remove small connected components
        min_volume = analysis.get('recommendations', {}).get('min_roi_volume', 10)
        segmentation = self._remove_small_components(segmentation, min_volume)
        
        # Morphological operations
        if analysis['is_3d']:
            segmentation = self._postprocess_3d(segmentation)
        else:
            segmentation = self._postprocess_2d(segmentation)
        
        return segmentation
    
    def _remove_small_components(self, segmentation: np.ndarray, 
                                 min_volume: int) -> np.ndarray:
        """Remove small connected components from segmentation."""
        try:
            from skimage.measure import label, regionprops
            from scipy import ndimage
            
            # Label connected components
            labeled = label(segmentation > 0)
            
            # Remove small components
            for region in regionprops(labeled):
                if region.area < min_volume:
                    segmentation[labeled == region.label] = 0
            
            return segmentation
            
        except ImportError:
            self.logger.warning("skimage or scipy not available for component filtering")
            return segmentation
    
    def _postprocess_3d(self, segmentation: np.ndarray) -> np.ndarray:
        """3D-specific post-processing."""
        try:
            from scipy import ndimage
            
            # Closing to fill small holes
            structure = np.ones((3, 3, 3))
            segmentation = ndimage.binary_closing(segmentation, structure, iterations=2)
            
            # Opening to remove small noise
            segmentation = ndimage.binary_opening(segmentation, structure, iterations=1)
            
            return segmentation.astype(np.uint8)
            
        except ImportError:
            return segmentation
    
    def _postprocess_2d(self, segmentation: np.ndarray) -> np.ndarray:
        """2D-specific post-processing."""
        try:
            from scipy import ndimage
            
            if len(segmentation.shape) == 3:
                # Process each slice
                for i in range(segmentation.shape[2]):
                    slice_seg = segmentation[:, :, i]
                    structure = np.ones((3, 3))
                    slice_seg = ndimage.binary_closing(slice_seg, structure, iterations=2)
                    slice_seg = ndimage.binary_opening(slice_seg, structure, iterations=1)
                    segmentation[:, :, i] = slice_seg
            else:
                # Single 2D image
                structure = np.ones((3, 3))
                segmentation = ndimage.binary_closing(segmentation, structure, iterations=2)
                segmentation = ndimage.binary_opening(segmentation, structure, iterations=1)
            
            return segmentation.astype(np.uint8)
            
        except ImportError:
            return segmentation
    
    def _generate_batch_report(self, results: List[Dict], analysis: Dict = None) -> Dict:
        """Generate comprehensive batch processing report."""
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        report = {
            'summary': {
                'total_images': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) if results else 0
            },
            'model_usage': {},
            'processing_times': {},
            'failed_cases': [{'path': r['input_path'], 'error': r['error']} for r in failed],
            'batch_analysis': analysis
        }
        
        # Analyze model usage
        for result in successful:
            model_type = result.get('model_used', 'unknown')
            report['model_usage'][model_type] = report['model_usage'].get(model_type, 0) + 1
        
        return report
    
    def get_segmentation_statistics(self, result: Dict) -> Dict:
        """Get statistics about the segmentation result."""
        if 'segmentation' not in result:
            return {'error': 'No segmentation result available'}
        
        seg = result['segmentation']
        analysis = result.get('analysis', {})
        
        stats = {
            'roi_volume_voxels': int(np.sum(seg > 0)),
            'roi_percentage': float(np.sum(seg > 0) / seg.size * 100),
            'roi_shape': seg.shape,
            'roi_slices': self._count_roi_slices(seg),
            'roi_centroid': self._calculate_centroid(seg),
            'bounding_box': self._calculate_bounding_box(seg)
        }
        
        # Add spacing information if available
        if 'slice_spacing' in analysis:
            voxel_volume = np.prod(analysis.get('voxel_size_mm', [1, 1, 1]))
            stats['roi_volume_mm3'] = stats['roi_volume_voxels'] * voxel_volume
        
        return stats
    
    def _count_roi_slices(self, segmentation: np.ndarray) -> int:
        """Count number of slices containing ROI."""
        if len(segmentation.shape) < 3:
            return int(np.sum(segmentation > 0) > 0)
        
        return int(np.sum(np.sum(np.sum(segmentation, axis=0), axis=0) > 0))
    
    def _calculate_centroid(self, segmentation: np.ndarray) -> List[float]:
        """Calculate centroid of ROI."""
        coords = np.where(segmentation > 0)
        if len(coords[0]) == 0:
            return [0.0, 0.0, 0.0]
        
        centroid = [np.mean(coords[i]) for i in range(len(coords))]
        return [float(c) for c in centroid]
    
    def _calculate_bounding_box(self, segmentation: np.ndarray) -> List[List[int]]:
        """Calculate bounding box of ROI."""
        coords = np.where(segmentation > 0)
        if len(coords[0]) == 0:
            return [[0, 0], [0, 0], [0, 0]]
        
        bbox = []
        for i in range(len(coords)):
            bbox.append([int(np.min(coords[i])), int(np.max(coords[i]))])
        
        return bbox