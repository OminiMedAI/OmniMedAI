"""
Radiomics Feature Extractor

This module provides comprehensive radiomics feature extraction capabilities for medical images.
Supports extraction of various radiomics features from NIfTI format images and masks.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import required dependencies
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Install with: pip install nibabel")

try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    logging.warning("pyradiomics not available. Install with: pip install pyradiomics")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK not available. Install with: pip install SimpleITK")

from ..config.settings import RadiomicsConfig
from ..utils.file_utils import load_image_mask_pair, get_matching_files, validate_file_paths
from ..utils.radiomics_utils import setup_radiomics_features, format_feature_names


class RadiomicsExtractor:
    """
    Main class for extracting radiomics features from medical images.
    
    Supports various feature types including first-order statistics, texture features,
    and morphological features. Provides batch processing capabilities and flexible
    configuration options.
    """
    
    def __init__(self, config: Optional[RadiomicsConfig] = None):
        """
        Initialize the radiomics extractor.
        
        Args:
            config: Configuration object for feature extraction parameters
        """
        self.config = config or RadiomicsConfig()
        self.extractor = None
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required. Install with: pip install nibabel")
        if not RADIOMICS_AVAILABLE:
            raise ImportError("pyradiomics is required. Install with: pip install pyradiomics")
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")
        
        # Setup radiomics extractor
        self._setup_extractor()
    
    def _setup_extractor(self):
        """Setup the PyRadiomics feature extractor with configuration."""
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Apply image preprocessing settings
        if self.config.resampled_pixel_spacing:
            self.extractor.settings['resampledPixelSpacing'] = self.config.resampled_pixel_spacing
        
        self.extractor.settings['interpolator'] = self.config.interpolator
        self.extractor.settings['binWidth'] = self.config.bin_width
        
        if self.config.normalize:
            self.extractor.settings['normalize'] = True
        if self.config.normalize_scale:
            self.extractor.settings['normalizeScale'] = self.config.normalize_scale
        
        # Apply weighting
        if self.config.weight_center and self.config.weight_radius:
            self.extractor.settings['weightingNorm'] = 'euclidean'
            self.extractor.settings['gldm_a'] = 2
        
        # Enable/disable feature classes
        setup_radiomics_features(self.extractor, self.config.feature_types)
        
        # Apply custom settings
        for key, value in self.config.custom_settings.items():
            self.extractor.settings[key] = value
        
        self.logger.info(f"Radiomics extractor initialized with features: {self.config.feature_types}")
    
    def extract_features(self, image_path: str, mask_path: str, 
                        patient_id: Optional[str] = None) -> Dict:
        """
        Extract radiomics features from a single image-mask pair.
        
        Args:
            image_path: Path to the NIfTI image file
            mask_path: Path to the NIfTI mask file
            patient_id: Optional patient identifier for results
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Load and validate image-mask pair
        try:
            image, mask = load_image_mask_pair(image_path, mask_path)
        except Exception as e:
            self.logger.error(f"Failed to load image-mask pair: {e}")
            raise
        
        # Generate patient ID if not provided
        if patient_id is None:
            patient_id = Path(image_path).stem
        
        try:
            # Extract features using PyRadiomics
            features = self.extractor.execute(image_path, mask_path)
            
            # Format feature names and convert to appropriate types
            formatted_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float, np.number)):
                    formatted_features[format_feature_names(key)] = float(value)
            
            # Add metadata
            result = {
                'patient_id': patient_id,
                'image_path': image_path,
                'mask_path': mask_path,
                'features': formatted_features,
                'metadata': {
                    'feature_types': self.config.feature_types,
                    'bin_width': self.config.bin_width,
                    'resampled_pixel_spacing': self.config.resampled_pixel_spacing,
                    'image_shape': image.shape if hasattr(image, 'shape') else 'unknown',
                    'mask_shape': mask.shape if hasattr(mask, 'shape') else 'unknown'
                }
            }
            
            self.logger.info(f"Successfully extracted {len(formatted_features)} features for {patient_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {patient_id}: {e}")
            raise
    
    def batch_extract_features(self, images_dir: str, masks_dir: str,
                            output_csv_path: str, file_pattern: str = "*.nii.gz",
                            n_jobs: int = 1) -> pd.DataFrame:
        """
        Batch extract radiomics features from multiple image-mask pairs.
        
        Args:
            images_dir: Directory containing NIfTI image files
            masks_dir: Directory containing NIfTI mask files
            output_csv_path: Path to save the CSV output file
            file_pattern: File pattern to match (e.g., "*.nii.gz")
            n_jobs: Number of parallel jobs for processing
            
        Returns:
            DataFrame containing extracted features for all patients
        """
        # Get matching image-mask pairs
        matching_pairs = get_matching_files(images_dir, masks_dir, file_pattern)
        
        if not matching_pairs:
            raise ValueError(f"No matching image-mask pairs found in {images_dir} and {masks_dir}")
        
        self.logger.info(f"Found {len(matching_pairs)} image-mask pairs for processing")
        
        # Prepare results storage
        all_results = []
        
        if n_jobs == 1:
            # Sequential processing
            for image_path, mask_path in matching_pairs:
                try:
                    patient_id = Path(image_path).stem
                    result = self.extract_features(image_path, mask_path, patient_id)
                    all_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    continue
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_pair = {}
                
                for image_path, mask_path in matching_pairs:
                    patient_id = Path(image_path).stem
                    future = executor.submit(self.extract_features, image_path, mask_path, patient_id)
                    future_to_pair[future] = (image_path, patient_id)
                
                for future in as_completed(future_to_pair):
                    image_path, patient_id = future_to_pair[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to process {image_path}: {e}")
                        continue
        
        if not all_results:
            raise RuntimeError("No features were successfully extracted")
        
        # Convert to DataFrame
        df = self._results_to_dataframe(all_results)
        
        # Save to CSV
        self._save_to_csv(df, output_csv_path)
        
        self.logger.info(f"Successfully extracted features for {len(df)} patients")
        self.logger.info(f"Results saved to: {output_csv_path}")
        
        return df
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert extraction results to a pandas DataFrame.
        
        Args:
            results: List of extraction result dictionaries
            
        Returns:
            DataFrame with features in columns and patients in rows
        """
        rows = []
        
        for result in results:
            row = {
                'patient_id': result['patient_id'],
                'image_path': result['image_path'],
                'mask_path': result['mask_path']
            }
            
            # Add features
            row.update(result['features'])
            
            # Add metadata columns
            for key, value in result['metadata'].items():
                row[f'meta_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_to_csv(self, df: pd.DataFrame, output_path: str):
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Output CSV file path
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Also save feature summary
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Radiomics Feature Extraction Summary\n")
            f.write(f"=====================================\n\n")
            f.write(f"Total patients: {len(df)}\n")
            f.write(f"Total features: {len([col for col in df.columns if not col.startswith('meta_') and col not in ['patient_id', 'image_path', 'mask_path']])}\n\n")
            
            f.write("Feature types used:\n")
            for feature_type in self.config.feature_types:
                f.write(f"  - {feature_type}\n")
            
            f.write(f"\nExtraction parameters:\n")
            f.write(f"  - Bin width: {self.config.bin_width}\n")
            f.write(f"  - Interpolator: {self.config.interpolator}\n")
            if self.config.resampled_pixel_spacing:
                f.write(f"  - Resampled pixel spacing: {self.config.resampled_pixel_spacing}\n")
        
        self.logger.info(f"Feature summary saved to: {summary_path}")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of available radiomics features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        if not self.extractor:
            raise RuntimeError("Extractor not initialized")
        
        feature_info = {}
        
        # Get feature classes and their descriptions
        for feature_class in self.config.feature_types:
            try:
                # Get feature names for this class
                features = self.extractor.getFeatureNames(feature_class)
                for feature_name in features:
                    formatted_name = format_feature_names(feature_name)
                    feature_info[formatted_name] = f"{feature_class}: {feature_name}"
            except:
                continue
        
        return feature_info
    
    def validate_extraction_setup(self, test_image_path: str, test_mask_path: str) -> Dict:
        """
        Validate the extraction setup with a test image-mask pair.
        
        Args:
            test_image_path: Path to test image
            test_mask_path: Path to test mask
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'test_features': None
        }
        
        try:
            # Test file existence
            if not os.path.exists(test_image_path):
                validation_results['errors'].append(f"Test image not found: {test_image_path}")
                validation_results['valid'] = False
            
            if not os.path.exists(test_mask_path):
                validation_results['errors'].append(f"Test mask not found: {test_mask_path}")
                validation_results['valid'] = False
            
            if not validation_results['valid']:
                return validation_results
            
            # Test feature extraction
            result = self.extract_features(test_image_path, test_mask_path, "test_validation")
            validation_results['test_features'] = result
            
            # Check if any features were extracted
            if len(result['features']) == 0:
                validation_results['errors'].append("No features were extracted")
                validation_results['valid'] = False
            
            # Add warnings for potential issues
            if len(result['features']) < 10:
                validation_results['warnings'].append("Very few features extracted, check configuration")
            
            self.logger.info("Extraction setup validation completed successfully")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['valid'] = False
            self.logger.error(f"Extraction setup validation failed: {e}")
        
        return validation_results