"""
Basic tests for onem_segment module

This script contains basic tests to verify the functionality of the onem_segment module.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSegmentationConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        from onem_segment.config.settings import SegmentationConfig
        self.config = SegmentationConfig()
    
    def test_default_config(self):
        """Test default configuration."""
        self.assertEqual(self.config.model_type, 'auto')
        self.assertEqual(self.config.quality_level, 'default')
        self.assertEqual(self.config.normalization, 'z_score')
        self.assertEqual(self.config.min_3d_slices, 30)
        self.assertTrue(self.config.morphological_operations)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        self.config.model_type = '2d'
        self.config.min_3d_slices = 25
        self.config.probability_threshold = 0.7
        self.config.validate()  # Should not raise
        
        # Invalid model type
        with self.assertRaises(ValueError):
            self.config.model_type = 'invalid'
            self.config.validate()
        
        # Invalid probability threshold
        with self.assertRaises(ValueError):
            self.config.probability_threshold = 1.5
            self.config.validate()
        
        # Invalid min_3d_slices
        with self.assertRaises(ValueError):
            self.config.min_3d_slices = -1
            self.config.validate()
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            self.config.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load configuration
            loaded_config = self.config.__class__.load(temp_path)
            self.assertEqual(self.config.model_type, loaded_config.model_type)
            self.assertEqual(self.config.min_3d_slices, loaded_config.min_3d_slices)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_preset_configs(self):
        """Test preset configurations."""
        from onem_segment.config.settings import PRESET_CONFIGS
        
        self.assertIn('default', PRESET_CONFIGS)
        self.assertIn('high_quality', PRESET_CONFIGS)
        self.assertIn('fast', PRESET_CONFIGS)
        
        # Check that presets are valid
        for preset_name, preset_config in PRESET_CONFIGS.items():
            self.assertIsInstance(preset_config.min_3d_slices, int)
            self.assertGreater(preset_config.min_3d_slices, 0)
            self.assertIn(preset_config.model_type, ['auto', '2d', '3d'])


class TestImageDimensionAnalyzer(unittest.TestCase):
    """Test image dimension analysis."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test NIfTI files
        self.test_files = self._create_test_nifti_files()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_nifti_files(self):
        """Create test NIfTI files for testing."""
        if not NIBABEL_AVAILABLE:
            return []
        
        test_files = []
        
        # Create 2D image (single slice)
        data_2d = np.random.rand(64, 64) * 100
        img_2d = nib.Nifti1Image(data_2d, np.eye(4))
        file_2d = os.path.join(self.temp_dir, "test_2d.nii.gz")
        nib.save(img_2d, file_2d)
        test_files.append(file_2d)
        
        # Create 3D image with few slices (should recommend 2D)
        data_small_3d = np.random.rand(64, 64, 10) * 100
        img_small_3d = nib.Nifti1Image(data_small_3d, np.eye(4))
        file_small_3d = os.path.join(self.temp_dir, "test_small_3d.nii.gz")
        nib.save(img_small_3d, file_small_3d)
        test_files.append(file_small_3d)
        
        # Create 3D image with many slices (should recommend 3D)
        data_large_3d = np.random.rand(64, 64, 50) * 100
        img_large_3d = nib.Nifti1Image(data_large_3d, np.eye(4))
        file_large_3d = os.path.join(self.temp_dir, "test_large_3d.nii.gz")
        nib.save(img_large_3d, file_large_3d)
        test_files.append(file_large_3d)
        
        return test_files
    
    @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
    def test_image_analysis(self):
        """Test single image analysis."""
        from onem_segment.utils.image_analyzer import ImageDimensionAnalyzer
        
        analyzer = ImageDimensionAnalyzer()
        
        # Test with 2D image
        file_2d = self.test_files[0]
        result = analyzer.analyze_image(file_2d)
        
        self.assertIn('shape', result)
        self.assertIn('is_3d', result)
        self.assertIn('content_analysis', result)
        self.assertIn('quality_metrics', result)
        self.assertIn('recommendations', result)
        
        # 2D image should not recommend 3D
        self.assertFalse(result['is_3d'])
    
    @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
    def test_batch_analysis(self):
        """Test batch image analysis."""
        from onem_segment.utils.image_analyzer import ImageDimensionAnalyzer
        
        analyzer = ImageDimensionAnalyzer()
        
        # Test batch analysis
        batch_result = analyzer.batch_analyze(self.test_files)
        
        self.assertIn('individual_results', batch_result)
        self.assertIn('summary', batch_result)
        self.assertIn('batch_recommendations', batch_result)
        
        summary = batch_result['summary']
        self.assertEqual(summary['total_images'], len(self.test_files))
        self.assertGreaterEqual(summary['recommend_3d'], 0)
        self.assertGreaterEqual(summary['recommend_2d'], 0)


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
    def test_nifti_operations(self):
        """Test NIfTI file operations."""
        from onem_segment.utils.file_utils import (
            load_nifti_image, save_nifti_image, validate_nifti_file
        )
        
        # Create test data
        test_data = np.random.rand(32, 32, 16) * 100
        test_affine = np.eye(4)
        
        # Save test image
        test_file = os.path.join(self.temp_dir, "test.nii.gz")
        save_nifti_image(test_data, test_affine, None, test_file)
        
        self.assertTrue(os.path.exists(test_file))
        
        # Load test image
        loaded_data, header, affine = load_nifti_image(test_file)
        
        self.assertEqual(loaded_data.shape, test_data.shape)
        np.testing.assert_array_almost_equal(affine, test_affine)
        
        # Validate file
        validation = validate_nifti_file(test_file)
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['file_info']['shape'], test_data.shape)
    
    def test_create_output_structure(self):
        """Test output structure creation."""
        from onem_segment.utils.file_utils import create_output_structure
        
        structure = create_output_structure(self.temp_dir)
        
        self.assertTrue(os.path.exists(structure['base']))
        self.assertTrue(os.path.exists(structure['segmentations']))
        self.assertTrue(os.path.exists(structure['probabilities']))
        self.assertTrue(os.path.exists(structure['reports']))
        self.assertTrue(os.path.exists(structure['logs']))
    
    def test_get_nifti_files(self):
        """Test NIfTI file discovery."""
        from onem_segment.utils.file_utils import get_nifti_files
        
        @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
        def create_test_files():
            import nibabel as nib
            # Create test files
            for name in ['test1.nii.gz', 'test2.nii', 'test3.nii.gz']:
                data = np.random.rand(16, 16, 16)
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, os.path.join(self.temp_dir, name))
        
        create_test_files()
        
        # Get files
        nifti_files = get_nifti_files(self.temp_dir)
        
        self.assertEqual(len(nifti_files), 3)
        for file_path in nifti_files:
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(file_path.endswith(('.nii', '.nii.gz')))


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_normalize_image(self):
        """Test image normalization."""
        from onem_segment.utils.preprocessing import normalize_image
        
        # Create test data
        test_data = np.array([1, 2, 3, 4, 5], dtype=float)
        
        # Z-score normalization
        z_scored = normalize_image(test_data, 'z_score')
        self.assertAlmostEqual(np.mean(z_scored), 0, places=6)
        self.assertAlmostEqual(np.std(z_scored), 1, places=6)
        
        # Min-max normalization
        min_maxed = normalize_image(test_data, 'min_max')
        self.assertAlmostEqual(np.min(min_maxed), 0, places=6)
        self.assertAlmostEqual(np.max(min_maxed), 1, places=6)
    
    def test_resize_image(self):
        """Test image resizing."""
        from onem_segment.utils.preprocessing import resize_image
        
        # Create 2D test data
        test_data_2d = np.random.rand(32, 32)
        resized_2d = resize_image(test_data_2d, (64, 64))
        
        self.assertEqual(resized_2d.shape, (64, 64))
        
        # Create 3D test data
        test_data_3d = np.random.rand(16, 16, 8)
        resized_3d = resize_image(test_data_3d, (32, 32))
        
        self.assertEqual(resized_3d.shape[:2], (32, 32))
        self.assertEqual(resized_3d.shape[2], 8)  # Third dimension should be preserved
    
    def test_clip_intensity(self):
        """Test intensity clipping."""
        from onem_segment.utils.preprocessing import clip_intensity
        
        # Create test data with outliers
        test_data = np.array([1, 2, 3, 4, 100], dtype=float)
        
        # Clip to 1-99 percentiles
        clipped = clip_intensity(test_data, (20, 80))
        
        # Values should be clipped
        self.assertLessEqual(np.max(clipped), test_data[3])  # Should clip the 100


class TestROISegmenter(unittest.TestCase):
    """Test ROI segmenter functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_segmenter_initialization(self):
        """Test segmenter initialization."""
        try:
            from onem_segment.segmenters.roi_segmenter import ROISegmenter
            
            # Should raise without nibabel
            if not NIBABEL_AVAILABLE:
                with self.assertRaises(ImportError):
                    ROISegmenter()
            else:
                segmenter = ROISegmenter()
                self.assertIsNotNone(segmenter.image_analyzer)
                self.assertIsNotNone(segmenter.model_manager)
                
        except ImportError as e:
            # Expected if dependencies are missing
            self.assertIn("required", str(e))
    
    @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
    def test_segmenter_statistics(self):
        """Test segmentation statistics calculation."""
        from onem_segment.segmenters.roi_segmenter import ROISegmenter
        
        segmenter = ROISegmenter()
        
        # Create test result
        test_segmentation = np.zeros((32, 32, 16), dtype=np.uint8)
        test_segmentation[10:20, 10:20, 5:10] = 1  # Add ROI
        
        test_result = {
            'segmentation': test_segmentation,
            'analysis': {'slice_spacing': 1.0, 'voxel_size_mm': [1, 1, 1]}
        }
        
        stats = segmenter.get_segmentation_statistics(test_result)
        
        self.assertIn('roi_volume_voxels', stats)
        self.assertIn('roi_percentage', stats)
        self.assertIn('roi_slices', stats)
        self.assertIn('centroid', stats)
        self.assertIn('bounding_box', stats)
        
        # Should detect some ROI voxels
        self.assertGreater(stats['roi_volume_voxels'], 0)


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running onem_segment basic tests...")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not NIBABEL_AVAILABLE:
        missing_deps.append("nibabel")
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests will be skipped.")
        print("Install with: pip install " + " ".join(missing_deps))
        print()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSegmentationConfig))
    test_suite.addTest(unittest.makeSuite(TestImageDimensionAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestFileUtils))
    test_suite.addTest(unittest.makeSuite(TestPreprocessing))
    test_suite.addTest(unittest.makeSuite(TestROISegmenter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


def main():
    """Main function to run tests."""
    print("onem_segment Basic Tests")
    print("=========================")
    
    # Run tests
    success = run_basic_tests()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())