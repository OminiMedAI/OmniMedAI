"""
Basic tests for onem_path module

This script contains basic tests to verify the functionality of onem_path module.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestPathologyConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        from onem_path.config.settings import PathologyConfig
        self.config = PathologyConfig()
    
    def test_default_config(self):
        """Test default configuration."""
        self.assertEqual(self.config.output_format, 'json')
        self.assertEqual(self.config.device, 'auto')
        self.assertTrue(self.config.extract_cellprofiler_features)
        self.assertTrue(self.config.extract_titan_features)
        self.assertEqual(self.config.image_size, (224, 224))
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        self.config.output_format = 'csv'
        self.config.device = 'cpu'
        self.config.titan_backbone = 'resnet50'
        self.config.validate()  # Should not raise
        
        # Invalid output format
        with self.assertRaises(ValueError):
            self.config.output_format = 'invalid'
            self.config.validate()
        
        # Invalid device
        with self.assertRaises(ValueError):
            self.config.device = 'invalid'
            self.config.validate()
        
        # Invalid image size
        with self.assertRaises(ValueError):
            self.config.image_size = (224, )  # Incomplete
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
            self.assertEqual(self.config.output_format, loaded_config.output_format)
            self.assertEqual(self.config.titan_backbone, loaded_config.titan_backbone)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_preset_configs(self):
        """Test preset configurations."""
        from onem_path.config.settings import PRESET_CONFIGS
        
        self.assertIn('default', PRESET_CONFIGS)
        self.assertIn('cellprofiler_only', PRESET_CONFIGS)
        self.assertIn('titan_only', PRESET_CONFIGS)
        self.assertIn('high_quality', PRESET_CONFIGS)
        
        # Check that presets are valid
        for preset_name, preset_config in PRESET_CONFIGS.items():
            self.assertIsInstance(preset_config.titan_feature_dim, int)
            self.assertGreater(preset_config.titan_feature_dim, 0)
            self.assertIn(preset_config.device, ['auto', 'cpu', 'cuda'])


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image files
        self.test_files = self._create_test_images()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_images(self):
        """Create test image files for testing."""
        test_files = []
        
        if PIL_AVAILABLE and NUMPY_AVAILABLE:
            # Create simple test images
            for i in range(3):
                # Create random RGB image
                image_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                image = Image.fromarray(image_data)
                
                file_path = os.path.join(self.temp_dir, f"test_image_{i}.png")
                image.save(file_path)
                test_files.append(file_path)
        
        return test_files
    
    def test_get_pathology_files(self):
        """Test pathology file discovery."""
        from onem_path.utils.file_utils import get_pathology_files
        
        if not self.test_files:
            self.skipTest("No test files created")
        
        # Test file discovery
        found_files = get_pathology_files(self.temp_dir)
        self.assertEqual(len(found_files), len(self.test_files))
        
        # Test with specific extensions
        png_files = get_pathology_files(self.temp_dir, extensions=['.png'])
        self.assertEqual(len(png_files), len(self.test_files))
        
        # Test recursive search
        all_files = get_pathology_files(self.temp_dir, recursive=True)
        self.assertGreaterEqual(len(all_files), len(self.test_files))
    
    def test_validate_pathology_image(self):
        """Test pathology image validation."""
        from onem_path.utils.file_utils import validate_pathology_image
        
        if not self.test_files:
            self.skipTest("No test files created")
        
        test_file = self.test_files[0]
        
        # Test valid image
        validation = validate_pathology_image(test_file)
        self.assertTrue(validation['valid'])
        self.assertIn('file_info', validation)
        
        # Test non-existent file
        validation = validate_pathology_image("/non/existent/path.png")
        self.assertFalse(validation['valid'])
        self.assertIn('error', validation)
    
    def test_create_output_structure(self):
        """Test output structure creation."""
        from onem_path.utils.file_utils import create_output_structure
        
        structure = create_output_structure(self.temp_dir)
        
        self.assertTrue(os.path.exists(structure['base']))
        self.assertTrue(os.path.exists(structure['features']))
        self.assertTrue(os.path.exists(structure['reports']))
        self.assertTrue(os.path.exists(structure['logs']))


class TestImageUtils(unittest.TestCase):
    """Test image utility functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image
        if PIL_AVAILABLE and NUMPY_AVAILABLE:
            self.test_image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            self.test_image = Image.fromarray(self.test_image_array)
            self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
            self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_preprocess_pathology_image(self):
        """Test image preprocessing."""
        from onem_path.utils.image_utils import preprocess_pathology_image
        
        if not hasattr(self, 'test_image_path'):
            self.skipTest("No test image created")
        
        # Test basic preprocessing
        config = {
            'target_size': (32, 32),
            'normalize_intensity': True,
            'color_conversion': 'gray'
        }
        
        preprocessed = preprocess_pathology_image(self.test_image_path, config)
        
        self.assertEqual(preprocessed.shape[:2], (32, 32))  # Resized
        self.assertEqual(len(preprocessed.shape), 2)  # Grayscale conversion
    
    def test_convert_color_space(self):
        """Test color space conversion."""
        if not hasattr(self, 'test_image_array') or len(self.test_image_array.shape) != 3:
            self.skipTest("No RGB test image available")
        
        from onem_path.utils.image_utils import convert_color_space
        
        # Test RGB to HSV
        hsv = convert_color_space(self.test_image_array, 'hsv')
        self.assertEqual(hsv.shape, self.test_image_array.shape)
        
        # Test RGB to gray
        gray = convert_color_space(self.test_image_array, 'gray')
        self.assertEqual(len(gray.shape), 2)
        
        # Test invalid conversion
        original = convert_color_space(self.test_image_array, 'rgb')
        np.testing.assert_array_equal(original, self.test_image_array)
    
    def test_normalize_image_intensity(self):
        """Test intensity normalization."""
        from onem_path.utils.image_utils import (
            normalize_image_intensity, _percentile_normalization
        )
        
        # Create test data
        test_data = np.random.rand(100, 100).astype(np.float32)
        
        # Test percentile normalization
        norm_data = normalize_image_intensity(test_data, 'percentile')
        self.assertTrue(np.all(norm_data >= 0) and np.all(norm_data <= 1))
        
        # Test min-max normalization
        minmax_data = normalize_image_intensity(test_data, 'minmax')
        self.assertAlmostEqual(np.min(minmax_data), 0.0, places=5)
        self.assertAlmostEqual(np.max(minmax_data), 1.0, places=5)
    
    def test_resize_pathology_image(self):
        """Test image resizing."""
        if not hasattr(self, 'test_image_array'):
            self.skipTest("No test image available")
        
        from onem_path.utils.image_utils import resize_pathology_image
        
        # Test resizing
        resized = resize_pathology_image(self.test_image_array, (32, 32))
        self.assertEqual(resized.shape[:2], (32, 32))
        
        # Test different methods
        if NUMPY_AVAILABLE:
            nearest = resize_pathology_image(self.test_image_array, (32, 32), 'nearest')
            bilinear = resize_pathology_image(self.test_image_array, (32, 32), 'bilinear')
            self.assertEqual(nearest.shape[:2], (32, 32))
            self.assertEqual(bilinear.shape[:2], (32, 32))


class TestCellProfilerExtractor(unittest.TestCase):
    """Test CellProfiler feature extractor."""
    
    def setUp(self):
        from onem_path.extractors.cellprofiler_extractor import CellProfilerExtractor
        self.extractor = CellProfilerExtractor()
        
        # Create test data
        self.test_image = self._create_test_image()
    
    def _create_test_image(self):
        """Create test image for CellProfiler testing."""
        if PIL_AVAILABLE and NUMPY_AVAILABLE:
            # Create image with some structure
            image_data = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Add some regions
            image_data[20:40, 20:40] = [255, 0, 0]  # Red region
            image_data[60:80, 60:80] = [0, 255, 0]  # Green region
            image_data[80:100, 80:100] = [0, 0, 255]  # Blue region
            
            # Add noise
            noise = np.random.randint(0, 50, image_data.shape, dtype=np.uint8)
            image_data = np.clip(image_data + noise, 0, 255)
            
            return image_data
        else:
            return np.zeros((128, 128, 3), dtype=np.uint8)
    
    def test_module_extraction(self):
        """Test individual module extraction."""
        if not hasattr(self, 'test_image'):
            self.skipTest("No test image created")
        
        # Test morphological features
        morph_features = self.extractor._extract_morphological_features(self.test_image)
        self.assertIsInstance(morph_features, dict)
        self.assertGreater(len(morph_features), 0)
        
        # Test texture features
        texture_features = self.extractor._extract_texture_features(self.test_image)
        self.assertIsInstance(texture_features, dict)
        self.assertGreater(len(texture_features), 0)
        
        # Test intensity features
        intensity_features = self.extractor._extract_intensity_features(self.test_image)
        self.assertIsInstance(intensity_features, dict)
        self.assertGreater(len(intensity_features), 0)
    
    def test_feature_extraction_pipeline(self):
        """Test complete feature extraction pipeline."""
        from onem_path.extractors.cellprofiler_extractor import CellProfilerExtractor
        
        if not hasattr(self, 'test_image'):
            self.skipTest("No test image created")
        
        extractor = CellProfilerExtractor()
        
        # Mock the file system operations for testing
        original_load_image = extractor._load_image
        extractor._load_image = lambda path: self.test_image
        
        try:
            # Test extraction with modules
            modules = ['morphological', 'intensity']
            result = extractor.extract_features(
                image_path="mock_path.png",
                modules=modules
            )
            
            self.assertIn('features', result)
            self.assertIn('module_results', result)
            self.assertIn('image_metadata', result)
            self.assertIn('extraction_metadata', result)
            
            # Check that requested modules were processed
            for module in modules:
                self.assertIn(f'module_{module}', result['module_results'])
                
        except Exception as e:
            self.fail(f"Feature extraction pipeline failed: {e}")
        
        # Restore original method
        extractor._load_image = original_load_image


class TestTITANExtractor(unittest.TestCase):
    """Test TITAN feature extractor."""
    
    def setUp(self):
        from onem_path.extractors.titan_extractor import TITANExtractor
        
        # Create test config
        config = {
            'titan_backbone': 'resnet18',  # Smaller model for testing
            'titan_feature_dim': 128,
            'titan_use_attention': False,
            'device': 'cpu'  # Use CPU for testing
        }
        
        self.extractor = TITANExtractor(config)
    
    def test_model_initialization(self):
        """Test TITAN model initialization."""
        if self.extractor.model is not None:
            self.assertIsNotNone(self.extractor.model)
            self.assertEqual(self.extractor.model.get_feature_dim(), 128)
    
    def test_feature_extraction(self):
        """Test TITAN feature extraction."""
        from onem_path.extractors.titan_extractor import TITANExtractor
        
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        # Create test image
        if PIL_AVAILABLE and NUMPY_AVAILABLE:
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Mock file loading
            original_preprocess = self.extractor._load_and_preprocess_image
            self.extractor._load_and_preprocess_image = lambda path: torch.randn(1, 3, 224, 224)
            
            try:
                result = self.extractor.extract_features("mock_path.jpg")
                
                self.assertIn('features', result)
                self.assertIn('model_info', result)
                self.assertIn('extraction_metadata', result)
                self.assertEqual(len(result['features']), 128)
                
            except Exception as e:
                self.fail(f"TITAN feature extraction failed: {e}")
            
            finally:
                self.extractor._load_and_preprocess_image = original_preprocess
        else:
            self.skipTest("PIL/NumPy not available")
    
    def test_model_validation(self):
        """Test TITAN model validation."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from onem_path.utils.titan_utils import validate_titan_model
        
        if self.extractor.model is not None:
            # Create test input
            test_input = torch.randn(1, 3, 224, 224)
            
            result = validate_titan_model(self.extractor.model, test_input)
            
            self.assertTrue(result['model_valid'])
            self.assertIn('output_shape', result)
            self.assertIn('feature_dim', result)


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running onem_path basic tests...")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not PIL_AVAILABLE:
        missing_deps.append("PIL")
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    if not PANDAS_AVAILABLE:
        missing_deps.append("pandas")
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests will be skipped.")
        print("Install with: pip install " + " ".join(missing_deps))
        print()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPathologyConfig))
    test_suite.addTest(unittest.makeSuite(TestFileUtils))
    test_suite.addTest(unittest.makeSuite(TestImageUtils))
    test_suite.addTest(unittest.makeSuite(TestCellProfilerExtractor))
    test_suite.addTest(unittest.makeSuite(TestTITANExtractor))
    
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
    print("onem_path Basic Tests")
    print("======================")
    
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