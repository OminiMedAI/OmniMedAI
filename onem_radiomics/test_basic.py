"""
Basic tests for onem_radiomics module

This script contains basic tests to verify the functionality of the onem_radiomics module.
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
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False


class TestRadiomicsConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        from onem_radiomics.config.settings import RadiomicsConfig
        self.config = RadiomicsConfig()
    
    def test_default_config(self):
        """Test default configuration."""
        self.assertEqual(self.config.bin_width, 25)
        self.assertEqual(self.config.interpolator, 'sitkBSpline')
        self.assertEqual(len(self.config.feature_types), 6)
        self.assertIn('firstorder', self.config.feature_types)
        self.assertIn('glcm', self.config.feature_types)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        self.config.feature_types = ['firstorder', 'glcm']
        self.config.bin_width = 16
        self.config.validate()  # Should not raise
        
        # Invalid interpolator
        with self.assertRaises(ValueError):
            self.config.interpolator = 'invalid_interpolator'
            self.config.validate()
        
        # Invalid bin width
        with self.assertRaises(ValueError):
            self.config.bin_width = -1
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
            self.assertEqual(self.config.bin_width, loaded_config.bin_width)
            self.assertEqual(self.config.feature_types, loaded_config.feature_types)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_preset_configs(self):
        """Test preset configurations."""
        from onem_radiomics.config.settings import PRESET_CONFIGS
        
        self.assertIn('standard', PRESET_CONFIGS)
        self.assertIn('comprehensive', PRESET_CONFIGS)
        self.assertIn('ct_lung', PRESET_CONFIGS)
        
        # Check that presets are valid
        for preset_name, preset_config in PRESET_CONFIGS.items():
            self.assertIsInstance(preset_config.bin_width, int)
            self.assertGreater(preset_config.bin_width, 0)
            self.assertIsInstance(preset_config.feature_types, list)


class TestFileUtilities(unittest.TestCase):
    """Test file utility functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_output_structure(self):
        """Test output structure creation."""
        from onem_radiomics.utils.file_utils import create_output_structure
        
        structure = create_output_structure(self.temp_dir)
        
        self.assertTrue(os.path.exists(structure['base']))
        self.assertTrue(os.path.exists(structure['csv_files']))
        self.assertTrue(os.path.exists(structure['logs']))
        self.assertTrue(os.path.exists(structure['temp']))
        self.assertTrue(os.path.exists(structure['reports']))
    
    def test_validate_file_paths(self):
        """Test file path validation."""
        from onem_radiomics.utils.file_utils import validate_file_paths
        
        # Create test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.nii.gz")
            Path(file_path).touch()
            test_files.append(file_path)
        
        # Add non-existent file
        test_files.append(os.path.join(self.temp_dir, "nonexistent.nii.gz"))
        
        existing, missing = validate_file_paths(test_files)
        
        self.assertEqual(len(existing), 3)
        self.assertEqual(len(missing), 1)
        self.assertIn("nonexistent.nii.gz", missing[0])
    
    @unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel not available")
    def test_get_file_info(self):
        """Test file information extraction."""
        from onem_radiomics.utils.file_utils import get_file_info
        
        # Create a simple test NIfTI file
        data = np.random.rand(10, 10, 10)
        img = nib.Nifti1Image(data, np.eye(4))
        test_file = os.path.join(self.temp_dir, "test.nii.gz")
        nib.save(img, test_file)
        
        info = get_file_info(test_file)
        
        self.assertEqual(info['shape'], (10, 10, 10))
        self.assertEqual(info['voxel_size_mm'], (1.0, 1.0, 1.0))
        self.assertGreater(info['file_size_mb'], 0)
        self.assertEqual(info['total_voxels'], 1000)


class TestRadiomicsUtilities(unittest.TestCase):
    """Test radiomics utility functions."""
    
    def test_format_feature_names(self):
        """Test feature name formatting."""
        from onem_radiomics.utils.radiomics_utils import format_feature_names
        
        # Test various feature name formats
        test_cases = [
            ("original_firstorder_Mean", "Original_Firstorder_Mean"),
            ("original_glcm_Correlation", "Original_GLCM_Correlation"),
            ("original_glrlm_LongRunLowGrayLevelEmphasis", "Original_GLRLM_LongRunLowGrayLevelEmphasis"),
            ("test_feature_name", "Test_Feature_Name")
        ]
        
        for input_name, expected_output in test_cases:
            result = format_feature_names(input_name)
            self.assertEqual(result, expected_output)
    
    def test_validate_features(self):
        """Test feature validation."""
        from onem_radiomics.utils.radiomics_utils import validate_features
        
        # Test feature dictionary with various data types
        test_features = {
            'feature1': 1.5,
            'feature2': 2,
            'feature3': np.float64(3.14),
            'feature4': np.nan,
            'feature5': np.inf,
            'feature6': 'string_value',
            'feature7': None
        }
        
        validated = validate_features(test_features)
        
        # Should only keep numeric values, replace NaN/Inf with None
        self.assertEqual(validated['feature1'], 1.5)
        self.assertEqual(validated['feature2'], 2.0)
        self.assertEqual(validated['feature3'], 3.14)
        self.assertIsNone(validated['feature4'])
        self.assertIsNone(validated['feature5'])
        self.assertNotIn('feature6', validated)  # String should be removed
        self.assertNotIn('feature7', validated)  # None should be removed
    
    def test_feature_statistics(self):
        """Test feature statistics calculation."""
        from onem_radiomics.utils.radiomics_utils import calculate_feature_statistics
        
        # Create test data
        data = np.random.normal(0, 1, (100, 10))  # 100 samples, 10 features
        
        stats = calculate_feature_statistics(data)
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('n_samples', stats)
        self.assertIn('n_features', stats)
        
        self.assertEqual(stats['n_samples'], 100)
        self.assertEqual(stats['n_features'], 10)
    
    def test_find_constant_features(self):
        """Test constant feature detection."""
        import pandas as pd
        from onem_radiomics.utils.radiomics_utils import find_constant_features
        
        # Create test DataFrame
        df = pd.DataFrame({
            'feature1': [1.0, 1.0, 1.0, 1.0],  # Constant
            'feature2': [1.0, 2.0, 3.0, 4.0],  # Variable
            'feature3': [5.0, 5.0, 5.0, 5.0],  # Constant
            'feature4': [0.1, 0.2, 0.3, 0.4]   # Variable
        })
        
        constant_features = find_constant_features(df)
        
        self.assertIn('feature1', constant_features)
        self.assertIn('feature3', constant_features)
        self.assertNotIn('feature2', constant_features)
        self.assertNotIn('feature4', constant_features)


class TestRadiomicsExtractor(unittest.TestCase):
    """Test radiomics extractor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(NIBABEL_AVAILABLE and RADIOMICS_AVAILABLE, 
                        "nibabel or pyradiomics not available")
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        from onem_radiomics.extractors.radiomics_extractor import RadiomicsExtractor
        from onem_radiomics.config.settings import RadiomicsConfig
        
        # Test default initialization
        extractor = RadiomicsExtractor()
        self.assertIsNotNone(extractor.extractor)
        self.assertIsNotNone(extractor.config)
        
        # Test custom configuration
        config = RadiomicsConfig(feature_types=['firstorder'], bin_width=16)
        extractor = RadiomicsExtractor(config)
        self.assertEqual(extractor.config.bin_width, 16)
        self.assertEqual(extractor.config.feature_types, ['firstorder'])
    
    @unittest.skipUnless(NIBABEL_AVAILABLE and RADIOMICS_AVAILABLE, 
                        "nibabel or pyradiomics not available")
    def test_feature_descriptions(self):
        """Test feature description functionality."""
        from onem_radiomics.extractors.radiomics_extractor import RadiomicsExtractor
        
        extractor = RadiomicsExtractor()
        descriptions = extractor.get_feature_descriptions()
        
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)
        
        # Check that descriptions are formatted correctly
        for feature_name, description in descriptions.items():
            self.assertIsInstance(feature_name, str)
            self.assertIsInstance(description, str)
    
    def test_dependency_checking(self):
        """Test dependency checking in extractor."""
        from onem_radiomics.extractors.radiomics_extractor import RadiomicsExtractor
        
        if not NIBABEL_AVAILABLE:
            with self.assertRaises(ImportError):
                RadiomicsExtractor()
        elif not RADIOMICS_AVAILABLE:
            with self.assertRaises(ImportError):
                RadiomicsExtractor()


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running onem_radiomics basic tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRadiomicsConfig))
    test_suite.addTest(unittest.makeSuite(TestFileUtilities))
    test_suite.addTest(unittest.makeSuite(TestRadiomicsUtilities))
    test_suite.addTest(unittest.makeSuite(TestRadiomicsExtractor))
    
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
    print("onem_radiomics Basic Tests")
    print("==========================")
    
    # Check dependencies
    missing_deps = []
    if not NIBABEL_AVAILABLE:
        missing_deps.append("nibabel")
    if not RADIOMICS_AVAILABLE:
        missing_deps.append("pyradiomics")
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests will be skipped.")
        print("Install with: pip install " + " ".join(missing_deps))
        print()
    
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