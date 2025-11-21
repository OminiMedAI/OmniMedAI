"""
基本功能测试
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# 尝试导入，处理依赖缺失情况
try:
    from onem_process.converters import DicomToNiftiConverter, BatchConverter
    from onem_process.processors import ROIProcessor, ImageProcessor
    from onem_process.config import ProcessingConfig, ConversionConfig, ConfigManager
    from onem_process.utils import file_utils, medical_utils
    import nibabel as nib
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestConfig(unittest.TestCase):
    """配置管理测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_processing_config_creation(self):
        """测试处理配置创建"""
        config = self.config_manager.get_processing_config()
        self.assertIsInstance(config, ProcessingConfig)
        self.assertEqual(config.normalize_method, 'z_score')
        self.assertEqual(config.resample_spacing, (1.0, 1.0, 1.0))
    
    def test_processing_config_update(self):
        """测试处理配置更新"""
        self.config_manager.update_processing_config(
            normalize_method='min_max',
            roi_padding=(20, 20, 20)
        )
        
        config = self.config_manager.get_processing_config()
        self.assertEqual(config.normalize_method, 'min_max')
        self.assertEqual(config.roi_padding, (20, 20, 20))
    
    def test_conversion_config_creation(self):
        """测试转换配置创建"""
        config = self.config_manager.get_conversion_config()
        self.assertIsInstance(config, ConversionConfig)
        self.assertTrue(config.batch_skip_existing)
        self.assertEqual(config.output_format, 'nii.gz')
    
    def test_config_export_import(self):
        """测试配置导入导出"""
        export_dir = Path(self.temp_dir) / "exported"
        self.config_manager.export_configs(export_dir)
        
        self.assertTrue((export_dir / "processing_config.json").exists())
        self.assertTrue((export_dir / "conversion_config.json").exists())


class TestUtils(unittest.TestCase):
    """工具函数测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_ensure_dir(self):
        """测试目录创建"""
        test_path = Path(self.temp_dir) / "test" / "subdir"
        result = file_utils.ensure_dir(test_path)
        
        self.assertTrue(result.exists())
        self.assertTrue(result.is_dir())
    
    def test_safe_filename(self):
        """测试安全文件名生成"""
        unsafe_name = "test<>:file|name?.txt"
        safe_name = file_utils.safe_filename(unsafe_name)
        
        self.assertNotIn("<", safe_name)
        self.assertNotIn(">", safe_name)
        self.assertNotIn(":", safe_name)
        self.assertNotIn("|", safe_name)
        self.assertNotIn("?", safe_name)
    
    def test_get_unique_filename(self):
        """测试唯一文件名生成"""
        # 创建测试文件
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.touch()
        
        unique_file = file_utils.get_unique_filename(test_file)
        
        self.assertNotEqual(str(test_file), str(unique_file))
        self.assertTrue(unique_file.name.startswith("test_"))
    
    def test_json_operations(self):
        """测试 JSON 操作"""
        test_data = {"key1": "value1", "key2": [1, 2, 3]}
        test_file = Path(self.temp_dir) / "test.json"
        
        # 保存
        file_utils.save_json(test_data, test_file)
        self.assertTrue(test_file.exists())
        
        # 加载
        loaded_data = file_utils.load_json(test_file)
        self.assertEqual(test_data, loaded_data)


class TestMedicalUtils(unittest.TestCase):
    """医学工具测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_normalize_intensity(self):
        """测试强度归一化"""
        # 创建测试数据
        data = np.random.randn(50, 50, 50).astype(np.float32)
        
        # Z-score 归一化
        normalized = medical_utils.normalize_intensity(data, method='z_score')
        self.assertAlmostEqual(normalized.mean(), 0, places=1)
        self.assertAlmostEqual(normalized.std(), 1, places=1)
        
        # Min-Max 归一化
        normalized = medical_utils.normalize_intensity(data, method='min_max')
        self.assertAlmostEqual(normalized.min(), 0, places=5)
        self.assertAlmostEqual(normalized.max(), 1, places=5)
    
    def test_apply_window(self):
        """测试窗宽窗位"""
        data = np.random.randn(50, 50, 50).astype(np.float32) * 100
        
        windowed = medical_utils.apply_window(data, window_center=0, window_width=200)
        
        self.assertGreaterEqual(windowed.min(), 0)
        self.assertLessEqual(windowed.max(), 1)
    
    def test_create_binary_mask(self):
        """测试二值掩码创建"""
        data = np.random.randn(50, 50, 50).astype(np.float32)
        
        # 阈值掩码
        mask = medical_utils.create_binary_mask(data, threshold=0.5)
        self.assertEqual(mask.shape, data.shape)
        self.assertTrue(np.all(mask >= 0))
        
        # Otsu 阈值
        mask_otsu = medical_utils.create_binary_mask(data, threshold='otsu')
        self.assertEqual(mask_otsu.shape, data.shape)


class TestImageProcessor(unittest.TestCase):
    """图像处理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ImageProcessor()
        
        # 创建测试 NIfTI 文件
        self.create_test_nifti()
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_nifti(self):
        """创建测试 NIfTI 文件"""
        data = np.random.randn(64, 64, 64).astype(np.float32)
        affine = np.eye(4)
        
        img = nib.Nifti1Image(data, affine)
        self.test_file = Path(self.temp_dir) / "test.nii.gz"
        nib.save(img, str(self.test_file))
    
    def test_normalize_image(self):
        """测试图像归一化"""
        output_file = Path(self.temp_dir) / "normalized.nii.gz"
        
        result = self.processor.normalize_image(
            str(self.test_file), 
            method='z_score',
            output_path=str(output_file)
        )
        
        self.assertEqual(result, str(output_file))
        self.assertTrue(output_file.exists())
    
    def test_crop_center(self):
        """测试中心裁剪"""
        output_file = Path(self.temp_dir) / "cropped.nii.gz"
        
        result = self.processor.crop_image(
            str(self.test_file),
            crop_size=(32, 32, 32),
            output_path=str(output_file)
        )
        
        self.assertEqual(result, str(output_file))
        self.assertTrue(output_file.exists())
        
        # 检查裁剪后的尺寸
        cropped_img = nib.load(str(output_file))
        self.assertEqual(cropped_img.shape, (32, 32, 32))
    
    def test_pad_image(self):
        """测试图像填充"""
        output_file = Path(self.temp_dir) / "padded.nii.gz"
        
        result = self.processor.pad_image(
            str(self.test_file),
            target_shape=(128, 128, 128),
            output_path=str(output_file)
        )
        
        self.assertEqual(result, str(output_file))
        self.assertTrue(output_file.exists())
        
        # 检查填充后的尺寸
        padded_img = nib.load(str(output_file))
        self.assertEqual(padded_img.shape, (128, 128, 128))


class TestROIProcessor(unittest.TestCase):
    """ROI 处理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        self.roi_processor = ROIProcessor(padding=(5, 5, 5))
        
        # 创建测试图像和掩码
        self.create_test_image_and_mask()
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_image_and_mask(self):
        """创建测试图像和掩码"""
        # 创建测试数据
        data = np.random.randn(64, 64, 64).astype(np.float32)
        
        # 创建掩码（中心区域）
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[20:44, 20:44, 20:44] = 1
        
        affine = np.eye(4)
        
        # 保存图像
        img = nib.Nifti1Image(data, affine)
        self.test_image = Path(self.temp_dir) / "image.nii.gz"
        nib.save(img, str(self.test_image))
        
        # 保存掩码
        mask_img = nib.Nifti1Image(mask, affine)
        self.test_mask = Path(self.temp_dir) / "mask.nii.gz"
        nib.save(mask_img, str(self.test_mask))
    
    def test_extract_roi_from_mask(self):
        """测试从掩码提取 ROI"""
        output_dir = Path(self.temp_dir) / "roi_output"
        
        roi_image, roi_mask = self.roi_processor.extract_roi_from_mask(
            str(self.test_image),
            str(self.test_mask),
            str(output_dir),
            roi_name="test_roi"
        )
        
        self.assertIsNotNone(roi_image)
        self.assertIsNotNone(roi_mask)
        self.assertTrue(Path(roi_image).exists())
        self.assertTrue(Path(roi_mask).exists())
    
    def test_get_roi_statistics(self):
        """测试 ROI 统计信息"""
        stats = self.roi_processor.get_roi_statistics(str(self.test_mask))
        
        self.assertIn('voxel_count', stats)
        self.assertIn('actual_volume_mm3', stats)
        self.assertIn('centroid', stats)
        self.assertGreater(stats['voxel_count'], 0)


def run_tests():
    """运行所有测试"""
    print("运行 onem_process 基本功能测试\n")
    
    if not DEPS_AVAILABLE:
        print(f"跳过测试 - 缺少依赖: {IMPORT_ERROR}")
        print("请安装所需依赖: pip install nibabel pydicom opencv-python scipy SimpleITK")
        return
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestConfig,
        TestUtils,
        TestMedicalUtils,
        TestImageProcessor,
        TestROIProcessor
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print(f"\n测试完成:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"  跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")


if __name__ == "__main__":
    run_tests()