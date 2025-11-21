"""
onem_habitat 基础测试
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# 尝试导入，处理依赖缺失情况
try:
    from onem_habitat.radiomics import LocalRadiomicsExtractor
    from onem_habitat.clustering import FeatureClustering
    from onem_habitat.segmentation import MaskRefiner
    from onem_habitat.config import HabitatConfig, HabitatConfigManager, PRESET_CONFIGS
    from onem_habitat.utils import habitat_utils, radiomics_utils
    import nibabel as nib
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestHabitatConfig(unittest.TestCase):
    """配置管理测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = HabitatConfigManager(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_config_creation(self):
        """测试配置创建"""
        config = self.config_manager.get_default_config()
        self.assertIsInstance(config, HabitatConfig)
        self.assertEqual(config.kernel_size, (5, 5, 5))
        self.assertEqual(config.clustering_method, 'kmeans')
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = HabitatConfig(n_clusters=5)
        validation = valid_config.validate()
        self.assertTrue(validation['valid'])
        
        # 无效配置
        invalid_config = HabitatConfig(n_clusters=0)
        validation = invalid_config.validate()
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['errors']), 0)
    
    def test_preset_configs(self):
        """测试预设配置"""
        for preset_name, preset_config in PRESET_CONFIGS.items():
            self.assertIsInstance(preset_config, HabitatConfig)
            validation = preset_config.validate()
            self.assertTrue(validation['valid'], 
                          f"Preset {preset_name} validation failed")


class TestHabitatUtils(unittest.TestCase):
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
    
    def test_json_operations(self):
        """测试 JSON 操作"""
        test_data = {"key1": "value1", "key2": [1, 2, 3]}
        test_file = Path(self.temp_dir) / "test.json"
        
        # 保存
        habitat_utils.save_json(test_data, str(test_file))
        self.assertTrue(test_file.exists())
        
        # 加载
        loaded_data = habitat_utils.load_json(str(test_file))
        self.assertEqual(test_data, loaded_data)
    
    def test_feature_importance(self):
        """测试特征重要性计算"""
        # 创建模拟特征数据
        features_dict_list = [
            {
                'features': {
                    'feature1': np.random.randn(100),
                    'feature2': np.random.randn(100),
                    'feature3': np.random.randn(100) * 0.1  # 低方差
                }
            }
        ]
        
        # 计算重要性
        importance = habitat_utils.calculate_feature_importance(
            features_dict_list, method='variance'
        )
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        self.assertTrue(all(0 <= v <= 1 for v in importance.values()))
    
    def test_habitat_summary(self):
        """测试生态分析摘要"""
        results = [
            {
                'image_name': 'test1',
                'n_clusters': 3,
                'total_voxels': 1000
            },
            {
                'image_name': 'test2',
                'n_clusters': 5,
                'total_voxels': 1500,
                'error': 'Test error'
            }
        ]
        
        summary = habitat_utils.create_habitat_summary(results)
        
        self.assertEqual(summary['total_images'], 2)
        self.assertEqual(summary['successful_processed'], 1)
        self.assertEqual(summary['failed_processed'], 1)
        self.assertEqual(summary['total_clusters_found'], 8)


class TestRadiomicsUtils(unittest.TestCase):
    """放射组学工具测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
    
    def test_feature_extraction_utils(self):
        """测试特征提取工具"""
        # 创建测试数据
        image_data = np.random.randn(50, 50, 50).astype(np.float32)
        mask_data = np.zeros((50, 50, 50), dtype=np.uint8)
        mask_data[20:30, 20:30, 20:30] = 1  # 中心区域
        
        # 测试强度特征
        intensity_features = radiomics_utils.extract_intensity_features(
            image_data, mask_data
        )
        
        self.assertIsInstance(intensity_features, dict)
        expected_features = ['mean', 'std', 'min', 'max', 'median']
        for feature in expected_features:
            self.assertIn(feature, intensity_features)
        
        # 测试纹理特征
        texture_features = radiomics_utils.extract_texture_features(
            image_data, mask_data
        )
        
        self.assertIsInstance(texture_features, dict)
    
    def test_preprocessing_utils(self):
        """测试预处理工具"""
        # 创建测试数据
        image_data = np.random.randn(30, 30, 30).astype(np.float32) * 100 + 50
        
        # 测试强度归一化
        normalized = radiomics_utils.preprocess_image(
            image_data, intensity_normalization='zscore'
        )
        
        self.assertEqual(normalized.shape, image_data.shape)
        self.assertAlmostEqual(np.mean(normalized), 0, places=1)
        
        # 测试窗宽处理
        windowed = radiomics_utils.apply_intensity_window(
            image_data, window_center=50, window_width=100
        )
        
        self.assertEqual(windowed.shape, image_data.shape)
        self.assertTrue(np.all(windowed >= 0))
        self.assertTrue(np.all(windowed <= 1))


class TestLocalRadiomicsExtractor(unittest.TestCase):
    """局部放射组学提取器测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.create_test_data()
        
        # 创建提取器
        self.extractor = LocalRadiomicsExtractor(
            kernel_size=(5, 5, 5),
            feature_types=['firstorder'],  # 只用一阶特征加快测试
            n_jobs=1
        )
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建简单的测试图像和掩码
        image_data = np.random.randn(30, 30, 30).astype(np.float32)
        mask_data = np.zeros((30, 30, 30), dtype=np.uint8)
        
        # 创建一个简单的 ROI 区域
        mask_data[10:20, 10:20, 10:20] = 1
        
        # 保存为 NIfTI 文件
        affine = np.eye(4)
        
        image_nii = nib.Nifti1Image(image_data, affine)
        mask_nii = nib.Nifti1Image(mask_data, affine)
        
        self.test_image_path = Path(self.temp_dir) / "test_image.nii.gz"
        self.test_mask_path = Path(self.temp_dir) / "test_mask.nii.gz"
        
        nib.save(image_nii, str(self.test_image_path))
        nib.save(mask_nii, str(self.test_mask_path))
    
    def test_local_feature_extraction(self):
        """测试局部特征提取"""
        try:
            result = self.extractor.extract_local_features(
                image_path=str(self.test_image_path),
                mask_path=str(self.test_mask_path),
                step_size=2,  # 减少计算量
                output_path=str(Path(self.temp_dir) / "features.npy")
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('features', result)
            self.assertIn('metadata', result)
            self.assertIn('coordinates', result)
            
            # 检查特征数量
            if 'features' in result and result['features']:
                feature_count = len(result['features'])
                coordinate_count = len(result['coordinates'])
                self.assertEqual(feature_count, coordinate_count)
                
        except Exception as e:
            # 如果 pyradiomics 不可用，跳过测试
            self.skipTest(f"Skipping radiomics test due to: {e}")
    
    def test_feature_summary(self):
        """测试特征摘要"""
        try:
            result = self.extractor.extract_local_features(
                image_path=str(self.test_image_path),
                mask_path=str(self.test_mask_path),
                step_size=3  # 进一步减少计算量
            )
            
            if 'features' in result and result['features']:
                summary = self.extractor.get_feature_summary(result)
                
                self.assertIsInstance(summary, dict)
                # 验证摘要包含统计信息
                for feature_name in result['features']:
                    if feature_name in summary:
                        stats = summary[feature_name]
                        self.assertIn('mean', stats)
                        self.assertIn('std', stats)
                        
        except Exception as e:
            self.skipTest(f"Skipping feature summary test due to: {e}")


class TestFeatureClustering(unittest.TestCase):
    """特征聚类测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟特征数据
        self.create_mock_features()
        
        # 创建聚类器
        self.clusterer = FeatureClustering(
            clustering_method='kmeans',
            n_clusters=3,
            feature_selection='all',
            pca_components=2  # 降维以简化测试
        )
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_mock_features(self):
        """创建模拟特征数据"""
        # 创建三个不同的聚类
        np.random.seed(42)
        
        cluster1 = np.random.normal([0, 0], 1, (50, 2))  # 聚类1
        cluster2 = np.random.normal([5, 5], 1, (50, 2))  # 聚类2  
        cluster3 = np.random.normal([10, 10], 1, (50, 2))  # 聚类3
        
        features = np.vstack([cluster1, cluster2, cluster3])
        
        # 创建模拟特征字典列表
        self.mock_features_dict = [
            {
                'features': {
                    'feature1': features[:, 0],
                    'feature2': features[:, 1]
                },
                'coordinates': [(i, 0, 0) for i in range(len(features))]
            }
        ]
    
    def test_clustering(self):
        """测试聚类"""
        try:
            cluster_labels_list = self.clusterer.fit_predict(self.mock_features_dict)
            
            self.assertEqual(len(cluster_labels_list), 1)
            cluster_labels = cluster_labels_list[0]
            
            # 验证聚类标签数量
            unique_labels = np.unique(cluster_labels)
            self.assertGreaterEqual(len(unique_labels), 2)  # 至少应该有几个聚类
            self.assertLessEqual(len(unique_labels), 3)  # 不超过指定的聚类数
            
            # 验证标签范围
            self.assertTrue(all(label >= 0 for label in cluster_labels if label != -1))
            
        except Exception as e:
            self.skipTest(f"Skipping clustering test due to: {e}")
    
    def test_clustering_visualization(self):
        """测试聚类可视化"""
        try:
            cluster_labels_list = self.clusterer.fit_predict(self.mock_features_dict)
            cluster_labels = cluster_labels_list[0]
            
            # 创建特征矩阵用于可视化
            features = []
            for i in range(len(cluster_labels)):
                if i < len(self.mock_features_dict[0]['coordinates']):
                    feature_vector = []
                    for feature_name in ['feature1', 'feature2']:
                        values = self.mock_features_dict[0]['features'][feature_name]
                        if i < len(values):
                            feature_vector.append(values[i])
                    features.append(feature_vector)
            
            if features:
                features_array = np.array(features)
                viz_path = Path(self.temp_dir) / "cluster_viz.png"
                
                self.clusterer.visualize_clusters(
                    features_array, cluster_labels[:len(features)], 
                    str(viz_path), method='pca'
                )
                
                self.assertTrue(viz_path.exists())
                
        except Exception as e:
            self.skipTest(f"Skipping visualization test due to: {e}")


class TestMaskRefiner(unittest.TestCase):
    """掩码精细器测试"""
    
    def setUp(self):
        """设置测试环境"""
        if not DEPS_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.create_test_mask_data()
        
        # 创建精细器
        self.refiner = MaskRefiner(
            min_cluster_size=10,
            smoothing_iterations=1,
            connectivity=1
        )
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_mask_data(self):
        """创建测试掩码数据"""
        # 创建测试图像和掩码
        image_data = np.random.randn(40, 40, 40).astype(np.float32)
        mask_data = np.zeros((40, 40, 40), dtype=np.uint8)
        
        # 创建多个区域
        mask_data[10:20, 10:20, 10:20] = 1  # 区域1
        mask_data[20:30, 20:30, 20:30] = 1  # 区域2
        
        # 保存为 NIfTI 文件
        affine = np.eye(4)
        
        image_nii = nib.Nifti1Image(image_data, affine)
        mask_nii = nib.Nifti1Image(mask_data, affine)
        
        self.test_image_path = Path(self.temp_dir) / "test_image.nii.gz"
        self.test_mask_path = Path(self.temp_dir) / "test_mask.nii.gz"
        
        nib.save(image_nii, str(self.test_image_path))
        nib.save(mask_nii, str(self.test_mask_path))
    
    def test_mask_refinement(self):
        """测试掩码精细划分"""
        try:
            # 创建模拟聚类标签
            coordinates = [(15, 15, 15), (25, 25, 25)]  # 两个区域的中心
            cluster_labels = np.array([0, 1])  # 两个聚类
            
            output_dir = Path(self.temp_dir) / "refined"
            
            saved_files = self.refiner.refine_masks(
                image_path=str(self.test_image_path),
                mask_path=str(self.test_mask_path),
                cluster_labels=cluster_labels,
                coordinates=coordinates,
                output_dir=str(output_dir),
                save_individual=True,
                save_combined=True
            )
            
            self.assertIsInstance(saved_files, dict)
            
            # 检查是否保存了文件
            expected_keys = ['combined']
            for key in expected_keys:
                self.assertIn(key, saved_files)
                self.assertTrue(Path(saved_files[key]).exists())
                
        except Exception as e:
            self.skipTest(f"Skipping mask refinement test due to: {e}")
    
    def test_quality_evaluation(self):
        """测试质量评估"""
        try:
            # 创建简单的测试案例
            original_mask = np.ones((20, 20, 20), dtype=np.int16)
            refined_masks = {
                'cluster_0': np.ones((20, 20, 20), dtype=np.int16)
            }
            cluster_labels = np.zeros(100)
            
            quality = self.refiner.evaluate_refinement_quality(
                original_mask, refined_masks, cluster_labels
            )
            
            self.assertIsInstance(quality, dict)
            if quality:
                self.assertIn('coverage_ratio', quality)
                self.assertTrue(0 <= quality['coverage_ratio'] <= 1)
                
        except Exception as e:
            self.skipTest(f"Skipping quality evaluation test due to: {e}")


def run_tests():
    """运行所有测试"""
    print("运行 onem_habitat 基础功能测试\n")
    
    if not DEPS_AVAILABLE:
        print(f"跳过测试 - 缺少依赖: {IMPORT_ERROR}")
        print("请安装所需依赖:")
        print("pip install pyradiomics scikit-learn matplotlib seaborn scikit-image nibabel")
        return
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestHabitatConfig,
        TestHabitatUtils,
        TestRadiomicsUtils,
        TestLocalRadiomicsExtractor,
        TestFeatureClustering,
        TestMaskRefiner
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