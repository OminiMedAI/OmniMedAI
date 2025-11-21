"""
onem_habitat 使用示例
"""

import os
import numpy as np
from pathlib import Path

# 导入主要模块
from onem_habitat.radiomics import LocalRadiomicsExtractor
from onem_habitat.clustering import FeatureClustering
from onem_habitat.segmentation import MaskRefiner
from onem_habitat.config import HabitatConfig, HabitatConfigManager, PRESET_CONFIGS


def example_1_local_features_extraction():
    """示例 1: 局部放射组学特征提取"""
    print("=== 示例 1: 局部放射组学特征提取 ===")
    
    # 创建特征提取器
    extractor = LocalRadiomicsExtractor(
        kernel_size=(5, 5, 5),
        feature_types=['firstorder', 'glcm', 'glrlm'],
        bin_width=25,
        n_jobs=1
    )
    
    # 提取单个图像的特征
    image_path = "data/images/patient001.nii.gz"
    mask_path = "data/masks/patient001_mask.nii.gz"
    output_path = "output/features/patient001_features.npy"
    
    if os.path.exists(image_path) and os.path.exists(mask_path):
        try:
            result = extractor.extract_local_features(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                step_size=2  # 每2个体素提取一次，加快速度
            )
            
            print(f"特征提取完成:")
            print(f"  体素数量: {result['metadata']['n_voxels']}")
            print(f"  特征数量: {len(result['metadata']['feature_types'])}")
            print(f"  特征类型: {result['metadata']['feature_types']}")
            print(f"  保存到: {output_path}")
            
            # 获取特征摘要
            summary = extractor.get_feature_summary(result)
            print(f"  特征摘要:")
            for feature_name, stats in list(summary.items())[:3]:  # 显示前3个特征
                print(f"    {feature_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                
        except Exception as e:
            print(f"特征提取失败: {e}")
    else:
        print(f"文件不存在: {image_path} 或 {mask_path}")


def example_2_batch_feature_extraction():
    """示例 2: 批量特征提取"""
    print("\n=== 示例 2: 批量特征提取 ===")
    
    # 创建特征提取器
    extractor = LocalRadiomicsExtractor(
        kernel_size=(5, 5, 5),
        feature_types=['firstorder', 'glcm'],
        bin_width=32,
        n_jobs=2
    )
    
    # 批量提取
    images_dir = "data/images"
    masks_dir = "data/masks"
    output_dir = "output/batch_features"
    
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        try:
            results = extractor.batch_extract_features(
                images_dir=images_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                file_pattern="*.nii.gz",
                step_size=3
            )
            
            print(f"批量特征提取完成:")
            print(f"  处理结果数量: {len(results)}")
            
            successful_count = sum(1 for r in results if 'error' not in r)
            print(f"  成功处理: {successful_count}")
            print(f"  失败处理: {len(results) - successful_count}")
            
            for result in results[:3]:  # 显示前3个结果
                if 'error' in result:
                    print(f"  失败: {result['image_file']} - {result['error']}")
                else:
                    print(f"  成功: {result['image_file']} - {result['n_voxels']} voxels")
                    
        except Exception as e:
            print(f"批量特征提取失败: {e}")
    else:
        print(f"目录不存在: {images_dir} 或 {masks_dir}")


def example_3_feature_clustering():
    """示例 3: 特征聚类分析"""
    print("\n=== 示例 3: 特征聚类分析 ===")
    
    # 加载特征数据
    features_files = [
        "output/features/patient001_features.npy",
        "output/features/patient002_features.npy"
    ]
    
    features_dict_list = []
    
    for features_file in features_files:
        if os.path.exists(features_file):
            try:
                features_dict = np.load(features_file, allow_pickle=True).item()
                features_dict_list.append(features_dict)
                print(f"加载特征: {features_file}")
            except Exception as e:
                print(f"加载特征失败 {features_file}: {e}")
    
    if features_dict_list:
        # 创建聚类器
        clusterer = FeatureClustering(
            clustering_method='kmeans',
            n_clusters=4,
            feature_selection='variance',
            pca_components=10,
            standardize=True
        )
        
        try:
            # 执行聚类
            cluster_labels_list = clusterer.fit_predict(features_dict_list)
            
            print(f"聚类分析完成:")
            print(f"  处理图像数量: {len(cluster_labels_list)}")
            
            for i, cluster_labels in enumerate(cluster_labels_list):
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                print(f"  图像 {i+1}: {dict(zip(unique_labels, counts))} 个聚类")
            
            # 可视化聚类结果
            if len(cluster_labels_list) > 0 and len(cluster_labels_list[0]) > 0:
                # 合并所有特征进行可视化
                all_features, feature_info = clusterer._prepare_features(features_dict_list)
                if all_features is not None:
                    # 重新执行聚类以获取标签
                    full_labels = clusterer._perform_clustering(
                        clusterer._preprocess_features(
                            clusterer._select_features(all_features, feature_info)
                        )
                    )
                    
                    # 保存可视化
                    viz_path = "output/clustering_visualization.png"
                    clusterer.visualize_clusters(
                        all_features, full_labels, viz_path, method='tsne'
                    )
                    print(f"  聚类可视化保存到: {viz_path}")
            
        except Exception as e:
            print(f"聚类分析失败: {e}")
    else:
        print("没有找到有效的特征文件")


def example_4_mask_refinement():
    """示例 4: 基于 Mask 重新划分"""
    print("\n=== 示例 4: 基于 Mask 重新划分 ===")
    
    # 创建掩码精细器
    refiner = MaskRefiner(
        min_cluster_size=50,
        smoothing_iterations=2,
        connectivity=1,
        fill_holes=True
    )
    
    # 加载必要文件
    image_path = "data/images/patient001.nii.gz"
    mask_path = "data/masks/patient001_mask.nii.gz"
    features_path = "output/features/patient001_features.npy"
    labels_path = "output/clustering/clustering_labels_patient001.npy"
    
    if all(os.path.exists(p) for p in [image_path, mask_path, features_path, labels_path]):
        try:
            # 加载聚类标签
            cluster_labels = np.load(labels_path)
            
            # 加载特征数据获取坐标
            features_dict = np.load(features_path, allow_pickle=True).item()
            coordinates = features_dict.get('coordinates', [])
            
            # 确保标签数量匹配
            if len(cluster_labels) != len(coordinates):
                print(f"警告: 标签数量({len(cluster_labels)})与坐标数量({len(coordinates)})不匹配")
                min_len = min(len(cluster_labels), len(coordinates))
                cluster_labels = cluster_labels[:min_len]
                coordinates = coordinates[:min_len]
            
            output_dir = "output/refined_masks"
            
            # 重新划分掩码
            saved_files = refiner.refine_masks(
                image_path=image_path,
                mask_path=mask_path,
                cluster_labels=cluster_labels,
                coordinates=coordinates,
                output_dir=output_dir,
                save_individual=True,
                save_combined=True
            )
            
            print(f"掩码重新划分完成:")
            print(f"  输出目录: {output_dir}")
            print(f"  保存文件:")
            for name, path in saved_files.items():
                print(f"    {name}: {path}")
            
            # 评估质量
            if 'combined' in saved_files:
                import nibabel as nib
                original_mask_data = nib.load(mask_path).get_fdata()
                refined_masks = {}
                
                # 重新加载精细化掩码进行评估
                for file_name, file_path in saved_files.items():
                    if file_name.startswith('cluster_'):
                        mask_data = nib.load(file_path).get_fdata()
                        refined_masks[file_name] = mask_data
                
                quality_metrics = refiner.evaluate_refinement_quality(
                    original_mask_data, refined_masks, cluster_labels
                )
                
                print(f"  质量评估:")
                for metric_name, value in quality_metrics.items():
                    print(f"    {metric_name}: {value:.3f}")
                    
        except Exception as e:
            print(f"掩码重新划分失败: {e}")
    else:
        print("缺少必要文件进行掩码重新划分")


def example_5_complete_habitat_workflow():
    """示例 5: 完整生态分析工作流程"""
    print("\n=== 示例 5: 完整生态分析工作流程 ===")
    
    # 使用预设配置
    config = PRESET_CONFIGS['ct_lung']
    
    # 创建配置管理器
    config_manager = HabitatConfigManager("config")
    config_manager.save_config(config, name="example_workflow")
    
    print("使用预设配置: CT Lung Habitat Analysis")
    print(f"  核大小: {config.kernel_size}")
    print(f"  聚类数量: {config.n_clusters}")
    print(f"  最小聚类大小: {config.min_cluster_size}")
    
    # 工作流程目录
    base_dir = "data"
    output_dir = "output/complete_workflow"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤 1: 特征提取
    print("\n步骤 1: 特征提取...")
    extractor = LocalRadiomicsExtractor(
        kernel_size=config.kernel_size,
        feature_types=config.feature_types,
        bin_width=config.bin_width,
        n_jobs=config.extraction_n_jobs,
        step_size=config.step_size
    )
    
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    features_dir = os.path.join(output_dir, "features")
    
    extraction_results = []
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        extraction_results = extractor.batch_extract_features(
            images_dir=images_dir,
            masks_dir=masks_dir,
            output_dir=features_dir,
            step_size=config.step_size
        )
        print(f"  成功提取 {len([r for r in extraction_results if 'error' not in r])} 个图像的特征")
    
    # 步骤 2: 特征聚类
    print("\n步骤 2: 特征聚类...")
    
    # 加载所有特征
    features_dict_list = []
    for result in extraction_results:
        if 'error' not in result:
            try:
                features_dict = np.load(result['output_file'], allow_pickle=True).item()
                features_dict_list.append(features_dict)
            except Exception as e:
                print(f"  跳过特征文件 {result['output_file']}: {e}")
    
    if features_dict_list:
        clusterer = FeatureClustering(
            clustering_method=config.clustering_method,
            n_clusters=config.n_clusters,
            feature_selection=config.feature_selection,
            pca_components=config.pca_components,
            standardize=config.standardize
        )
        
        cluster_labels_list = clusterer.fit_predict(features_dict_list)
        
        # 保存聚类结果
        clustering_dir = os.path.join(output_dir, "clustering")
        clustering_files = clusterer.save_clustering_results(
            features_dict_list, cluster_labels_list, clustering_dir
        )
        
        print(f"  聚类分析完成，结果保存到 {clustering_dir}")
        
        # 步骤 3: 掩码精细划分
        print("\n步骤 3: 掩码精细划分...")
        refiner = MaskRefiner(
            min_cluster_size=config.min_cluster_size,
            smoothing_iterations=config.smoothing_iterations,
            connectivity=config.connectivity,
            fill_holes=config.fill_holes
        )
        
        refinement_results = refiner.batch_refine_masks(
            images_dir=images_dir,
            masks_dir=masks_dir,
            features_dir=features_dir,
            clustering_results_dir=clustering_dir,
            output_dir=os.path.join(output_dir, "refined_masks")
        )
        
        successful_refinements = [r for r in refinement_results if 'error' not in r]
        print(f"  成功精细化 {len(successful_refinements)} 个掩码")
        
        # 步骤 4: 生成报告
        print("\n步骤 4: 生成分析报告...")
        report = {
            'configuration': config.to_dict(),
            'extraction_summary': {
                'total_images': len(extraction_results),
                'successful_extractions': len(extraction_results) - sum(1 for r in extraction_results if 'error' in r),
                'extraction_results': extraction_results
            },
            'clustering_summary': {
                'total_images_processed': len(cluster_labels_list),
                'average_clusters_per_image': np.mean([len(np.unique(labels)) for labels in cluster_labels_list]) if cluster_labels_list else 0
            },
            'refinement_summary': {
                'total_masks_processed': len(refinement_results),
                'successful_refinements': len(successful_refinements),
                'refinement_results': refinement_results
            }
        }
        
        from onem_habitat.utils import habitat_utils
        report_path = os.path.join(output_dir, "habitat_analysis_report.json")
        habitat_utils.save_json(report, report_path)
        
        print(f"  分析报告保存到: {report_path}")
        
        print("\n✅ 完整工作流程执行完成!")
        print(f"  结果保存在: {output_dir}")
        print(f"  配置保存在: config/habitat_config.json")
        
    else:
        print("  没有有效的特征数据进行聚类")


def example_6_config_management():
    """示例 6: 配置管理"""
    print("\n=== 示例 6: 配置管理 ===")
    
    # 创建配置管理器
    manager = HabitatConfigManager("config")
    
    # 获取默认配置
    default_config = manager.get_default_config()
    print(f"默认配置:")
    print(f"  核大小: {default_config.kernel_size}")
    print(f"  聚类方法: {default_config.clustering_method}")
    print(f"  聚类数量: {default_config.n_clusters}")
    
    # 创建自定义配置
    custom_config = HabitatConfig(
        kernel_size=(7, 7, 7),
        clustering_method='hierarchical',
        n_clusters=5,
        feature_types=['firstorder', 'glcm', 'shape'],
        min_cluster_size=100,
        extraction_n_jobs=4
    )
    
    # 验证配置
    validation = custom_config.validate()
    if validation['valid']:
        print("✅ 配置验证通过")
        # 保存自定义配置
        saved_path = manager.save_config(custom_config, name="custom_example")
        print(f"  配置保存到: {saved_path}")
    else:
        print("❌ 配置验证失败:")
        for error in validation['errors']:
            print(f"  错误: {error}")
    
    # 列出所有预设
    presets = manager.list_presets()
    print(f"\n可用预设配置: {presets}")
    
    # 加载预设配置
    if presets:
        preset_name = presets[0]
        preset_config = manager.load_config(preset_name)
        print(f"加载预设 '{preset_name}':")
        print(f"  聚类数量: {preset_config.n_clusters}")
        print(f"  特征类型: {preset_config.feature_types}")


def main():
    """运行所有示例"""
    print("onem_habitat 使用示例\n")
    
    # 创建必要的输出目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/features", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # 运行示例（注释掉需要实际数据的示例）
    example_1_local_features_extraction()
    # example_2_batch_feature_extraction()
    # example_3_feature_clustering()
    # example_4_mask_refinement()
    # example_5_complete_habitat_workflow()
    example_6_config_management()
    
    print("\n示例运行完成!")
    print("\n📝 注意:")
    print("- 需要安装依赖: pip install pyradiomics scikit-learn matplotlib seaborn scikit-image nibabel")
    print("- 准备数据目录: data/images/ 和 data/masks/")
    print("- 取消注释以运行需要实际数据的示例")


if __name__ == "__main__":
    main()