"""
onem_process 使用示例
"""

import os
from pathlib import Path

# 导入主要模块
from onem_process.converters import DicomToNiftiConverter, BatchConverter
from onem_process.processors import ROIProcessor, ImageProcessor
from onem_process.config import ProcessingConfig, ConversionConfig, ConfigManager


def example_1_single_dicom_conversion():
    """示例 1: 单个 DICOM 文件转换"""
    print("=== 示例 1: 单个 DICOM 文件转换 ===")
    
    # 创建转换器
    converter = DicomToNiftiConverter(output_dir="output/nifti")
    
    # 转换单个 DICOM 文件
    dicom_file = "data/single_dicom.dcm"
    if os.path.exists(dicom_file):
        try:
            nifti_file = converter.convert_single_dicom(dicom_file)
            print(f"转换成功: {nifti_file}")
        except Exception as e:
            print(f"转换失败: {e}")
    else:
        print(f"文件不存在: {dicom_file}")


def example_2_dicom_series_conversion():
    """示例 2: DICOM 序列转换"""
    print("\n=== 示例 2: DICOM 序列转换 ===")
    
    # 创建转换器
    converter = DicomToNiftiConverter(output_dir="output/nifti")
    
    # 转换 DICOM 序列
    dicom_dir = "data/dicom_series"
    if os.path.exists(dicom_dir):
        try:
            nifti_file = converter.convert_dicom_series(dicom_dir)
            print(f"序列转换成功: {nifti_file}")
        except Exception as e:
            print(f"序列转换失败: {e}")
    else:
        print(f"目录不存在: {dicom_dir}")


def example_3_batch_dataset_conversion():
    """示例 3: 批量数据集转换"""
    print("\n=== 示例 3: 批量数据集转换 ===")
    
    # 创建批量转换器
    base_dir = "data/dataset"
    if os.path.exists(base_dir):
        converter = BatchConverter(base_dir, output_base_dir="output/converted")
        
        try:
            # 转换整个数据集
            converted_images, converted_masks = converter.convert_dataset(
                images_dir="images",
                masks_dir="masks",
                skip_existing=True
            )
            
            # 获取转换摘要
            summary = converter.get_conversion_summary(converted_images, converted_masks)
            print(f"转换完成:")
            print(f"  图像文件: {summary['total_images']}")
            print(f"  掩码文件: {summary['total_masks']}")
            print(f"  成功转换: {summary['successful_conversions']}")
            
        except Exception as e:
            print(f"批量转换失败: {e}")
    else:
        print(f"数据集目录不存在: {base_dir}")


def example_4_roi_extraction():
    """示例 4: ROI 提取"""
    print("\n=== 示例 4: ROI 提取 ===")
    
    # 创建 ROI 处理器
    roi_processor = ROIProcessor(padding=(15, 15, 15))
    
    # 提取单个 ROI
    image_file = "output/converted/images_nifti/patient001.nii.gz"
    mask_file = "output/converted/masks_nifti/patient001_mask.nii.gz"
    output_dir = "output/rois"
    
    if os.path.exists(image_file) and os.path.exists(mask_file):
        try:
            roi_image, roi_mask = roi_processor.extract_roi_from_mask(
                image_file, mask_file, output_dir, roi_name="patient001_roi"
            )
            
            print(f"ROI 提取成功:")
            print(f"  ROI 图像: {roi_image}")
            print(f"  ROI 掩码: {roi_mask}")
            
            # 获取 ROI 统计信息
            stats = roi_processor.get_roi_statistics(roi_mask)
            print(f"  体积: {stats.get('actual_volume_mm3', 0):.2f} mm³")
            print(f"  质心: {stats.get('centroid', (0, 0, 0))}")
            
        except Exception as e:
            print(f"ROI 提取失败: {e}")
    else:
        print(f"输入文件不存在")


def example_5_batch_roi_extraction():
    """示例 5: 批量 ROI 提取"""
    print("\n=== 示例 5: 批量 ROI 提取 ===")
    
    # 创建 ROI 处理器
    roi_processor = ROIProcessor(padding=(10, 10, 10))
    
    # 批量提取 ROI
    image_dir = "output/converted/images_nifti"
    mask_dir = "output/converted/masks_nifti"
    output_dir = "output/batch_rois"
    
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        try:
            results = roi_processor.batch_extract_rois(
                image_dir, mask_dir, output_dir, file_pattern="*.nii.gz"
            )
            
            print(f"批量 ROI 提取完成，处理了 {len(results)} 个文件")
            
            # 显示前几个结果
            for i, result in enumerate(results[:3]):
                print(f"  文件 {i+1}:")
                print(f"    输入图像: {result['image_file']}")
                print(f"    ROI 图像: {result['roi_image']}")
                
        except Exception as e:
            print(f"批量 ROI 提取失败: {e}")
    else:
        print("输入目录不存在")


def example_6_image_processing():
    """示例 6: 图像处理"""
    print("\n=== 示例 6: 图像处理 ===")
    
    # 创建图像处理器
    processor = ImageProcessor()
    
    # 处理图像
    input_file = "output/converted/images_nifti/patient001.nii.gz"
    
    if os.path.exists(input_file):
        try:
            # 1. 重采样
            print("1. 重采样到 1mm^3...")
            resampled_file = "output/processed/resampled.nii.gz"
            processor.resample_image(
                input_file, 
                target_spacing=(1.0, 1.0, 1.0),
                output_path=resampled_file
            )
            
            # 2. 归一化
            print("2. Z-score 归一化...")
            normalized_file = "output/processed/normalized.nii.gz"
            processor.normalize_image(
                resampled_file, 
                method='z_score',
                output_path=normalized_file
            )
            
            # 3. 中心裁剪
            print("3. 中心裁剪...")
            cropped_file = "output/processed/cropped.nii.gz"
            processor.crop_image(
                normalized_file,
                crop_size=(128, 128, 128),
                output_path=cropped_file
            )
            
            print("图像处理完成")
            
        except Exception as e:
            print(f"图像处理失败: {e}")
    else:
        print(f"输入文件不存在: {input_file}")


def example_7_configuration_management():
    """示例 7: 配置管理"""
    print("\n=== 示例 7: 配置管理 ===")
    
    # 创建配置管理器
    config_manager = ConfigManager("config")
    
    # 获取处理配置
    processing_config = config_manager.get_processing_config()
    print(f"当前归一化方法: {processing_config.normalize_method}")
    print(f"当前重采样间距: {processing_config.resample_spacing}")
    
    # 更新配置
    config_manager.update_processing_config(
        normalize_method='percentile',
        resample_spacing=(0.5, 0.5, 0.5),
        roi_padding=(20, 20, 20)
    )
    
    print("配置已更新")
    
    # 获取转换配置
    conversion_config = config_manager.get_conversion_config()
    print(f"跳过已存在文件: {conversion_config.batch_skip_existing}")
    
    # 导出配置
    config_manager.export_configs("output/configs")
    print("配置已导出到 output/configs")


def example_8_complete_workflow():
    """示例 8: 完整工作流程"""
    print("\n=== 示例 8: 完整工作流程 ===")
    
    base_dir = "data/example_dataset"
    output_base_dir = "output/complete_workflow"
    
    if os.path.exists(base_dir):
        try:
            # 1. 配置管理
            config_manager = ConfigManager("config")
            
            # 2. 批量转换
            print("步骤 1: 批量格式转换...")
            batch_converter = BatchConverter(base_dir, output_base_dir)
            converted_images, converted_masks = batch_converter.convert_dataset()
            
            # 3. ROI 提取
            print("步骤 2: ROI 提取...")
            roi_processor = ROIProcessor(
                padding=config_manager.get_processing_config().roi_padding
            )
            
            roi_dir = Path(output_base_dir) / "rois"
            roi_results = roi_processor.batch_extract_rois(
                str(Path(output_base_dir) / "images_nifti"),
                str(Path(output_base_dir) / "masks_nifti"),
                str(roi_dir)
            )
            
            # 4. 图像处理
            print("步骤 3: 图像处理...")
            processor = ImageProcessor()
            
            processed_dir = Path(output_base_dir) / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            for result in roi_results[:5]:  # 处理前5个
                roi_image = result['roi_image']
                
                # 重采样
                resampled = processor.resample_image(
                    roi_image,
                    target_spacing=(1.0, 1.0, 1.0),
                    output_path=str(processed_dir / f"{Path(roi_image).stem}_resampled.nii.gz")
                )
                
                # 归一化
                processor.normalize_image(
                    resampled,
                    method='z_score',
                    output_path=str(processed_dir / f"{Path(roi_image).stem}_final.nii.gz")
                )
            
            print("完整工作流程执行完成!")
            
        except Exception as e:
            print(f"工作流程执行失败: {e}")
    else:
        print(f"示例数据集不存在: {base_dir}")


def main():
    """运行所有示例"""
    print("onem_process 使用示例\n")
    
    # 创建必要的输出目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # 运行示例（注释掉需要实际数据的示例）
    example_1_single_dicom_conversion()
    # example_2_dicom_series_conversion()
    # example_3_batch_dataset_conversion()
    # example_4_roi_extraction()
    # example_5_batch_roi_extraction()
    # example_6_image_processing()
    example_7_configuration_management()
    # example_8_complete_workflow()
    
    print("\n示例运行完成!")


if __name__ == "__main__":
    main()