"""
TITAN Deep Transfer Learning Feature Extractor

This module provides deep transfer learning feature extraction for pathology images
using the TITAN model with pre-trained backbones.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import time

# Try to import required dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from ..models.titan_model import TITANModel, create_titan_model
    TITAN_AVAILABLE = True
except ImportError:
    TITAN_AVAILABLE = False

try:
    from ..utils.titan_utils import (
        load_titan_model, extract_titan_features, preprocess_for_titan
    )
    TITAN_UTILS_AVAILABLE = True
except ImportError:
    TITAN_UTILS_AVAILABLE = False


class TITANExtractor:
    """
    Extract deep transfer learning features from pathology images using TITAN model.
    
    This class provides a comprehensive interface for extracting deep features
    using various pre-trained backbones with transfer learning capabilities.
    """
    
    def __init__(self, config=None):
        """
        Initialize TITAN extractor.
        
        Args:
            config: Configuration object for feature extraction parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = self._setup_device()
        self.transform = None
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize model and transforms
        self._initialize_model()
        self._initialize_transforms()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available. Install with: pip install Pillow")
        
        if not TORCHVISION_AVAILABLE:
            self.logger.warning("torchvision not available. Install with: pip install torchvision")
        
        if not TITAN_AVAILABLE:
            self.logger.warning("TITAN model not available")
        
        if not TITAN_UTILS_AVAILABLE:
            self.logger.warning("TITAN utils not available")
    
    def _setup_device(self) -> str:
        """Setup computation device."""
        if self.config and hasattr(self.config, 'device'):
            device = self.config.device
        else:
            device = 'auto'
        
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return 'cpu'
        else:
            return device
    
    def _initialize_model(self):
        """Initialize TITAN model."""
        try:
            # Get model parameters from config
            if self.config:
                backbone_name = getattr(self.config, 'backbone_name', 'resnet50')
                pretrained = getattr(self.config, 'pretrained', True)
                feature_dim = getattr(self.config, 'feature_dim', 1024)
                use_attention = getattr(self.config, 'use_attention', True)
                checkpoint_path = getattr(self.config, 'checkpoint_path', None)
            else:
                # Default parameters
                backbone_name = 'resnet50'
                pretrained = True
                feature_dim = 1024
                use_attention = True
                checkpoint_path = None
            
            # Create model
            self.model = create_titan_model(
                backbone_name=backbone_name,
                pretrained=pretrained,
                feature_dim=feature_dim,
                use_attention=use_attention,
                checkpoint_path=checkpoint_path
            )
            
            # Move to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Initialized TITAN model with {backbone_name} backbone")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TITAN model: {e}")
            raise
    
    def _initialize_transforms(self):
        """Initialize image preprocessing transforms."""
        # Default transforms for pathology images
        transform_list = [
            transforms.Resize((224, 224)),  # Standard ImageNet size
            transforms.ToTensor(),
        ]
        
        # Add normalization if specified
        if self.config and getattr(self.config, 'normalize', True):
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def extract_features(self, image_path: str, output_path: str = None) -> Dict:
        """
        Extract deep features from a single pathology image.
        
        Args:
            image_path: Path to pathology image file
            output_path: Path to save extracted features
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            self.logger.info(f"Extracting features from {Path(image_path).name}")
            
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            if image_tensor is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            start_time = time.time()
            
            # Extract features
            with torch.no_grad():
                # Add batch dimension
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                image_tensor = image_tensor.to(self.device)
                
                # Extract features
                features = self.model.extract_features(image_tensor)
                
                # Convert to numpy
                if features.is_cuda:
                    features = features.cpu()
                features = features.numpy()
                
                # Remove batch dimension
                if features.ndim == 2 and features.shape[0] == 1:
                    features = features[0]
            
            extraction_time = time.time() - start_time
            
            # Prepare result
            result = {
                'image_path': image_path,
                'features': features,
                'feature_names': [f'titan_feature_{i}' for i in range(len(features))],
                'model_info': self.model.get_model_info(),
                'extraction_metadata': {
                    'extraction_time': extraction_time,
                    'device': self.device,
                    'input_shape': tuple(image_tensor.shape),
                    'feature_dim': len(features)
                }
            }
            
            # Save features if output path provided
            if output_path:
                self._save_features(result, output_path)
            
            self.logger.info(f"Successfully extracted {len(features)} features "
                           f"in {extraction_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            raise
    
    def extract_batch_features(self, image_dir: str, output_dir: str = None,
                             file_pattern: str = "*.jpg *.png *.tif *.tiff *.bmp",
                             batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame:
        """
        Extract features from multiple images in a directory.
        
        Args:
            image_dir: Directory containing pathology images
            output_dir: Directory to save feature results
            file_pattern: File pattern to match images
            batch_size: Batch size for processing
            num_workers: Number of worker processes
            
        Returns:
            DataFrame containing features for all images
        """
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            # Find image files
            image_files = self._find_image_files(image_dir, file_pattern)
            
            if not image_files:
                raise ValueError(f"No images found in {image_dir}")
            
            self.logger.info(f"Found {len(image_files)} images for batch processing")
            
            # Create output structure
            if output_dir:
                output_structure = self._create_output_structure(output_dir)
            else:
                output_structure = None
            
            # Create dataset and dataloader
            dataset = TITANDataset(image_files, self.transform)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Process images in batches
            all_features = []
            all_paths = []
            all_metadata = []
            
            total_time = 0
            
            for batch_idx, (batch_paths, batch_images) in enumerate(dataloader):
                try:
                    batch_start_time = time.time()
                    
                    batch_images = batch_images.to(self.device)
                    
                    with torch.no_grad():
                        batch_features = self.model.extract_features(batch_images)
                        
                        # Convert to numpy
                        if batch_features.is_cuda:
                            batch_features = batch_features.cpu()
                        batch_features = batch_features.numpy()
                    
                    batch_time = time.time() - batch_start_time
                    total_time += batch_time
                    
                    # Store results
                    for i, path in enumerate(batch_paths):
                        all_paths.append(path)
                        all_features.append(batch_features[i])
                        all_metadata.append({
                            'batch_idx': batch_idx,
                            'extraction_time': batch_time / len(batch_paths)
                        })
                    
                    if (batch_idx + 1) % 10 == 0:
                        self.logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process batch {batch_idx}: {e}")
            
            # Prepare results
            all_results = []
            for path, features, metadata in zip(all_paths, all_features, all_metadata):
                result = {
                    'image_path': path,
                    'features': features,
                    'feature_names': [f'titan_feature_{i}' for i in range(len(features))],
                    'model_info': self.model.get_model_info(),
                    'extraction_metadata': {
                        **metadata,
                        'device': self.device,
                        'feature_dim': len(features)
                    }
                }
                all_results.append(result)
            
            # Convert to DataFrame
            df = self._results_to_dataframe(all_results)
            
            # Save batch results
            if output_structure:
                self._save_batch_results(df, output_structure)
            
            avg_time_per_image = total_time / len(image_files)
            self.logger.info(f"Batch processing completed: {len(image_files)} images "
                           f"in {total_time:.2f}s (avg: {avg_time_per_image:.3f}s per image)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Batch feature extraction failed: {e}")
            raise
    
    def extract_layer_features(self, image_path: str, layers: List[str] = None) -> Dict:
        """
        Extract features from specific layers of the model.
        
        Args:
            image_path: Path to pathology image
            layers: List of layer names to extract features from
            
        Returns:
            Dictionary containing features from specified layers
        """
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            if image_tensor is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Hook for capturing intermediate features
            layer_features = {}
            hooks = []
            
            def create_hook(layer_name):
                def hook_fn(module, input, output):
                    # Convert to numpy and remove batch dimension
                    features = output.detach().cpu().numpy()
                    if features.ndim == 4 and features.shape[0] == 1:
                        features = features[0]  # Remove batch dimension
                    layer_features[layer_name] = features
                return hook_fn
            
            # Register hooks for specified layers
            if layers:
                for name, layer in self._find_model_layers(layers):
                    hook = layer.register_forward_hook(create_hook(name))
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(image_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Prepare result
            result = {
                'image_path': image_path,
                'layer_features': layer_features,
                'layers_requested': layers or list(layer_features.keys()),
                'model_info': self.model.get_model_info()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Layer feature extraction failed for {image_path}: {e}")
            raise
    
    def benchmark_model(self, input_shape: Tuple = (3, 224, 224),
                       num_iterations: int = 100) -> Dict:
        """
        Benchmark model performance.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            num_iterations: Number of iterations for timing
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            # Create random input
            batch_size = 1
            input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
            
            self.logger.info(f"Benchmarking model with input shape {input_shape}")
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model.extract_features(input_tensor)
            
            # Benchmark
            self.model.eval()
            with torch.no_grad():
                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                for _ in range(num_iterations):
                    features = self.model.extract_features(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = 1.0 / avg_time
            
            # Calculate memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            else:
                memory_used = 0
                memory_cached = 0
            
            benchmark_result = {
                'input_shape': input_shape,
                'batch_size': batch_size,
                'feature_dim': features.shape[-1],
                'total_time': total_time,
                'avg_time_per_inference': avg_time,
                'fps': fps,
                'memory_used_gb': memory_used,
                'memory_cached_gb': memory_cached,
                'device': self.device,
                'model_info': self.model.get_model_info()
            }
            
            self.logger.info(f"Benchmark completed: {avg_time:.4f}s per inference, "
                           f"{fps:.1f} FPS")
            
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess image."""
        try:
            if PIL_AVAILABLE:
                # Use PIL for image loading
                image = Image.open(image_path)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply transforms
                if self.transform:
                    return self.transform(image)
                else:
                    return transforms.ToTensor()(image)
            else:
                # Fallback method
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _find_image_files(self, directory: str, pattern: str) -> List[str]:
        """Find image files in directory."""
        from pathlib import Path
        
        dir_path = Path(directory)
        
        # Supported extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(dir_path.glob(ext))
        
        # Convert to strings and sort
        image_files = sorted([str(f) for f in image_files])
        
        return image_files
    
    def _find_model_layers(self, requested_layers: List[str]) -> List[Tuple[str, nn.Module]]:
        """Find specific layers in the model."""
        found_layers = []
        
        for name, module in self.model.named_modules():
            if any(req in name for req in requested_layers):
                found_layers.append((name, module))
        
        return found_layers
    
    def _save_features(self, result: Dict, output_path: str):
        """Save extracted features to file."""
        try:
            # Prepare feature data
            feature_data = {
                'image_path': result['image_path'],
                'features': result['features'].tolist(),
                'feature_names': result['feature_names'],
                'model_info': result['model_info'],
                'extraction_metadata': result['extraction_metadata']
            }
            
            # Save as JSON
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(feature_data, f, indent=2, default=str)
            else:
                # Save as CSV
                # Create DataFrame with feature columns
                df_data = {
                    'image_path': [result['image_path']]
                }
                
                for i, feature_value in enumerate(result['features']):
                    df_data[f'titan_feature_{i}'] = [feature_value]
                
                # Add metadata columns
                for key, value in result['extraction_metadata'].items():
                    df_data[f'meta_{key}'] = [value]
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to save features to {output_path}: {e}")
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert extraction results to DataFrame."""
        rows = []
        
        for result in results:
            if 'error' in result:
                continue
            
            # Flatten features into columns
            row = {
                'image_path': result['image_path'],
                'extraction_success': True
            }
            
            # Add feature columns
            for i, feature_value in enumerate(result['features']):
                row[f'titan_feature_{i}'] = feature_value
            
            # Add metadata columns
            for key, value in result['extraction_metadata'].items():
                row[f'meta_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_output_structure(self, base_dir: str) -> Dict:
        """Create output directory structure."""
        base_path = Path(base_dir)
        
        structure = {
            'base': str(base_path),
            'features': str(base_path / 'features'),
            'layer_features': str(base_path / 'layer_features'),
            'reports': str(base_path / 'reports'),
            'logs': str(base_path / 'logs')
        }
        
        for path in structure.values():
            os.makedirs(path, exist_ok=True)
        
        return structure
    
    def _save_batch_results(self, df: pd.DataFrame, structure: Dict):
        """Save batch results."""
        try:
            # Save combined CSV
            csv_path = Path(structure['features']) / 'batch_titan_features.csv'
            df.to_csv(csv_path, index=False)
            
            # Save summary report
            report = {
                'total_images': len(df),
                'successful_extractions': len(df[df['extraction_success'] == True]),
                'feature_dimension': len([col for col in df.columns 
                                       if col.startswith('titan_feature_')]),
                'model_info': self.model.get_model_info(),
                'device': self.device,
                'feature_columns': [col for col in df.columns if col.startswith('titan_feature_')]
            }
            
            report_path = Path(structure['reports']) / 'batch_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save batch results: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model:
            return self.model.get_model_info()
        else:
            return {'error': 'Model not initialized'}


class TITANDataset(torch.utils.data.Dataset):
    """Dataset class for TITAN feature extraction."""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        if PIL_AVAILABLE:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            # Fallback
            image = None
        
        # Apply transforms
        if self.transform and image is not None:
            image = self.transform(image)
        
        return image_path, image