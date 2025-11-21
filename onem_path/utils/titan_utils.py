"""
TITAN model utility functions
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import required libraries
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


def load_titan_model(model_path: str = None,
                   backbone_name: str = 'resnet50',
                   feature_dim: int = 1024,
                   use_attention: bool = True,
                   device: str = 'auto') -> TITANModel:
    """
    Load TITAN model for feature extraction.
    
    Args:
        model_path: Path to model checkpoint
        backbone_name: Name of backbone architecture
        feature_dim: Dimension of extracted features
        use_attention: Whether to use attention mechanisms
        device: Computation device
        
    Returns:
        Loaded TITAN model
    """
    try:
        if not TITAN_AVAILABLE:
            raise ImportError("TITAN model not available")
        
        # Create model
        model = create_titan_model(
            backbone_name=backbone_name,
            pretrained=(model_path is None),  # Use pretrained if no checkpoint
            feature_dim=feature_dim,
            use_attention=use_attention,
            checkpoint_path=model_path
        )
        
        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = model.to(device)
        model.eval()
        
        logging.info(f"Loaded TITAN model with {backbone_name} backbone on {device}")
        
        return model
        
    except Exception as e:
        logging.error(f"Failed to load TITAN model: {e}")
        raise


def extract_titan_features(model: TITANModel,
                         image_path: str,
                         device: str = 'auto',
                         preprocessing: Dict = None) -> np.ndarray:
    """
    Extract features using TITAN model.
    
    Args:
        model: Loaded TITAN model
        image_path: Path to input image
        device: Computation device
        preprocessing: Preprocessing parameters
        
    Returns:
        Extracted features as numpy array
    """
    try:
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create transforms
        transform = create_titan_transforms(preprocessing or {})
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_path, transform)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Add batch dimension and move to device
        image_tensor = image.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.extract_features(image_tensor)
            
            # Convert to numpy
            if features.is_cuda:
                features = features.cpu()
            
            features = features.numpy()
            
            # Remove batch dimension if present
            if features.ndim == 3 and features.shape[0] == 1:
                features = features[0]
        
        return features
        
    except Exception as e:
        logging.error(f"Failed to extract TITAN features: {e}")
        raise


def preprocess_for_titan(image_path: str, 
                        preprocessing: Dict = None) -> torch.Tensor:
    """
    Preprocess image for TITAN model input.
    
    Args:
        image_path: Path to input image
        preprocessing: Preprocessing parameters
        
    Returns:
        Preprocessed image tensor
    """
    try:
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        # Create transforms
        transform = create_titan_transforms(preprocessing or {})
        
        # Load and preprocess image
        image_tensor = load_and_preprocess_image(image_path, transform)
        
        if image_tensor is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image_tensor
        
    except Exception as e:
        logging.error(f"Failed to preprocess image for TITAN: {e}")
        raise


def create_titan_transforms(preprocessing: Dict) -> transforms.Compose:
    """
    Create torchvision transforms for TITAN model.
    
    Args:
        preprocessing: Preprocessing parameters
        
    Returns:
        Composed transforms
    """
    # Default ImageNet transforms
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    
    # Customize based on preprocessing parameters
    if preprocessing.get('image_size'):
        image_size = preprocessing['image_size']
        if len(image_size) == 2:
            transform_list[0] = transforms.Resize(image_size)
        elif len(image_size) == 3:
            # For 3D images, handle differently
            transform_list[0] = transforms.Resize(image_size[:2])
    
    if preprocessing.get('normalize', True) is False:
        # Remove normalization
        transform_list = [t for t in transform_list if not isinstance(t, transforms.Normalize)]
    
    if preprocessing.get('custom_mean') and preprocessing.get('custom_std'):
        # Custom normalization
        transform_list = [t for t in transform_list if not isinstance(t, transforms.Normalize)]
        transform_list.append(transforms.Normalize(
            mean=preprocessing['custom_mean'],
            std=preprocessing['custom_std']
        ))
    
    if preprocessing.get('augmentation', False):
        # Add augmentation (for training)
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ]
        transform_list.extend(augment_transforms)
    
    return transforms.Compose(transform_list)


def load_and_preprocess_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Load and preprocess image using transforms.
    
    Args:
        image_path: Path to image file
        transform: Image transforms
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image)
        
        return image_tensor
        
    except Exception as e:
        logging.error(f"Failed to load and preprocess image {image_path}: {e}")
        return None


def extract_layer_features(model: TITANModel,
                      image_path: str,
                      target_layers: List[str] = None,
                      device: str = 'auto') -> Dict[str, np.ndarray]:
    """
    Extract features from specific layers of TITAN model.
    
    Args:
        model: Loaded TITAN model
        image_path: Path to input image
        target_layers: List of layer names to extract
        device: Computation device
        
    Returns:
        Dictionary mapping layer names to features
    """
    try:
        if not target_layers:
            return {}
        
        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_path, create_titan_transforms({}))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_tensor = image.unsqueeze(0).to(device)
        
        # Setup hooks for layer extraction
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
        
        # Find target layers
        target_modules = {}
        for name, module in model.named_modules():
            if any(target in name for target in target_layers):
                target_modules[name] = module
        
        # Register hooks
        for name, module in target_modules.items():
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(image_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_features
        
    except Exception as e:
        logging.error(f"Failed to extract layer features: {e}")
        raise


def benchmark_titan_extraction(model: TITANModel,
                          image_size: Tuple[int, int] = (224, 224),
                          num_iterations: int = 100,
                          device: str = 'auto') -> Dict:
    """
    Benchmark TITAN feature extraction performance.
    
    Args:
        model: Loaded TITAN model
        image_size: Input image size
        num_iterations: Number of iterations for timing
        device: Computation device
        
    Returns:
        Performance metrics dictionary
    """
    try:
        import time
        
        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create random input
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, *image_size).to(device)
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.extract_features(input_tensor)
        
        # Benchmark
        model.eval()
        with torch.no_grad():
            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            for _ in range(num_iterations):
                features = model.extract_features(input_tensor)
                
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
            'input_size': image_size,
            'batch_size': batch_size,
            'feature_dim': features.shape[-1],
            'total_time': total_time,
            'avg_time_per_inference': avg_time,
            'fps': fps,
            'memory_used_gb': memory_used,
            'memory_cached_gb': memory_cached,
            'device': device,
            'num_iterations': num_iterations
        }
        
        logging.info(f"TITAN benchmark completed: {avg_time:.4f}s per inference, "
                       f"{fps:.1f} FPS")
        
        return benchmark_result
        
    except Exception as e:
        logging.error(f"TITAN benchmark failed: {e}")
        raise


def get_available_backbones() -> List[str]:
    """Get list of available TITAN backbones."""
    backbones = []
    
    # ResNet variants
    backbones.extend(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    
    # EfficientNet variants
    backbones.extend(['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                    'efficientnet_b6', 'efficientnet_b7'])
    
    # DenseNet variants
    backbones.extend(['densenet121', 'densenet161', 'densenet169', 'densenet201'])
    
    # VGG variants
    backbones.extend(['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    
    # MobileNet variants
    backbones.extend(['mobilenet_v3_small', 'mobilenet_v3_large'])
    
    return backbones


def validate_titan_model(model: TITANModel, input_tensor: torch.Tensor) -> Dict:
    """
    Validate TITAN model with test input.
    
    Args:
        model: Loaded TITAN model
        input_tensor: Test input tensor
        
    Returns:
        Validation results
    """
    try:
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            features = model.extract_features(input_tensor)
            
            # Check for common issues
            validation_results = {
                'model_valid': True,
                'output_shape': features.shape,
                'output_dtype': features.dtype,
                'output_min': float(features.min()),
                'output_max': float(features.max()),
                'output_mean': float(features.mean()),
                'output_std': float(features.std()),
                'has_nan': bool(torch.isnan(features).any()),
                'has_inf': bool(torch.isinf(features).any()),
                'feature_dim': model.get_feature_dim()
            }
        
        # Check for issues
        if validation_results['has_nan']:
            validation_results['model_valid'] = False
            validation_results['issues'] = ['NaN values in output']
        
        if validation_results['has_inf']:
            validation_results['model_valid'] = False
            if 'issues' not in validation_results:
                validation_results['issues'] = []
            validation_results['issues'].append('Infinite values in output')
        
        return validation_results
        
    except Exception as e:
        logging.error(f"TITAN model validation failed: {e}")
        return {
            'model_valid': False,
            'error': str(e)
        }


def save_titan_model(model: TITANModel, save_path: str, metadata: Dict = None):
    """
    Save TITAN model checkpoint.
    
    Args:
        model: TITAN model to save
        save_path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    try:
        # Create save directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info(),
            'metadata': metadata or {}
        }
        
        # Add timestamp
        import datetime
        checkpoint['metadata']['save_time'] = datetime.datetime.now().isoformat()
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        logging.info(f"TITAN model saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Failed to save TITAN model: {e}")
        raise