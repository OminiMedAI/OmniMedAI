"""
Model Manager

This module manages loading and inference for 2D and 3D segmentation models.
It provides a unified interface for different model types and handles
model loading, preprocessing, and inference.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Try to import deep learning frameworks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

try:
    from onem_torch import ModelFactory
    ONEM_TORCH_AVAILABLE = True
except ImportError:
    ONEM_TORCH_AVAILABLE = False
    logging.warning("onem_torch not available. Check module installation")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Install with: pip install nibabel")


class ModelManager:
    """
    Manages loading and inference for 2D and 3D segmentation models.
    
    This class provides a unified interface for different model architectures
    and handles model loading, preprocessing, and inference for both 2D and 3D models.
    """
    
    def __init__(self, model_dir: str = None, device: str = 'auto'):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory containing model files
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        self.model_dir = model_dir
        self.loaded_models = {}
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required. Install with: pip install nibabel")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                return 'cpu'
            return device
        else:
            return device
    
    def load_model(self, model_name: str, model_type: str, 
                   model_path: str = None, config: Dict = None) -> bool:
        """
        Load a segmentation model.
        
        Args:
            model_name: Name identifier for the model
            model_type: Type of model ('2d_unet', '3d_unet', etc.)
            model_path: Path to model weights file
            config: Model configuration parameters
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_name in self.loaded_models:
                self.logger.warning(f"Model {model_name} already loaded")
                return True
            
            # Determine model loading method
            if ONEM_TORCH_AVAILABLE:
                model = self._load_onem_torch_model(model_type, model_path, config)
            else:
                model = self._load_pytorch_model(model_type, model_path, config)
            
            if model is None:
                return False
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            # Store model information
            self.loaded_models[model_name] = {
                'model': model,
                'model_type': model_type,
                'config': config or {},
                'model_path': model_path,
                'input_size': self._get_input_size(model_type, config),
                'preprocessing': self._get_preprocessing_requirements(model_type)
            }
            
            self.logger.info(f"Successfully loaded model {model_name} ({model_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _load_onem_torch_model(self, model_type: str, model_path: str, config: Dict):
        """Load model using onem_torch ModelFactory."""
        try:
            # Map model type to factory parameters
            factory_configs = {
                '2d_unet': {
                    'model_name': 'unet',
                    'num_classes': 1,
                    'in_channels': 1
                },
                '3d_unet': {
                    'model_name': 'unet3d',
                    'num_classes': 1,
                    'in_channels': 1
                },
                '2d_attention_unet': {
                    'model_name': 'attention_unet',
                    'num_classes': 1,
                    'in_channels': 1
                },
                '3d_vnet': {
                    'model_name': 'vnet3d',
                    'num_classes': 1,
                    'in_channels': 1
                }
            }
            
            if model_type not in factory_configs:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None
            
            # Get configuration
            factory_config = factory_configs[model_type]
            if config:
                factory_config.update(config)
            
            # Create model
            model = ModelFactory.create_model(**factory_config)
            
            # Load weights if path provided
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded weights from {model_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model with onem_torch: {e}")
            return None
    
    def _load_pytorch_model(self, model_type: str, model_path: str, config: Dict):
        """Load model using direct PyTorch implementation."""
        # This would contain custom PyTorch model implementations
        # For now, return None as a placeholder
        self.logger.warning("Direct PyTorch model loading not implemented")
        return None
    
    def _get_input_size(self, model_type: str, config: Dict) -> tuple:
        """Get expected input size for model type."""
        input_sizes = {
            '2d_unet': (256, 256),
            '3d_unet': (64, 64, 64),
            '2d_attention_unet': (256, 256),
            '3d_vnet': (128, 128, 128)
        }
        
        # Check config for custom input size
        if config and 'input_size' in config:
            return config['input_size']
        
        return input_sizes.get(model_type, (256, 256))
    
    def _get_preprocessing_requirements(self, model_type: str) -> Dict:
        """Get preprocessing requirements for model type."""
        requirements = {
            'normalization': 'z_score',
            'resize': True,
            'padding': True
        }
        
        # Adjust based on model type
        if '3d' in model_type:
            requirements['slice_dimension'] = 3
        else:
            requirements['slice_dimension'] = 2
        
        return requirements
    
    def predict(self, model_name: str, image_data: np.ndarray, 
                return_probabilities: bool = False) -> np.ndarray:
        """
        Perform inference with loaded model.
        
        Args:
            model_name: Name of loaded model
            image_data: Input image data (numpy array)
            return_probabilities: Whether to return probabilities or binary segmentation
            
        Returns:
            Segmentation result as numpy array
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded. Call load_model first.")
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        model_type = model_info['model_type']
        preprocessing = model_info['preprocessing']
        
        try:
            # Preprocess input
            processed_data = self._preprocess_input(image_data, preprocessing)
            
            # Convert to tensor
            if '3d' in model_type:
                # 3D model: add batch and channel dimensions
                tensor = torch.from_numpy(processed_data).float()
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B, C, D, H, W]
            else:
                # 2D model: process slice by slice
                return self._predict_2d_slices(model_info, image_data, return_probabilities)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = model(tensor)
                
                if return_probabilities:
                    result = torch.sigmoid(output).cpu().numpy()
                else:
                    result = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
            
            # Remove batch and channel dimensions
            result = result[0, 0]
            
            # Postprocess result
            result = self._postprocess_output(result, image_data.shape)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed for model {model_name}: {e}")
            raise
    
    def _predict_2d_slices(self, model_info: Dict, image_data: np.ndarray, 
                          return_probabilities: bool) -> np.ndarray:
        """Predict 2D model on 3D data slice by slice."""
        model = model_info['model']
        preprocessing = model_info['preprocessing']
        input_size = model_info['input_size']
        
        if len(image_data.shape) == 2:
            # Single 2D image
            processed_data = self._preprocess_input(image_data, preprocessing)
            tensor = torch.from_numpy(processed_data).float()
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
            tensor = tensor.to(self.device)
            
            with torch.no_grad():
                output = model(tensor)
                if return_probabilities:
                    result = torch.sigmoid(output).cpu().numpy()
                else:
                    result = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
            
            return result[0, 0]
        
        else:
            # 3D image - process slice by slice
            n_slices = image_data.shape[2]
            result_3d = np.zeros(image_data.shape, dtype=np.float32)
            
            for i in range(n_slices):
                slice_data = image_data[:, :, i]
                
                # Preprocess slice
                processed_slice = self._preprocess_input(slice_data, preprocessing)
                
                # Resize to input size if needed
                if processed_slice.shape != input_size:
                    processed_slice = self._resize_slice(processed_slice, input_size)
                
                # Convert to tensor and predict
                tensor = torch.from_numpy(processed_slice).float()
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
                tensor = tensor.to(self.device)
                
                with torch.no_grad():
                    output = model(tensor)
                    
                    if return_probabilities:
                        slice_result = torch.sigmoid(output).cpu().numpy()
                    else:
                        slice_result = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
                
                # Resize back to original slice size
                slice_result = slice_result[0, 0]
                if slice_result.shape != slice_data.shape:
                    slice_result = self._resize_slice(slice_result, slice_data.shape)
                
                result_3d[:, :, i] = slice_result
            
            return result_3d
    
    def _preprocess_input(self, data: np.ndarray, requirements: Dict) -> np.ndarray:
        """Preprocess input data according to model requirements."""
        processed = data.copy()
        
        # Normalize
        if requirements.get('normalization') == 'z_score':
            mean = np.mean(processed)
            std = np.std(processed)
            if std > 0:
                processed = (processed - mean) / std
        
        # Resize (for 2D models)
        if requirements.get('resize') and len(processed.shape) == 2:
            input_size = (256, 256)  # Default
            processed = self._resize_slice(processed, input_size)
        
        return processed.astype(np.float32)
    
    def _postprocess_output(self, output: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Postprocess model output."""
        # Resize back to original shape if needed
        if output.shape != original_shape:
            if len(original_shape) == 2:
                output = self._resize_slice(output, original_shape)
            elif len(original_shape) == 3:
                # For 3D, assume output needs to match first two dimensions
                target_2d = original_shape[:2]
                output = self._resize_slice(output, target_2d)
        
        return output
    
    def _resize_slice(self, slice_data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize a 2D slice to target shape."""
        try:
            from skimage.transform import resize
            return resize(slice_data, target_shape, preserve_range=True, anti_aliasing=True)
        except ImportError:
            # Fallback to nearest neighbor using numpy
            if len(slice_data.shape) == 2 and len(target_shape) == 2:
                h_scale = target_shape[0] / slice_data.shape[0]
                w_scale = target_shape[1] / slice_data.shape[1]
                
                # Simple nearest neighbor interpolation
                h_indices = (np.arange(target_shape[0]) / h_scale).astype(int)
                w_indices = (np.arange(target_shape[1]) / w_scale).astype(int)
                
                h_indices = np.clip(h_indices, 0, slice_data.shape[0] - 1)
                w_indices = np.clip(w_indices, 0, slice_data.shape[1] - 1)
                
                resized = slice_data[h_indices][:, w_indices]
                return resized
            else:
                return slice_data
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about loaded model."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        info = self.loaded_models[model_name].copy()
        # Remove the actual model object from return value
        info.pop('model', None)
        return info
    
    def list_loaded_models(self) -> list:
        """List all loaded models."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"Unloaded model {model_name}")
            return True
        return False
    
    def unload_all_models(self):
        """Unload all models from memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Unloaded all models")
    
    def get_available_model_types(self) -> list:
        """Get list of available model types."""
        if ONEM_TORCH_AVAILABLE:
            return ['2d_unet', '3d_unet', '2d_attention_unet', '3d_vnet']
        else:
            return []