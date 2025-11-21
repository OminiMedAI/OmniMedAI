"""
TITAN Model for Pathology Deep Transfer Learning

This module implements the TITAN (Transfer learning for pathology) model
for deep feature extraction from pathology images using transfer learning.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Try to import torchvision and timm
try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("torchvision not available. Install with: pip install torchvision")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm not available. Install with: pip install timm")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")


class TITANModel(nn.Module):
    """
    TITAN (Transfer learning for pathology) model.
    
    This model uses pre-trained backbones with custom heads for pathology
    feature extraction. Supports multiple backbone architectures and 
    feature extraction at different levels.
    """
    
    def __init__(self, 
                 backbone_name: str = 'resnet50',
                 pretrained: bool = True,
                 num_classes: int = 1,
                 feature_dim: int = 1024,
                 dropout_rate: float = 0.1,
                 freeze_backbone: bool = False,
                 use_attention: bool = True):
        """
        Initialize TITAN model.
        
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            feature_dim: Dimension of extracted features
            dropout_rate: Dropout rate
            freeze_backbone: Whether to freeze backbone parameters
            use_attention: Whether to use attention mechanisms
        """
        super(TITANModel, self).__init__()
        
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        self._check_dependencies()
        
        # Build backbone
        self.backbone = self._build_backbone(backbone_name, pretrained)
        
        # Get feature dimension from backbone
        backbone_features = self._get_backbone_features(backbone_name)
        
        # Feature extraction head
        self.feature_extractor = self._build_feature_extractor(
            backbone_features, feature_dim, use_attention
        )
        
        # Classification head (optional)
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            self.classifier = None
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Initialize weights
        self._initialize_weights()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required. Install with: pip install torchvision")
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required. Install with: pip install Pillow")
    
    def _build_backbone(self, backbone_name: str, pretrained: bool):
        """Build the backbone architecture."""
        if TIMM_AVAILABLE:
            # Try to use timm for more models
            try:
                if pretrained:
                    model = timm.create_model(backbone_name, pretrained=True)
                else:
                    model = timm.create_model(backbone_name, pretrained=False)
                
                self.logger.info(f"Created {backbone_name} backbone using timm")
                return model
            except:
                pass
        
        # Fallback to torchvision models
        if backbone_name.startswith('resnet'):
            return self._build_resnet(backbone_name, pretrained)
        elif backbone_name.startswith('efficientnet'):
            return self._build_efficientnet(backbone_name, pretrained)
        elif backbone_name.startswith('densenet'):
            return self._build_densenet(backbone_name, pretrained)
        elif backbone_name.startswith('vgg'):
            return self._build_vgg(backbone_name, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _build_resnet(self, backbone_name: str, pretrained: bool):
        """Build ResNet backbone."""
        if backbone_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif backbone_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
        
        # Remove final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model
    
    def _build_efficientnet(self, backbone_name: str, pretrained: bool):
        """Build EfficientNet backbone."""
        if backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=pretrained)
        elif backbone_name == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {backbone_name}")
        
        # Get features before classification
        features = model.features
        return features
    
    def _build_densenet(self, backbone_name: str, pretrained: bool):
        """Build DenseNet backbone."""
        if backbone_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif backbone_name == 'densenet161':
            model = models.densenet161(pretrained=pretrained)
        elif backbone_name == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
        elif backbone_name == 'densenet201':
            model = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported DenseNet variant: {backbone_name}")
        
        # Get features before classification
        features = model.features
        return features
    
    def _build_vgg(self, backbone_name: str, pretrained: bool):
        """Build VGG backbone."""
        if backbone_name == 'vgg11':
            model = models.vgg11(pretrained=pretrained)
        elif backbone_name == 'vgg13':
            model = models.vgg13(pretrained=pretrained)
        elif backbone_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif backbone_name == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported VGG variant: {backbone_name}")
        
        # Get features before classification
        features = model.features
        return features
    
    def _get_backbone_features(self, backbone_name: str) -> int:
        """Get the output feature dimension of backbone."""
        # Typical feature dimensions for different backbones
        feature_dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560,
            'densenet121': 1024,
            'densenet161': 2208,
            'densenet169': 1664,
            'densenet201': 1920,
            'vgg11': 512,
            'vgg13': 512,
            'vgg16': 512,
            'vgg19': 512
        }
        
        return feature_dims.get(backbone_name, 2048)
    
    def _build_feature_extractor(self, backbone_features: int, feature_dim: int, 
                               use_attention: bool) -> nn.Module:
        """Build feature extraction head."""
        if use_attention:
            # Attention-based feature extraction
            return AttentionFeatureExtractor(backbone_features, feature_dim)
        else:
            # Standard feature extraction
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(backbone_features, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim)
            )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone parameters frozen")
    
    def _initialize_weights(self):
        """Initialize weights of new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_features: Whether to return features instead of classification
            
        Returns:
            Features tensor or (features, classification) tuple
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Extract deep features
        deep_features = self.feature_extractor(features)
        
        if not return_features or self.classifier is None:
            return deep_features
        
        # Classification
        logits = self.classifier(deep_features)
        
        return deep_features, logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only."""
        return self.forward(x, return_features=True)
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        return self.feature_dim
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone_name': self.backbone_name,
            'pretrained': self.pretrained,
            'feature_dim': self.feature_dim,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'has_classifier': self.classifier is not None
        }


class AttentionFeatureExtractor(nn.Module):
    """Attention-based feature extractor."""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8):
        super(AttentionFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, output_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x = x.view(x.size(0), x.size(1))  # [B, C]
        
        # Reshape for attention [B, 1, C]
        x = x.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # Global average pooling and reshape
        x = x.mean(dim=1)  # [B, C]
        
        return x


# Utility functions for TITAN model
def create_titan_model(backbone_name: str = 'resnet50',
                     pretrained: bool = True,
                     feature_dim: int = 1024,
                     use_attention: bool = True,
                     checkpoint_path: str = None) -> TITANModel:
    """
    Create a TITAN model with specified parameters.
    
    Args:
        backbone_name: Name of backbone architecture
        pretrained: Whether to use pretrained weights
        feature_dim: Dimension of extracted features
        use_attention: Whether to use attention mechanisms
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Initialized TITAN model
    """
    model = TITANModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        feature_dim=feature_dim,
        use_attention=use_attention
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    return model


def get_available_backbones() -> List[str]:
    """Get list of available backbone architectures."""
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
    
    # If timm is available, add more models
    if TIMM_AVAILABLE:
        try:
            timm_models = timm.list_models(pretrained=True)
            # Filter for common image classification models
            common_backbones = [model for model in timm_models 
                              if any(x in model.lower() for x in ['resnet', 'efficientnet', 'densenet', 'vgg', 'mobilenet'])]
            backbones.extend(common_backbones[:50])  # Limit to first 50
        except:
            pass
    
    return sorted(list(set(backbones)))


def benchmark_backbone_performance(model: TITANModel, 
                             input_tensor: torch.Tensor,
                             num_iterations: int = 100) -> Dict:
    """
    Benchmark backbone performance.
    
    Args:
        model: TITAN model to benchmark
        input_tensor: Input tensor for testing
        num_iterations: Number of iterations for timing
        
    Returns:
        Performance metrics dictionary
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.extract_features(input_tensor)
    
    # Benchmark
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            features = model.extract_features(input_tensor)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    return {
        'total_time': total_time,
        'avg_time_per_inference': avg_time,
        'fps': fps,
        'feature_dim': features.shape[-1],
        'batch_size': input_tensor.shape[0]
    }