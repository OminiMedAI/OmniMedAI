"""Medical image super-resolution reconstruction interfaces."""

from .base import ReconstructionConfig, ReconstructionResult
from .interpolation import InterpolationReconstructor
from .presets import RECONSTRUCTION_PRESETS, srgan_4x_mri_config
from .registry import (
    ReconstructionAlgorithmSpec,
    create_reconstruction_config,
    get_reconstruction_algorithm,
    list_reconstruction_algorithms,
    register_reconstruction_algorithm,
)
from .torch_adapter import TorchSuperResolutionAdapter

__all__ = [
    "InterpolationReconstructor",
    "RECONSTRUCTION_PRESETS",
    "ReconstructionAlgorithmSpec",
    "ReconstructionConfig",
    "ReconstructionResult",
    "TorchSuperResolutionAdapter",
    "create_reconstruction_config",
    "get_reconstruction_algorithm",
    "list_reconstruction_algorithms",
    "register_reconstruction_algorithm",
    "srgan_4x_mri_config",
]
