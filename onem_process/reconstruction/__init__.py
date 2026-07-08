"""Medical image super-resolution reconstruction interfaces."""

from .base import ReconstructionConfig, ReconstructionResult
from .interpolation import InterpolationReconstructor
from .presets import (
    RECONSTRUCTION_PRESETS,
    custom_torch_4x_mri_config,
    esrgan_4x_mri_config,
    hat_4x_mri_config,
    rdgan_4x_mri_config,
    srgan_4x_mri_config,
    swin2sr_4x_mri_config,
    swinir_4x_mri_config,
)
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
    "custom_torch_4x_mri_config",
    "esrgan_4x_mri_config",
    "get_reconstruction_algorithm",
    "hat_4x_mri_config",
    "list_reconstruction_algorithms",
    "rdgan_4x_mri_config",
    "register_reconstruction_algorithm",
    "srgan_4x_mri_config",
    "swin2sr_4x_mri_config",
    "swinir_4x_mri_config",
]
