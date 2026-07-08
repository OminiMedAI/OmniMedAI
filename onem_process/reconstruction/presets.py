"""Reusable reconstruction presets for common medical SR workflows."""

from typing import Optional

from .registry import create_reconstruction_config


def _mri_4x_config(
    algorithm: str,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
    **model_parameters,
):
    return create_reconstruction_config(
        algorithm,
        scale_factors=(4.0, 4.0, 1.0),
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        mode="slice_wise",
        input_channels=1,
        normalization="z_score",
        output_format="nifti",
        **model_parameters,
    )


def srgan_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise SRGAN MRI reconstruction config."""
    return _mri_4x_config("srgan", checkpoint_path, batch_size)


def esrgan_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise ESRGAN MRI reconstruction config."""
    return _mri_4x_config("esrgan", checkpoint_path, batch_size)


def rdgan_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise RDGAN MRI reconstruction config."""
    return _mri_4x_config("rdgan", checkpoint_path, batch_size)


def swinir_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise SwinIR MRI reconstruction config."""
    return _mri_4x_config("swinir", checkpoint_path, batch_size)


def swin2sr_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise Swin2SR MRI reconstruction config."""
    return _mri_4x_config("swin2sr", checkpoint_path, batch_size)


def hat_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise HAT MRI reconstruction config."""
    return _mri_4x_config("hat", checkpoint_path, batch_size)


def custom_torch_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
    architecture: str = "user_defined",
):
    """Create a 4x MRI config for a user-defined PyTorch SR model."""
    return _mri_4x_config(
        "custom_torch",
        checkpoint_path,
        batch_size,
        architecture=architecture,
    )


RECONSTRUCTION_PRESETS = {
    "custom_torch_4x_mri": custom_torch_4x_mri_config,
    "esrgan_4x_mri": esrgan_4x_mri_config,
    "hat_4x_mri": hat_4x_mri_config,
    "rdgan_4x_mri": rdgan_4x_mri_config,
    "srgan_4x_mri": srgan_4x_mri_config,
    "swin2sr_4x_mri": swin2sr_4x_mri_config,
    "swinir_4x_mri": swinir_4x_mri_config,
}
