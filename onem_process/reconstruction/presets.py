"""Reusable reconstruction presets for common medical SR workflows."""

from typing import Optional

from .registry import create_reconstruction_config


def srgan_4x_mri_config(
    checkpoint_path: Optional[str] = None,
    batch_size: int = 16,
):
    """Create the common 4x slice-wise SRGAN MRI reconstruction config."""
    return create_reconstruction_config(
        "srgan",
        scale_factors=(4.0, 4.0, 1.0),
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        mode="slice_wise",
        input_channels=1,
        normalization="z_score",
        output_format="nifti",
    )


RECONSTRUCTION_PRESETS = {
    "srgan_4x_mri": srgan_4x_mri_config,
}
