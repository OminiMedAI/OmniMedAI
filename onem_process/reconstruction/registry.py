"""Algorithm registry for reusable medical image super-resolution."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .base import ReconstructionConfig


@dataclass(frozen=True)
class ReconstructionAlgorithmSpec:
    """Public metadata for a supported reconstruction algorithm family."""

    name: str
    family: str
    framework: str
    default_scale_factors: Tuple[float, ...] = (4.0, 4.0, 1.0)
    requires_checkpoint: bool = True
    description: str = ""


_REGISTRY: Dict[str, ReconstructionAlgorithmSpec] = {}


def register_reconstruction_algorithm(spec: ReconstructionAlgorithmSpec):
    """Register or replace a reconstruction algorithm specification."""
    if not spec.name:
        raise ValueError("Algorithm name cannot be empty")
    _REGISTRY[spec.name.lower()] = spec
    return spec


def get_reconstruction_algorithm(name: str) -> ReconstructionAlgorithmSpec:
    """Return metadata for a registered algorithm."""
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown reconstruction algorithm: {name}. Known: {known}")
    return _REGISTRY[key]


def list_reconstruction_algorithms() -> Dict[str, ReconstructionAlgorithmSpec]:
    """List available algorithm specifications."""
    return dict(sorted(_REGISTRY.items()))


def create_reconstruction_config(
    algorithm: str,
    scale_factors: Optional[Tuple[float, ...]] = None,
    batch_size: int = 1,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    preserve_intensity_range: bool = True,
    **model_parameters,
) -> ReconstructionConfig:
    """Create a validated config for a registered reconstruction algorithm."""
    spec = get_reconstruction_algorithm(algorithm)
    config = ReconstructionConfig(
        algorithm=spec.name,
        scale_factors=scale_factors or spec.default_scale_factors,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        device=device,
        preserve_intensity_range=preserve_intensity_range,
        model_parameters={
            "family": spec.family,
            "framework": spec.framework,
            **model_parameters,
        },
    )
    config.validate()
    return config


for _spec in (
    ReconstructionAlgorithmSpec(
        name="interpolation",
        family="baseline",
        framework="scipy",
        default_scale_factors=(2.0, 2.0, 1.0),
        requires_checkpoint=False,
        description="Deterministic nearest, linear, cubic, or Lanczos resizing.",
    ),
    ReconstructionAlgorithmSpec("srcnn", "cnn", "pytorch", description="SRCNN-style CNN super-resolution."),
    ReconstructionAlgorithmSpec("fsrcnn", "cnn", "pytorch", description="Fast SRCNN-style reconstruction."),
    ReconstructionAlgorithmSpec("edsr", "cnn", "pytorch", description="Enhanced deep residual SR."),
    ReconstructionAlgorithmSpec("rdn", "cnn", "pytorch", description="Residual dense network SR."),
    ReconstructionAlgorithmSpec("rcan", "cnn", "pytorch", description="Residual channel-attention SR."),
    ReconstructionAlgorithmSpec("srgan", "gan", "pytorch", description="SRGAN generator-based reconstruction."),
    ReconstructionAlgorithmSpec("esrgan", "gan", "pytorch", description="Enhanced SRGAN reconstruction."),
    ReconstructionAlgorithmSpec("rdgan", "gan", "pytorch", description="Residual dense GAN reconstruction."),
    ReconstructionAlgorithmSpec("swinir", "transformer", "pytorch", description="Swin Transformer image restoration."),
    ReconstructionAlgorithmSpec("swin2sr", "transformer", "pytorch", description="Swin2SR transformer reconstruction."),
    ReconstructionAlgorithmSpec("hat", "transformer", "pytorch", description="Hybrid attention transformer SR."),
    ReconstructionAlgorithmSpec("custom_torch", "custom", "pytorch", description="User-supplied PyTorch SR model."),
):
    register_reconstruction_algorithm(_spec)
