"""Shared configuration and result types for image reconstruction."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ReconstructionConfig:
    """Configuration shared by interpolation and learned reconstruction."""

    algorithm: str = "interpolation"
    scale_factors: Tuple[float, ...] = (2.0, 2.0, 1.0)
    interpolation: str = "cubic"
    batch_size: int = 1
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    preserve_intensity_range: bool = True
    model_parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if not self.algorithm or not isinstance(self.algorithm, str):
            raise ValueError("algorithm must be a non-empty string")
        if not self.scale_factors or any(value <= 0 for value in self.scale_factors):
            raise ValueError("scale_factors must contain positive values")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.interpolation not in {"nearest", "linear", "cubic", "lanczos"}:
            raise ValueError("interpolation must be nearest, linear, cubic, or lanczos")

    def to_dict(self):
        self.validate()
        return asdict(self)


@dataclass
class ReconstructionResult:
    """Reconstructed image plus provenance metadata."""

    image: Any
    metadata: Dict[str, Any]
    output_path: Optional[str] = None
