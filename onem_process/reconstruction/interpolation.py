"""Deterministic interpolation baselines for medical image reconstruction."""

from pathlib import Path
from typing import Optional

from .base import ReconstructionConfig, ReconstructionResult


class InterpolationReconstructor:
    """Upsample arrays or NIfTI images with a documented interpolation baseline."""

    _ORDERS = {"nearest": 0, "linear": 1, "cubic": 3}

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self.config.validate()

    def reconstruct_array(self, image):
        try:
            import numpy as np
            from scipy import ndimage
        except ImportError as exc:
            raise ImportError(
                "Interpolation reconstruction requires numpy and scipy"
            ) from exc

        array = np.asarray(image)
        if array.ndim not in {2, 3}:
            raise ValueError("image must be a 2D or 3D array")
        factors = self._scale_factors_for(array.ndim)
        if self.config.interpolation == "lanczos":
            reconstructed = self._lanczos_resize(array, factors)
        else:
            reconstructed = ndimage.zoom(
                array,
                factors,
                order=self._ORDERS[self.config.interpolation],
                mode="nearest",
                prefilter=self.config.interpolation == "cubic",
            )
        if self.config.preserve_intensity_range:
            reconstructed = np.clip(reconstructed, array.min(), array.max())
        return ReconstructionResult(
            image=reconstructed.astype(array.dtype, copy=False),
            metadata={
                "algorithm": f"interpolation:{self.config.interpolation}",
                "input_shape": list(array.shape),
                "output_shape": list(reconstructed.shape),
                "scale_factors": list(factors),
                "checkpoint_path": None,
                "parameters": self.config.to_dict(),
            },
        )

    def reconstruct_nifti(self, input_path, output_path):
        try:
            import nibabel as nib
            import numpy as np
        except ImportError as exc:
            raise ImportError("NIfTI reconstruction requires nibabel and numpy") from exc

        source = nib.load(str(input_path))
        result = self.reconstruct_array(source.get_fdata())
        factors = self._scale_factors_for(source.ndim)
        affine = source.affine.copy()
        for axis, factor in enumerate(factors[:3]):
            affine[:3, axis] = affine[:3, axis] / factor
        header = source.header.copy()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(result.image, affine, header), str(output_path))
        result.output_path = str(output_path)
        result.metadata.update(
            {
                "input_path": str(Path(input_path)),
                "output_path": str(output_path),
                "input_spacing": list(source.header.get_zooms()[: source.ndim]),
                "output_spacing": [
                    spacing / factor
                    for spacing, factor in zip(
                        source.header.get_zooms()[: source.ndim], factors
                    )
                ],
            }
        )
        return result

    def _scale_factors_for(self, ndim):
        factors = tuple(self.config.scale_factors)
        if len(factors) < ndim:
            factors = factors + (1.0,) * (ndim - len(factors))
        return factors[:ndim]

    def _lanczos_resize(self, array, factors):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Lanczos interpolation requires opencv-python"
            ) from exc
        target_height = max(1, round(array.shape[0] * factors[0]))
        target_width = max(1, round(array.shape[1] * factors[1]))
        if array.ndim == 2:
            return cv2.resize(
                array,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        slices = [
            cv2.resize(
                array[..., index],
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            for index in range(array.shape[-1])
        ]
        resized = np.stack(slices, axis=-1)
        if factors[2] != 1:
            from scipy import ndimage

            resized = ndimage.zoom(resized, (1, 1, factors[2]), order=3)
        return resized
