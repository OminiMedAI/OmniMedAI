"""Adapter for user-supplied PyTorch super-resolution models."""

from pathlib import Path
from typing import Callable, Optional

from .base import ReconstructionConfig, ReconstructionResult


class TorchSuperResolutionAdapter:
    """Run a caller-provided PyTorch model without prescribing architecture."""

    def __init__(
        self,
        model,
        config: Optional[ReconstructionConfig] = None,
        preprocess: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch reconstruction requires torch") from exc

        self.torch = torch
        self.config = config or ReconstructionConfig()
        self.config.validate()
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)
        if self.config.checkpoint_path:
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.device,
            )
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def reconstruct_array(self, image):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for reconstruction") from exc

        array = np.asarray(image, dtype=np.float32)
        tensor = self.preprocess(array) if self.preprocess else self._default_preprocess(array)
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        reconstructed = (
            self.postprocess(output)
            if self.postprocess
            else self._default_postprocess(output, array.ndim)
        )
        reconstructed = np.asarray(reconstructed)
        if self.config.preserve_intensity_range:
            reconstructed = np.clip(reconstructed, array.min(), array.max())
        return ReconstructionResult(
            image=reconstructed,
            metadata={
                "algorithm": self.config.algorithm,
                "model_class": self.model.__class__.__name__,
                "input_shape": list(array.shape),
                "output_shape": list(reconstructed.shape),
                "scale_factors": list(self.config.scale_factors),
                "batch_size": self.config.batch_size,
                "checkpoint_path": self.config.checkpoint_path,
                "parameters": self.config.to_dict(),
            },
        )

    def reconstruct_nifti(self, input_path, output_path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("NIfTI reconstruction requires nibabel") from exc

        source = nib.load(str(input_path))
        result = self.reconstruct_array(source.get_fdata())
        factors = tuple(
            result.image.shape[index] / source.shape[index]
            for index in range(source.ndim)
        )
        affine = source.affine.copy()
        for axis, factor in enumerate(factors[:3]):
            affine[:3, axis] = affine[:3, axis] / factor
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(result.image, affine, source.header.copy()),
            str(output_path),
        )
        result.output_path = str(output_path)
        result.metadata.update(
            {"input_path": str(input_path), "output_path": str(output_path)}
        )
        return result

    def _resolve_device(self, requested):
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested

    def _default_preprocess(self, array):
        tensor = self.torch.from_numpy(array).float()
        if array.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if array.ndim == 3:
            return tensor.unsqueeze(0).unsqueeze(0)
        raise ValueError("Default preprocessing supports 2D or 3D arrays")

    def _default_postprocess(self, output, input_ndim):
        if isinstance(output, (tuple, list)):
            output = output[0]
        array = output.detach().cpu().numpy()
        while array.ndim > input_ndim:
            array = array[0]
        return array
