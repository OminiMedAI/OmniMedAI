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
            state_dict = self._checkpoint_state_dict(checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def reconstruct_array(self, image):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for reconstruction") from exc

        array = np.asarray(image, dtype=np.float32)
        mode = self.config.model_parameters.get("mode", "volume")
        if mode == "slice_wise":
            reconstructed = self._reconstruct_slices(array)
        else:
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
                "mode": mode,
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

    def _reconstruct_slices(self, array):
        """Reconstruct a 3D volume as batches of two-dimensional slices."""
        import numpy as np

        if array.ndim != 3:
            raise ValueError("slice_wise reconstruction requires a 3D array")
        slice_axis = int(self.config.model_parameters.get("slice_axis", -1))
        slice_axis %= array.ndim
        slices = np.moveaxis(array, slice_axis, 0)
        normalized, location, scale = self._normalize_for_model(slices)
        reconstructed_batches = []

        for start in range(0, len(normalized), self.config.batch_size):
            batch = normalized[start : start + self.config.batch_size]
            if self.preprocess:
                tensor = self.preprocess(batch)
            else:
                tensor = self.torch.from_numpy(batch[:, None, :, :]).float()
            tensor = tensor.to(self.device)
            with self.torch.no_grad():
                output = self.model(tensor)
            if self.postprocess:
                output_array = np.asarray(self.postprocess(output))
            else:
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output_array = output.detach().cpu().numpy()
                if output_array.ndim == 4 and output_array.shape[1] == 1:
                    output_array = output_array[:, 0]
            if output_array.ndim != 3 or output_array.shape[0] != len(batch):
                raise ValueError(
                    "Slice-wise model output must have shape (batch, height, width) "
                    "or (batch, 1, height, width)"
                )
            reconstructed_batches.append(output_array)

        reconstructed = np.concatenate(reconstructed_batches, axis=0)
        reconstructed = reconstructed * scale + location
        return np.moveaxis(reconstructed, 0, slice_axis)

    def _normalize_for_model(self, slices):
        """Normalize model input while retaining values needed for inversion."""
        import numpy as np

        method = self.config.model_parameters.get("normalization", "none")
        if method in {None, "none"}:
            return slices, 0.0, 1.0
        finite = np.isfinite(slices)
        nonzero = finite & (slices != 0)
        values = slices[nonzero] if nonzero.any() else slices[finite]
        if not values.size:
            raise ValueError("Input image contains no finite values")
        if method == "z_score":
            location = float(values.mean())
            scale = float(values.std())
            if scale == 0:
                scale = 1.0
            return (slices - location) / scale, location, scale
        if method == "min_max":
            location = float(values.min())
            scale = float(values.max() - location)
            if scale == 0:
                scale = 1.0
            return (slices - location) / scale, location, scale
        raise ValueError(f"Unsupported reconstruction normalization: {method}")

    def _resolve_device(self, requested):
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested

    def _checkpoint_state_dict(self, checkpoint):
        """Resolve common generator checkpoint layouts without hiding mismatches."""
        if not isinstance(checkpoint, dict):
            return checkpoint
        requested_key = self.config.model_parameters.get("state_dict_key")
        candidate_keys = [
            requested_key,
            "state_dict",
            "generator",
            "generator_state_dict",
            "model_state_dict",
            "model",
        ]
        state_dict = checkpoint
        for key in candidate_keys:
            if key and isinstance(checkpoint.get(key), dict):
                state_dict = checkpoint[key]
                break
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {
                str(key)[len("module.") :]: value
                for key, value in state_dict.items()
            }
        return state_dict

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
