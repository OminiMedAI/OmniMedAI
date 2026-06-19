"""Image-quality comparison for standard and reconstructed medical images."""

from pathlib import Path
from typing import Optional


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for image-quality analysis") from exc
    return np


def _load_array(value):
    np = _require_numpy()
    if isinstance(value, (str, Path)):
        path = str(value)
        if path.endswith((".nii", ".nii.gz")):
            try:
                import nibabel as nib
            except ImportError as exc:
                raise ImportError(
                    "nibabel is required to compare NIfTI images"
                ) from exc
            return nib.load(path).get_fdata().astype(float)
        if path.endswith(".npy"):
            return np.load(path).astype(float)
        raise ValueError("Supported image paths are .nii, .nii.gz, and .npy")
    return np.asarray(value, dtype=float)


def compare_image_quality(
    reference,
    reconstructed,
    roi_mask=None,
    background_mask=None,
    data_range: Optional[float] = None,
):
    """Compare a reconstructed image with a reference image.

    Inputs may be NumPy arrays, NIfTI paths, or ``.npy`` paths. PSNR and SSIM
    require a true reference image; SNR and CNR are included for reviewer-facing
    reconstruction reporting.
    """
    np = _require_numpy()
    reference_array = _load_array(reference)
    reconstructed_array = _load_array(reconstructed)
    if reference_array.shape != reconstructed_array.shape:
        raise ValueError(
            f"Image shapes differ: {reference_array.shape} vs {reconstructed_array.shape}"
        )
    if not np.isfinite(reference_array).all() or not np.isfinite(reconstructed_array).all():
        raise ValueError("Images contain NaN or infinite values")

    difference = reference_array - reconstructed_array
    mse = float(np.mean(difference ** 2))
    if data_range is None:
        data_range = float(reference_array.max() - reference_array.min())
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    psnr = float("inf") if mse == 0 else float(20 * np.log10(data_range) - 10 * np.log10(mse))

    try:
        from skimage.metrics import structural_similarity

        if reference_array.ndim == 3:
            ssim_values = [
                structural_similarity(
                    reference_array[..., index],
                    reconstructed_array[..., index],
                    data_range=data_range,
                )
                for index in range(reference_array.shape[-1])
                if min(reference_array[..., index].shape) >= 7
            ]
            ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
        else:
            ssim = float(
                structural_similarity(
                    reference_array,
                    reconstructed_array,
                    data_range=data_range,
                )
            )
    except ImportError:
        ssim = None

    analysis_mask = (
        _load_array(roi_mask).astype(bool)
        if roi_mask is not None
        else np.ones(reference_array.shape, dtype=bool)
    )
    if analysis_mask.shape != reference_array.shape:
        raise ValueError("ROI mask shape does not match image shape")
    signal = reconstructed_array[analysis_mask]
    snr = float(signal.mean() / signal.std()) if signal.std() > 0 else float("inf")

    cnr = None
    if background_mask is not None:
        background = _load_array(background_mask).astype(bool)
        if background.shape != reference_array.shape:
            raise ValueError("Background mask shape does not match image shape")
        background_values = reconstructed_array[background]
        pooled_variance = signal.var() + background_values.var()
        cnr = (
            float(abs(signal.mean() - background_values.mean()) / np.sqrt(pooled_variance))
            if pooled_variance > 0
            else float("inf")
        )

    return {
        "shape": list(reference_array.shape),
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "snr": snr,
        "cnr": cnr,
    }


def compare_image_pairs(
    manifest,
    reference_column: str = "reference_path",
    reconstructed_column: str = "reconstructed_path",
    patient_column: str = "patient_id",
    roi_column: Optional[str] = None,
    background_column: Optional[str] = None,
):
    """Compare image pairs listed in a CSV file or DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for image-pair manifests") from exc
    table = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    required = {patient_column, reference_column, reconstructed_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing image manifest columns: {missing}")
    rows = []
    for _, record in table.iterrows():
        metrics = compare_image_quality(
            record[reference_column],
            record[reconstructed_column],
            roi_mask=record[roi_column] if roi_column else None,
            background_mask=record[background_column] if background_column else None,
        )
        metrics[patient_column] = record[patient_column]
        rows.append(metrics)
    return pd.DataFrame(rows)
