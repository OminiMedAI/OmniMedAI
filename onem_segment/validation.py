"""Validation and agreement metrics for manual or automatic ROI masks."""

from pathlib import Path
from typing import Sequence


def _load_nifti(path):
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise ImportError("Mask validation requires nibabel and numpy") from exc
    image = nib.load(str(path))
    return image, np.asarray(image.get_fdata())


def validate_external_mask(
    image_path,
    mask_path,
    require_binary: bool = True,
    affine_tolerance: float = 1e-4,
):
    """Validate an externally created mask against its source image."""
    import numpy as np

    image, image_data = _load_nifti(image_path)
    mask, mask_data = _load_nifti(mask_path)
    issues = []
    if image_data.shape != mask_data.shape:
        issues.append(f"shape mismatch: {image_data.shape} vs {mask_data.shape}")
    if not np.allclose(image.affine, mask.affine, atol=affine_tolerance):
        issues.append("affine mismatch")
    unique_values = np.unique(mask_data)
    if require_binary and not set(unique_values).issubset({0, 1}):
        issues.append(f"mask is not binary: values={unique_values[:10].tolist()}")
    if not np.isfinite(mask_data).all():
        issues.append("mask contains NaN or infinite values")
    foreground_voxels = int((mask_data > 0).sum())
    if foreground_voxels == 0:
        issues.append("mask is empty")
    return {
        "valid": not issues,
        "image_path": str(Path(image_path)),
        "mask_path": str(Path(mask_path)),
        "shape": list(mask_data.shape),
        "foreground_voxels": foreground_voxels,
        "issues": issues,
    }


def validate_multisequence_masks(
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
    affine_tolerance: float = 1e-4,
):
    """Validate one image-mask pair per sequence and summarize consistency."""
    if len(image_paths) != len(mask_paths):
        raise ValueError("image_paths and mask_paths must have the same length")
    reports = [
        validate_external_mask(image, mask, affine_tolerance=affine_tolerance)
        for image, mask in zip(image_paths, mask_paths)
    ]
    shapes = {tuple(report["shape"]) for report in reports}
    return {
        "valid": all(report["valid"] for report in reports) and len(shapes) == 1,
        "pair_reports": reports,
        "consistent_shapes": len(shapes) == 1,
    }


def segmentation_agreement(reference, prediction, spacing=None):
    """Calculate Dice, Jaccard, volume difference, and optional HD95."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for segmentation metrics") from exc
    reference = np.asarray(reference) > 0
    prediction = np.asarray(prediction) > 0
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    intersection = int((reference & prediction).sum())
    reference_volume = int(reference.sum())
    prediction_volume = int(prediction.sum())
    denominator = reference_volume + prediction_volume
    dice = 1.0 if denominator == 0 else 2 * intersection / denominator
    union = int((reference | prediction).sum())
    jaccard = 1.0 if union == 0 else intersection / union
    volume_difference = (
        0.0
        if reference_volume == 0
        else (prediction_volume - reference_volume) / reference_volume
    )
    result = {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "reference_volume_voxels": reference_volume,
        "prediction_volume_voxels": prediction_volume,
        "relative_volume_difference": float(volume_difference),
        "hd95": None,
    }
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        reference_surface = reference ^ binary_erosion(reference)
        prediction_surface = prediction ^ binary_erosion(prediction)
        if reference_surface.any() and prediction_surface.any():
            sampling = spacing if spacing is not None else None
            distance_to_prediction = distance_transform_edt(
                ~prediction_surface, sampling=sampling
            )
            distance_to_reference = distance_transform_edt(
                ~reference_surface, sampling=sampling
            )
            distances = np.concatenate(
                [
                    distance_to_prediction[reference_surface],
                    distance_to_reference[prediction_surface],
                ]
            )
            result["hd95"] = float(np.percentile(distances, 95))
    except ImportError:
        pass
    return result
