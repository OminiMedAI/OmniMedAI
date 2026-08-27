"""Tests for segmentation validation utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class TestSegmentationAgreement(unittest.TestCase):
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy unavailable")
    def test_identical_masks(self):
        from onem_segment import segmentation_agreement

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        result = segmentation_agreement(mask, mask)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["jaccard"], 1.0)


class TestExternalMaskValidation(unittest.TestCase):
    @unittest.skipUnless(
        NUMPY_AVAILABLE and NIBABEL_AVAILABLE,
        "numpy or nibabel unavailable",
    )
    def test_accepts_binary_values_with_resampling_roundoff(self):
        from onem_segment import validate_external_mask

        image_data = np.zeros((4, 4, 2), dtype=np.float32)
        mask_data = np.zeros_like(image_data)
        mask_data[1:3, 1:3, :] = 1.00000006
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image_path)
            nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

            report = validate_external_mask(image_path, mask_path)

        self.assertTrue(report["valid"])
        self.assertEqual(report["foreground_voxels"], 8)


if __name__ == "__main__":
    unittest.main()
