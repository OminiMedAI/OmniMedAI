"""Tests for segmentation validation utilities."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestSegmentationAgreement(unittest.TestCase):
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy unavailable")
    def test_identical_masks(self):
        from onem_segment import segmentation_agreement

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        result = segmentation_agreement(mask, mask)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
