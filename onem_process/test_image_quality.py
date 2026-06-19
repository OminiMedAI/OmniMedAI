"""Synthetic tests for reconstructed-image quality metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestImageQuality(unittest.TestCase):
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy unavailable")
    def test_identical_images(self):
        from onem_process import compare_image_quality

        image = np.arange(64, dtype=float).reshape(8, 8)
        result = compare_image_quality(image, image)
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["psnr"], float("inf"))


if __name__ == "__main__":
    unittest.main()
