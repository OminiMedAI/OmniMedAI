"""Tests for reusable super-resolution reconstruction interfaces."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import scipy  # noqa: F401

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class TestInterpolationReconstruction(unittest.TestCase):
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "numpy or scipy unavailable")
    def test_cubic_array_reconstruction(self):
        from onem_process import InterpolationReconstructor, ReconstructionConfig

        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = InterpolationReconstructor(
            ReconstructionConfig(scale_factors=(2, 3), interpolation="cubic")
        ).reconstruct_array(image)
        self.assertEqual(result.image.shape, (8, 12))
        self.assertEqual(result.metadata["algorithm"], "interpolation:cubic")
        self.assertEqual(result.metadata["scale_factors"], [2, 3])


if __name__ == "__main__":
    unittest.main()
