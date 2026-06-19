"""Tests for clinical table checks."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestClinicalValidation(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas not available")
    def test_stage_size_consistency(self):
        from onem_process import check_stage_size_consistency

        table = pd.DataFrame(
            {
                "t_stage": ["T1", "T3", "T2"],
                "tumor_diameter_cm": [2.4, 3.7, 3.0],
            }
        )
        self.assertEqual(len(check_stage_size_consistency(table)), 2)


if __name__ == "__main__":
    unittest.main()
