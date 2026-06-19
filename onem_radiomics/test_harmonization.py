"""Synthetic tests for multi-center feature summaries."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestBatchEffects(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_batch_effect_summary(self):
        from onem_radiomics import batch_effect_summary

        table = pd.DataFrame(
            {
                "center": ["A", "A", "B", "B"],
                "feature_1": [0.0, 1.0, 10.0, 11.0],
                "feature_2": [2.0, 3.0, 4.0, 5.0],
            }
        )
        result = batch_effect_summary(table, batch_column="center")
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["batch"]), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
