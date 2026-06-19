"""Synthetic tests for pathological TSR aggregation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestTSRGroundTruth(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_aggregate_and_icc(self):
        from onem_path import aggregate_tsr_ground_truth, tsr_interobserver_icc

        table = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "reader_id": ["r1", "r2", "r1", "r2", "r1", "r2"],
                "slide_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "specimen_type": ["resection"] * 6,
                "tsr_percent": [30, 32, 60, 62, 45, 47],
            }
        )
        result = aggregate_tsr_ground_truth(table, cutoff=50)
        self.assertEqual(len(result), 3)
        self.assertGreater(tsr_interobserver_icc(table)["icc2_1"], 0.9)


if __name__ == "__main__":
    unittest.main()
