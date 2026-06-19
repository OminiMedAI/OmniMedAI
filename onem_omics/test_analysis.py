"""Synthetic tests for omics validation and cell composition."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestOmicsAnalysis(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_expression_and_mecaf_workflow(self):
        from onem_omics import meca_fibroblast_summary, validate_expression_metadata

        expression = pd.DataFrame(
            {"gene_id": ["g1", "g2"], "s1": [1.0, 2.0], "s2": [3.0, 4.0]}
        )
        metadata = pd.DataFrame({"sample_id": ["s1", "s2"]})
        self.assertTrue(
            validate_expression_metadata(expression, metadata)["valid"]
        )

        cells = pd.DataFrame(
            {
                "sample_id": ["s1", "s1", "s2", "s2"],
                "tsr_group": ["H", "H", "L", "L"],
                "cell_type": ["meCAF", "T_cell", "meCAF", "meCAF"],
            }
        )
        result = meca_fibroblast_summary(cells)
        self.assertEqual(len(result["sample_proportions"]), 2)


if __name__ == "__main__":
    unittest.main()
