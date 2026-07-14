"""Tests for modeling validation utilities."""

import sys
import unittest
from pathlib import Path
from math import isclose

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestValidationUtilities(unittest.TestCase):
    def test_seed_stability(self):
        from onem_modeling import summarize_feature_selection_stability

        stability = summarize_feature_selection_stability(
            {1: ["a", "b"], 2: ["a", "b", "c"], 3: ["a", "b", "d"]}
        )
        self.assertEqual(stability["consensus_features"], ["a", "b"])
        self.assertEqual(len(stability["pairwise_jaccard"]), 3)
        self.assertTrue(isclose(stability["mean_pairwise_jaccard"], 11 / 18))
        self.assertTrue(isclose(stability["min_pairwise_jaccard"], 0.5))
        self.assertTrue(isclose(stability["max_pairwise_jaccard"], 2 / 3))
        self.assertEqual(stability["feature_selection_rate"]["a"], 1.0)
        self.assertTrue(isclose(stability["feature_selection_rate"]["c"], 1 / 3))

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas not available")
    def test_overlap_check(self):
        from onem_modeling import assert_no_patient_overlap

        first = pd.DataFrame({"patient_id": ["p1", "p2"]})
        second = pd.DataFrame({"patient_id": ["p3", "p4"]})
        self.assertTrue(assert_no_patient_overlap(first, second))

    @unittest.skipUnless(PANDAS_AVAILABLE and SKLEARN_AVAILABLE, "pandas or scikit-learn not available")
    def test_patient_split(self):
        from onem_modeling import (
            assert_no_patient_overlap,
            patient_level_train_test_split,
        )

        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
                "feature": range(8),
            }
        )
        train_df, test_df, summary = patient_level_train_test_split(
            df,
            test_size=0.5,
            random_state=7,
        )
        self.assertTrue(assert_no_patient_overlap(train_df, test_df))
        self.assertEqual(summary["split_level"], "patient")


if __name__ == "__main__":
    unittest.main()
