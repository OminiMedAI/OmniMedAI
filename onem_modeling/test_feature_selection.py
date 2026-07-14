"""Synthetic tests for sequential radiomics feature selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    import scipy  # noqa: F401
    import sklearn  # noqa: F401

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class TestSequentialRadiomicsSelector(unittest.TestCase):
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "analysis dependencies unavailable")
    def test_selects_signal_features(self):
        from onem_modeling import FeatureSelectionConfig, SequentialRadiomicsSelector

        rng = np.random.default_rng(12)
        label = np.array([0] * 30 + [1] * 30)
        table = pd.DataFrame(
            {
                "signal": label + rng.normal(0, 0.2, 60),
                "signal_copy": label + rng.normal(0, 0.2, 60),
                "noise_1": rng.normal(size=60),
                "noise_2": rng.normal(size=60),
            }
        )
        selector = SequentialRadiomicsSelector(
            FeatureSelectionConfig(
                univariate_p_threshold=0.1,
                correlation_threshold=0.8,
                mrmr_features=2,
                lasso_cv_folds=3,
            )
        ).fit(table, label)
        self.assertIn("signal", selector.stage_features_["univariate"])
        self.assertGreaterEqual(len(selector.selected_features_), 1)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "analysis dependencies unavailable")
    def test_repeats_complete_selection_across_random_seeds(self):
        from onem_modeling import (
            FeatureSelectionConfig,
            repeated_seed_feature_selection,
        )

        rng = np.random.default_rng(21)
        label = np.array([0] * 30 + [1] * 30)
        table = pd.DataFrame(
            {
                "signal": label + rng.normal(0, 0.15, 60),
                "signal_2": label + rng.normal(0, 0.25, 60),
                "noise_1": rng.normal(size=60),
                "noise_2": rng.normal(size=60),
            }
        )
        result = repeated_seed_feature_selection(
            table,
            label,
            config=FeatureSelectionConfig(
                univariate_p_threshold=0.1,
                correlation_threshold=0.9,
                mrmr_features=3,
                lasso_cv_folds=3,
                random_state=100,
            ),
            n_repeats=10,
        )

        self.assertEqual(result["n_runs"], 10)
        self.assertEqual(len(result["selected_features_by_seed"]), 10)
        self.assertEqual(result["stability"]["comparison_scope"], "random_seed")
        self.assertEqual(len(result["stability"]["pairwise_jaccard"]), 45)
        self.assertIn("signal", result["stability"]["feature_frequency"])


if __name__ == "__main__":
    unittest.main()
