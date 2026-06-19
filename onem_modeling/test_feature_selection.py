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


if __name__ == "__main__":
    unittest.main()
