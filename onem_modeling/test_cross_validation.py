"""Synthetic tests for nested patient-level validation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class TestNestedPatientValidation(unittest.TestCase):
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "numpy, pandas, or scikit-learn unavailable")
    def test_nested_validation_returns_oof_predictions(self):
        from onem_modeling import NestedCVConfig, nested_patient_cross_validate

        rng = np.random.default_rng(7)
        n_patients = 40
        feature_a = rng.normal(size=n_patients)
        labels = (feature_a + rng.normal(scale=0.4, size=n_patients) > 0).astype(int)
        table = pd.DataFrame(
            {
                "patient_id": [f"p{index:03d}" for index in range(n_patients)],
                "label": labels,
                "feature_a": feature_a,
                "feature_b": rng.normal(size=n_patients),
                "feature_c": rng.normal(size=n_patients),
            }
        )
        result = nested_patient_cross_validate(
            table,
            label_column="label",
            config=NestedCVConfig(
                outer_folds=4,
                inner_folds=2,
                n_features=2,
                param_grid={"model__C": [0.1, 1.0]},
            ),
        )
        self.assertEqual(len(result["predictions"]), n_patients)
        self.assertEqual(result["predictions"]["patient_id"].nunique(), n_patients)
        self.assertEqual(len(result["fold_results"]), 4)


if __name__ == "__main__":
    unittest.main()
