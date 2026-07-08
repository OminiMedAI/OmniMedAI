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

    def test_xgboost_expanded_grid_contains_reviewer_ranges(self):
        from onem_modeling import xgboost_param_grid

        grid = xgboost_param_grid("expanded")
        self.assertEqual(grid["model__n_estimators"], [50, 80, 100, 150, 200])
        self.assertEqual(grid["model__max_depth"], [2, 3, 4, 5, 6, 7])
        self.assertEqual(grid["model__learning_rate"], [0.03, 0.05, 0.08, 0.1])
        self.assertEqual(grid["model__min_child_weight"], [3, 5, 8, 10])
        self.assertEqual(grid["model__gamma"], [0, 0.05, 0.1, 0.125, 0.2, 0.5])
        self.assertEqual(grid["model__subsample"], [0.6, 0.7, 0.8])
        self.assertEqual(grid["model__colsample_bytree"], [0.5, 0.6, 0.7, 0.8])
        self.assertEqual(grid["model__reg_alpha"], [0, 0.05, 0.1, 0.2, 0.5])
        self.assertEqual(grid["model__reg_lambda"], [1, 2, 3, 5, 10])

    def test_common_model_grids_are_available(self):
        from onem_modeling import model_param_grid

        expected_keys = {
            "logistic_regression": "model__C",
            "svm": "model__kernel",
            "random_forest": "model__max_depth",
            "extra_trees": "model__max_depth",
            "knn": "model__n_neighbors",
            "naive_bayes": "model__var_smoothing",
        }
        for model_type, key in expected_keys.items():
            grid = model_param_grid(model_type, "compact")
            if isinstance(grid, list):
                flattened = set().union(*(item.keys() for item in grid))
                self.assertIn(key, flattened)
            else:
                self.assertIn(key, grid)


if __name__ == "__main__":
    unittest.main()
