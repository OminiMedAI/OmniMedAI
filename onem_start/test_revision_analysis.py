"""Integration tests for the configuration-driven revision workflow."""

import json
import sys
import tempfile
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


class TestRevisionAnalysisWorkflow(unittest.TestCase):
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "analysis dependencies unavailable")
    def test_modeling_writes_random_seed_stability_report(self):
        from onem_start.revision_analysis import run_modeling

        rng = np.random.default_rng(31)
        n_patients = 40
        signal = rng.normal(size=n_patients)
        labels = (signal + rng.normal(scale=0.35, size=n_patients) > 0).astype(int)
        table = pd.DataFrame(
            {
                "patient_id": [f"p{index:03d}" for index in range(n_patients)],
                "label": labels,
                "signal": signal,
                "signal_2": signal + rng.normal(scale=0.2, size=n_patients),
                "noise": rng.normal(size=n_patients),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "features.csv"
            output_dir = temp_path / "results"
            output_dir.mkdir()
            table.to_csv(input_csv, index=False)

            run_modeling(
                {
                    "input_csv": str(input_csv),
                    "patient_column": "patient_id",
                    "label_column": "label",
                    "config": {
                        "task": "classification",
                        "model_type": "logistic_regression",
                        "feature_selection": "radiomics_sequence",
                        "n_features": 3,
                        "outer_folds": 2,
                        "inner_folds": 2,
                        "random_state": 100,
                        "param_grid": {"model__C": [1.0]},
                        "selection_parameters": {
                            "univariate_p_threshold": 0.2,
                            "correlation_threshold": 0.9,
                            "lasso_cv_folds": 2,
                        },
                    },
                    "seed_stability": {
                        "enabled": True,
                        "n_repeats": 3,
                    },
                },
                output_dir,
            )

            report = json.loads(
                (output_dir / "model_seed_feature_stability.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["n_runs"], 3)
            self.assertEqual(report["stability"]["comparison_scope"], "random_seed")
            self.assertEqual(len(report["stability"]["pairwise_jaccard"]), 3)
            self.assertTrue((output_dir / "model_feature_stability.json").exists())


if __name__ == "__main__":
    unittest.main()
