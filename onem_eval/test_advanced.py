"""Tests for advanced evaluation and reporting utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

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


class TestWorkflowManifest(unittest.TestCase):
    def test_manifest_save_and_completeness(self):
        from onem_eval import WorkflowManifest

        manifest = WorkflowManifest(
            study_name="Example",
            cohort_name="Training",
            task="classification",
        )
        self.assertFalse(manifest.completeness_report()["complete"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = manifest.save_json(Path(tmpdir) / "manifest.json")
            self.assertTrue(path.exists())


class TestAdvancedEvaluation(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas not available")
    def test_calibration_and_decision_curve(self):
        from onem_eval import calibration_table, decision_curve

        y_true = [0, 0, 1, 1, 0, 1]
        y_score = [0.1, 0.2, 0.8, 0.7, 0.3, 0.9]
        self.assertGreater(len(calibration_table(y_true, y_score, n_bins=3)), 0)
        self.assertEqual(len(decision_curve(y_true, y_score, thresholds=[0.2, 0.5])), 2)

    @unittest.skipUnless(PANDAS_AVAILABLE and SKLEARN_AVAILABLE, "pandas or scikit-learn not available")
    def test_bootstrap_auc(self):
        from onem_eval import bootstrap_auc_ci

        report = bootstrap_auc_ci(
            [0, 0, 1, 1, 0, 1],
            [0.1, 0.2, 0.8, 0.7, 0.3, 0.9],
            n_bootstraps=50,
        )
        self.assertGreaterEqual(report["auc"], 0.9)


if __name__ == "__main__":
    unittest.main()
