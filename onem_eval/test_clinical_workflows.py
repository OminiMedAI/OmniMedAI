"""Synthetic tests for survival and treatment workflow inputs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestClinicalWorkflowTables(unittest.TestCase):
    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_survival_validation_and_risk_table(self):
        from onem_eval import number_at_risk_table, validate_survival_table

        table = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4"],
                "time_months": [3.0, 6.0, 9.0, 12.0],
                "event": [1, 0, 1, 0],
                "tsr_group": ["L", "L", "H", "H"],
            }
        )
        validated = validate_survival_table(
            table, "time_months", "event"
        )
        self.assertEqual(len(validated), 4)
        risk = number_at_risk_table(
            table,
            "time_months",
            "event",
            time_points=[0, 6, 12],
            group_column="tsr_group",
        )
        self.assertEqual(len(risk), 6)

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_waterfall_table(self):
        from onem_eval import waterfall_table

        table = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "best_change_percent": [15.0, -42.0, -10.0],
                "tsr_group": ["H", "H", "L"],
            }
        )
        result = waterfall_table(
            table,
            change_column="best_change_percent",
            group_column="tsr_group",
        )
        self.assertEqual(result.iloc[0]["patient_id"], "p2")
        self.assertEqual(result["plot_order"].tolist(), [1, 2, 3])

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas unavailable")
    def test_stabilized_iptw_and_balance_table(self):
        from onem_eval import (
            estimate_propensity_weights,
            standardized_mean_differences,
        )

        table = pd.DataFrame(
            {
                "patient_id": [f"p{index}" for index in range(20)],
                "treatment": [0, 1] * 10,
                "age": list(range(50, 70)),
                "sex": ["F", "M"] * 10,
            }
        )
        propensity = estimate_propensity_weights(
            table,
            treatment_column="treatment",
            covariates=["age", "sex"],
            stabilized=True,
        )
        self.assertTrue(propensity["stabilized"])
        self.assertTrue((propensity["weights"]["weight"] > 0).all())
        weighted = table.merge(
            propensity["weights"][["patient_id", "weight"]],
            on="patient_id",
        )
        balance = standardized_mean_differences(
            weighted,
            treatment_column="treatment",
            covariates=["age", "sex"],
            weight_column="weight",
        )
        self.assertEqual(set(balance["weighted"]), {True})
        self.assertIn("absolute_smd", balance.columns)


if __name__ == "__main__":
    unittest.main()
