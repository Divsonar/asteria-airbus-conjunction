"""Early-only inference, target isolation and probability semantics."""
import unittest
import numpy as np
import pandas as pd
from src.pipeline import FEATURES, predict_messages

class StubModel:
    classes_ = np.array(["Critical", "Elevated", "Safe"])
    def predict_proba(self, features):
        assert list(features.columns) == FEATURES
        assert "risk" not in features.columns
        assert (features.time_to_tca >= 2).all()
        return np.tile([.1, .2, .7], (len(features), 1))

class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({name: [10.0] * 4 for name in FEATURES})
        self.frame["event_id"] = ["001", "002", "003", "004"]
        self.frame["source_row_id"] = ["019", "025", "030", "031"]
        self.frame["time_to_tca"] = [3.0, 2.0, 1.9999, "invalid"]
        self.frame["c_object_type"] = "DEBRIS"
        self.frame["risk"] = [-3, -30, -6, -1]
        self.frame["final_tier"] = "Critical"
    def test_output_preserves_rows_and_excludes_late_telemetry(self):
        result = predict_messages(self.frame, model=StubModel())
        self.assertEqual(result.source_row_id.tolist(), ["019", "025", "030", "031"])
        self.assertEqual(result.event_id.tolist(), ["001", "002", "003", "004"])
        self.assertEqual(result.predicted_final_tier.tolist(), ["Safe", "Safe", "Incomplete", "Incomplete"])
        np.testing.assert_allclose(result.prob_safe.iloc[:2], [.7, .7])
        self.assertTrue(result.prob_safe.iloc[2:].isna().all())
        for column in ("risk", "final_tier", "predicted_log10_pc", "estimated_pc"):
            self.assertNotIn(column, result)
        self.assertEqual(result.risk_tier.iloc[0], "Safe")
    def test_required_columns_are_checked(self):
        with self.assertRaises(ValueError):
            predict_messages(self.frame.drop(columns=["relative_speed"]), model=StubModel())

if __name__ == "__main__":
    unittest.main()
