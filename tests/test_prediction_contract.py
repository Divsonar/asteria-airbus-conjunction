"""Input isolation and row-output checks for the inference contract."""
import unittest
import numpy as np
import pandas as pd
from src.pipeline import FEATURES, predict_messages


class StubModel:
    def predict(self, features):
        assert list(features.columns) == FEATURES
        assert "risk" not in features.columns
        return np.array([-7.0, -3.0])


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "source_row_id": [19, 25], "event_id": [1, 2],
            "time_to_tca": [3.0, "not numeric"], "miss_distance": [1000.0, "bad"],
            "relative_speed": [8000.0, 7500.0],
            "relative_position_r": [10.0, 20.0], "relative_position_t": [100.0, 200.0],
            "relative_position_n": [100.0, 200.0], "c_object_type": ["DEBRIS", "UNKNOWN"],
            "c_sigma_r": [100.0, 100.0], "t_sigma_r": [10.0, 10.0],
            "max_risk_estimate": [-6.0, -5.0], "F10": [100.0, 100.0], "AP": [10.0, 10.0],
            "risk": [-30.0, -4.0],
        })

    def test_output_is_ordered_and_target_column_is_never_input(self):
        result = predict_messages(self.frame, model=StubModel())
        self.assertEqual(result.source_row_id.tolist(), [19, 25])
        self.assertEqual(result.event_id.tolist(), [1, 2])
        self.assertNotIn("risk", result.columns)
        self.assertEqual(result.risk_tier.tolist(), ["Elevated", "Critical"])
        self.assertAlmostEqual(result.estimated_pc.iloc[0], 1e-7)
        self.assertAlmostEqual(result.estimated_pc.iloc[1], 1e-3)
        self.assertTrue(pd.isna(result.time_to_tca.iloc[1]))
        self.assertTrue(pd.isna(result.miss_distance.iloc[1]))
        self.assertIn("Imputed model inputs", result.data_quality.iloc[1])

    def test_required_columns_are_checked(self):
        with self.assertRaises(ValueError):
            predict_messages(self.frame.drop(columns=["relative_speed"]), model=StubModel())


if __name__ == "__main__":
    unittest.main()
