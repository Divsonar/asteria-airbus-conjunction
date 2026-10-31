"""Exercise actual fitted-model inference on a small synthetic upload."""
from io import StringIO
import csv
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.pipeline import KEYS, predict_messages
from src.validation import read_telemetry_csv, spreadsheet_safe_csv, validate_telemetry

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "artifacts/risk_model.joblib").exists() and
                     (ROOT / "artifacts/test_predictions.csv").exists(), "Trained artifacts required")
class UploadIntegrationTests(unittest.TestCase):
    def test_literal_ids_quality_flags_and_both_export_modes(self):
        # Reuse only permitted features; the labels below are fabricated, not
        # loaded from or compared with the organiser's locked test labels.
        sample = pd.read_csv(ROOT / "artifacts/test_predictions.csv", usecols=KEYS, nrows=3)
        sample["event_id"] = ["001", "=formula", "normal"]
        sample.insert(0, "source_row_id", ["0001", "@formula", "n-003"])
        sample.loc[1, "miss_distance"] = -5
        sample["relative_speed"] = sample["relative_speed"].astype(object)
        sample.loc[2, "relative_speed"] = "bad"
        sample["risk"] = [-3, -4, -7]  # synthetic extra column, never a feature
        parsed = read_telemetry_csv(StringIO(sample.to_csv(index=False)))
        check = validate_telemetry(parsed)
        self.assertEqual(parsed.event_id.tolist(), ["001", "=formula", "normal"])
        self.assertEqual(parsed.source_row_id.tolist(), ["0001", "@formula", "n-003"])
        self.assertIn("R-RANGE", check.flags.review_codes.iloc[1])
        self.assertIn("R-NUMERIC", check.flags.review_codes.iloc[2])

        result = predict_messages(parsed)
        altered = parsed.copy()
        altered["risk"] = [999, -999, 0]
        again = predict_messages(altered)
        self.assertEqual(len(result), 3)
        self.assertNotIn("risk", result.columns)
        self.assertEqual(result.source_row_id.tolist(), ["0001", "@formula", "n-003"])
        np.testing.assert_allclose(result[["prob_critical", "prob_elevated", "prob_safe"]], again[["prob_critical", "prob_elevated", "prob_safe"]])

        # The safe spreadsheet export neutralizes user-controlled formulas;
        # the exact export retains the original IDs for machine scoring.
        exact = list(csv.DictReader(StringIO(result.to_csv(index=False))))
        safe = list(csv.DictReader(StringIO(spreadsheet_safe_csv(result).decode("utf-8"))))
        self.assertEqual(exact[1]["event_id"], "=formula")
        self.assertEqual(exact[1]["source_row_id"], "@formula")
        self.assertEqual(safe[1]["event_id"], "'=formula")
        self.assertEqual(safe[1]["source_row_id"], "'@formula")
        self.assertEqual(exact[0]["source_row_id"], "0001")
        self.assertEqual(safe[0]["source_row_id"], "0001")
        self.assertEqual(exact[1]["miss_distance"], safe[1]["miss_distance"])


if __name__ == "__main__":
    unittest.main()
