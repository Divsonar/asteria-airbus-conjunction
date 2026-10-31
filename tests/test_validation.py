"""Validation flags are advisory, preserve rows and expose actionable issues."""
import unittest
import csv
from io import StringIO

import numpy as np
import pandas as pd

from src.risk import classify_with_rule_ids
from src.validation import REQUIRED_COLUMNS, read_telemetry_csv, spreadsheet_safe_csv, validate_telemetry


def valid_frame():
    return pd.DataFrame({
        "source_row_id": [42, 19, 23], "event_id": [10, 11, 12],
        "time_to_tca": [3., 3., 3.], "miss_distance": [1000., 2000., 3000.],
        "relative_speed": [8000., 8000., 8000.],
        "relative_position_r": [1000., 2000., 3000.],
        "relative_position_t": [0., 0., 0.], "relative_position_n": [0., 0., 0.],
        "c_object_type": ["DEBRIS", "PAYLOAD", "UNKNOWN"],
        "c_sigma_r": [100., 100., 100.], "t_sigma_r": [10., 10., 10.],
        "max_risk_estimate": [-6., -5., -7.], "F10": [90., np.nan, 90.],
        "AP": [10., np.nan, 10.],
    }, index=[7, 7, 2])


class TelemetryValidationTests(unittest.TestCase):
    def test_csv_reader_preserves_identifiers_and_allows_missing_schema(self):
        frame = read_telemetry_csv(StringIO('event_id,source_row_id,AP\n001,0002,\nNA,NULL,NA\n'))
        self.assertEqual(frame.event_id.tolist(), ["001", "NA"])
        self.assertEqual(frame.source_row_id.tolist(), ["0002", "NULL"])
        self.assertTrue(frame.AP.isna().all())
        missing = read_telemetry_csv(StringIO('miss_distance\n2\n'))
        self.assertIn('event_id', validate_telemetry(missing).summary['missing_columns'])

    def test_csv_reader_excludes_risk_and_extra_columns(self):
        frame = read_telemetry_csv(StringIO('event_id,source_row_id,AP,risk,other\n001,0002,10,-3,private\n'))
        self.assertEqual(frame.columns.tolist(), ['event_id', 'source_row_id', 'AP'])
        self.assertEqual(frame.event_id.iloc[0], '001')
        with self.assertRaisesRegex(ValueError, 'Missing required columns'):
            read_telemetry_csv(StringIO('unrelated\nvalue\n'))

    def test_empty_csv_with_headers_is_validated_without_a_dtype_crash(self):
        source = StringIO(",".join(["source_row_id", *REQUIRED_COLUMNS]) + "\n")
        result = validate_telemetry(read_telemetry_csv(source))
        self.assertEqual(result.summary["rows"], 0)
        self.assertEqual(result.summary["missing_columns"], [])
        self.assertTrue(result.flags.empty)

    def test_spreadsheet_export_neutralizes_formulas_without_mutating_data(self):
        frame = pd.DataFrame({"event_id": ["=1+1", "  @bad", "001", "\tcmd"],
                              "source_row_id": ["+2", "-2", "text", "0001"],
                              "miss_distance": [-5.0, 1.0, 2.0, 3.0]})
        original = frame.copy(deep=True)
        result = list(csv.DictReader(StringIO(spreadsheet_safe_csv(frame).decode("utf-8"))))
        self.assertEqual([r["event_id"] for r in result], ["'=1+1", "'  @bad", "001", "'\tcmd"])
        self.assertEqual([r["source_row_id"] for r in result], ["'+2", "'-2", "text", "0001"])
        self.assertEqual(result[0]["miss_distance"], "-5.0")
        pd.testing.assert_frame_equal(frame, original)

    def test_valid_and_optional_fields_preserve_identity(self):
        frame = valid_frame()
        original = frame.copy(deep=True)
        result = validate_telemetry(frame)
        self.assertEqual(result.flags.index.tolist(), [7, 7, 2])
        self.assertEqual(result.flags.source_row_id.tolist(), [42, 19, 23])
        self.assertFalse(result.flags.review_required.any())
        self.assertEqual(result.summary["missing_counts"]["F10"], 1)
        self.assertEqual(result.summary["missing_counts"]["AP"], 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_and_suspicious_values_report_without_mutation(self):
        frame = valid_frame()
        frame["relative_speed"] = frame["relative_speed"].astype(object)
        frame.iloc[0, frame.columns.get_loc("miss_distance")] = -10.
        frame.iloc[1, frame.columns.get_loc("relative_speed")] = "bad"
        frame.iloc[1, frame.columns.get_loc("c_sigma_r")] = 60_000_000.
        frame.iloc[2, frame.columns.get_loc("relative_position_r")] = 2000.
        frame.iloc[2, frame.columns.get_loc("c_object_type")] = "UNKNOWN TYPE"
        result = validate_telemetry(frame)
        self.assertIn("R-RANGE", result.flags.review_codes.iloc[0])
        self.assertIn("R-NUMERIC", result.flags.review_codes.iloc[1])
        self.assertIn("R-SIGMA", result.flags.review_codes.iloc[1])
        self.assertIn("screening heuristic", result.flags.review_reasons.iloc[1])
        self.assertIn("R-RTN", result.flags.review_codes.iloc[2])
        self.assertIn("R-OBJECT", result.flags.review_codes.iloc[2])
        self.assertEqual(result.summary["invalid_numeric_counts"]["relative_speed"], 1)
        self.assertEqual(result.summary["review_rows"], 3)
        self.assertEqual(result.flags.source_row_id.tolist(), [42, 19, 23])

    def test_schema_missing_infinite_and_past_messages(self):
        frame = valid_frame().drop(columns=["max_risk_estimate"])
        frame.iloc[0, frame.columns.get_loc("time_to_tca")] = -1
        frame.iloc[1, frame.columns.get_loc("miss_distance")] = np.inf
        result = validate_telemetry(frame)
        self.assertEqual(result.summary["missing_columns"], ["max_risk_estimate"])
        self.assertEqual(result.summary["missing_counts"]["max_risk_estimate"], 3)
        self.assertEqual(result.summary["invalid_numeric_counts"]["miss_distance"], 1)
        self.assertIn("R-AFTER-TCA", result.flags.review_codes.iloc[0])
        self.assertIn("R-NUMERIC", result.flags.review_codes.iloc[1])
        self.assertTrue(result.flags.review_required.all())

    def test_duplicate_source_identifiers_are_flagged_without_discarding_rows(self):
        frame = valid_frame()
        frame["source_row_id"] = [42, 42, 23]
        result = validate_telemetry(frame)
        self.assertIn("R-ROW-ID", result.flags.review_codes.iloc[0])
        self.assertIn("R-ROW-ID", result.flags.review_codes.iloc[1])
        self.assertNotIn("R-ROW-ID", result.flags.review_codes.iloc[2])
        self.assertEqual(result.flags.source_row_id.tolist(), [42, 42, 23])

    def test_category_format_that_model_would_not_recognize_is_flagged(self):
        frame = valid_frame()
        frame.iloc[0, frame.columns.get_loc("c_object_type")] = "debris"
        result = validate_telemetry(frame)
        self.assertIn("R-OBJECT", result.flags.review_codes.iloc[0])

    def test_source_identifiers_accept_alphanumeric_and_preserve_leading_zeros(self):
        frame = valid_frame()
        frame["source_row_id"] = ["001", "A-2", "1"]
        flags = validate_telemetry(frame).flags
        self.assertEqual(flags.source_row_id.tolist(), ["001", "A-2", "1"])
        self.assertFalse(flags.review_required.any())
        frame["source_row_id"] = ["001", "001", ""]
        flags = validate_telemetry(frame).flags
        self.assertTrue(all("R-ROW-ID" in code for code in flags.review_codes))
        for invalid in (True, float("inf"), " " * 3, "x" * 257):
            frame = valid_frame()
            frame["source_row_id"] = frame["source_row_id"].astype(object)
            frame.iloc[0, frame.columns.get_loc("source_row_id")] = invalid
            with self.subTest(value=invalid):
                self.assertIn("R-ROW-ID", validate_telemetry(frame).flags.review_codes.iloc[0])

    def test_event_identifiers_follow_the_review_log_contract(self):
        frame = valid_frame()
        frame["event_id"] = frame["event_id"].astype(object)
        frame.iloc[0, frame.columns.get_loc("event_id")] = "e" * 129
        frame.iloc[1, frame.columns.get_loc("event_id")] = float("inf")
        flags = validate_telemetry(frame).flags
        self.assertIn("R-ID", flags.review_codes.iloc[0])
        self.assertIn("R-ID", flags.review_codes.iloc[1])
        self.assertNotIn("R-ID", flags.review_codes.iloc[2])

    def test_very_long_tca_is_flagged_but_value_is_preserved(self):
        frame = valid_frame()
        frame.iloc[0, frame.columns.get_loc("time_to_tca")] = 1e300
        frame.iloc[1, frame.columns.get_loc("time_to_tca")] = 365
        result = validate_telemetry(frame)
        self.assertIn("R-TCA", result.flags.review_codes.iloc[0])
        self.assertNotIn("R-TCA", result.flags.review_codes.iloc[1])
        self.assertEqual(frame.time_to_tca.iloc[0], 1e300)

    def test_review_flag_does_not_replace_a_critical_distance_rule(self):
        frame = valid_frame()
        frame.iloc[0, frame.columns.get_loc("miss_distance")] = 900.
        frame.iloc[0, frame.columns.get_loc("relative_position_r")] = 900.
        frame.iloc[0, frame.columns.get_loc("c_sigma_r")] = np.nan
        review = validate_telemetry(frame).flags
        tiers, _, _, ids = classify_with_rule_ids([-10.] * len(frame), frame["miss_distance"])
        self.assertTrue(review.review_required.iloc[0])
        self.assertIn("R-MISSING", review.review_codes.iloc[0])
        self.assertEqual(tiers[0], "Critical")
        self.assertEqual(ids[0], "C2")


if __name__ == "__main__":
    unittest.main()
