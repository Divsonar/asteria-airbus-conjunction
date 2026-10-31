"""Read-only integration checks for replay, filters and bundle attribution."""
from pathlib import Path
from unittest.mock import patch
import io
import unittest

import numpy as np
import pandas as pd

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
TEST_PREDICTIONS = ROOT / "artifacts" / "test_predictions.csv"


@unittest.skipUnless(AppTest and TEST_PREDICTIONS.exists(), "Streamlit and trained artifacts required")
class DashboardWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_rows = pd.read_csv(TEST_PREDICTIONS)

    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def event_widget(self):
        return next(x for x in self.app.selectbox if x.label == "Event to inspect")

    def replay_widget(self):
        return next(x for x in self.app.slider if x.label == "Replay recorded messages")

    def event_messages(self, event_id):
        rows = self.test_rows.reset_index(names="position")
        return rows[(rows.event_id == event_id) & (rows.time_to_tca >= 0)].sort_values(
            ["time_to_tca", "position"], ascending=[False, False], kind="stable")

    def assert_selected_row(self, row):
        expected = "MESSAGE #" + str(row.source_row_id)
        self.assertTrue(any(expected in item.value for item in self.app.markdown), expected)
        self.assertTrue(any("input position " + str(row.position) in item.value for item in self.app.caption))
        self.assertTrue(any("event assessment PDF" in item.proto.label for item in self.app.get("download_button")))

    def test_replay_starts_at_latest_changes_evidence_and_resets_for_new_event(self):
        widget = self.event_widget()
        rows = self.event_messages(widget.value)
        self.assertGreater(len(rows), 1)
        self.assertEqual(self.replay_widget().value, len(rows) - 1)
        self.assert_selected_row(rows.iloc[-1])
        self.replay_widget().set_value(0).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assert_selected_row(rows.iloc[0])
        next_id = int(widget.options[1].split()[-1])
        widget.set_value(next_id).run()
        next_rows = self.event_messages(next_id)
        if len(next_rows) > 1:
            self.assertEqual(self.replay_widget().value, len(next_rows) - 1)
        self.assert_selected_row(next_rows.iloc[-1])

    def test_returning_to_dashboard_restores_latest_replay_message(self):
        event_id = self.event_widget().value
        rows = self.event_messages(event_id)
        self.replay_widget().set_value(0).run()
        self.app.radio[0].set_value("Model evidence").run()
        self.app.radio[0].set_value("Mission control").run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assertEqual(self.replay_widget().value, len(rows) - 1)
        self.assert_selected_row(rows.iloc[-1])

    def test_filtered_buckets_reconcile_with_latest_event_messages(self):
        next(x for x in self.app.multiselect if x.label == "Risk tier").set_value(["Critical"]).run()
        next(x for x in self.app.selectbox if x.label == "Object type").set_value("DEBRIS").run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        data = self.test_rows.reset_index(names="position")
        latest = data[data.time_to_tca >= 0].sort_values(["event_id", "time_to_tca", "position"], kind="stable").drop_duplicates("event_id")
        expected = latest[(latest.risk_tier == "Critical") & (latest.c_object_type == "DEBRIS") & latest.time_to_tca.between(0, 7)]
        queue = self.app.dataframe[0].value
        buckets = self.app.dataframe[1].value
        self.assertEqual(len(queue), len(expected))
        self.assertEqual(int(buckets["Events"].sum()), len(expected))
        self.assertEqual(int(buckets["Under 1 km"].sum()), int(expected.miss_distance.ge(0).mul(expected.miss_distance.lt(1000)).sum()))
        self.assertEqual(int(buckets["1–5 km"].sum()), int(expected.miss_distance.ge(1000).mul(expected.miss_distance.le(5000)).sum()))

    def test_no_matching_events_still_offers_prediction_download(self):
        next(x for x in self.app.text_input if x.label == "Find event ID").input("no-such-event").run()
        self.assertTrue(any("No events match" in x.value for x in self.app.info))
        self.assertTrue(any("Download" in x.proto.label and "predictions" in x.proto.label
                            for x in self.app.get("download_button")))

    def test_bundle_mismatch_does_not_attribute_model_to_the_displayed_results(self):
        mismatch = {"status": "mismatch", "file_status": {}, "model_sha256": "a" * 64,
                    "details": "Artifact contents changed; lineage is unverified."}
        with patch("src.provenance.verify_manifest", return_value=mismatch):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any("integrity is not verified" in x.value for x in app.warning))
        self.assertTrue(any("Model: Not verified" in x.value for x in app.caption))
        self.assertFalse(any("a" * 64 in x.value for x in app.caption))

    def test_negative_distance_is_not_counted_as_a_close_approach(self):
        # Synthetic in-memory data exercises the rendered KPI; no artifacts are
        # modified. The first row is invalid and the second is a real <1 km case.
        import streamlit as st
        synthetic = self.test_rows.head(2).copy()
        synthetic["event_id"] = [909990, 909991]
        synthetic["source_row_id"] = [123450, 123451]
        synthetic["time_to_tca"] = [.25, .50]
        synthetic["miss_distance"] = [-1., 500.]
        synthetic["risk_tier"] = ["Incomplete", "Critical"]
        original_read_csv = pd.read_csv
        def read_csv(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).name == "test_predictions.csv":
                return synthetic.copy()
            return original_read_csv(path, *args, **kwargs)
        st.cache_data.clear()
        try:
            with patch("pandas.read_csv", side_effect=read_csv):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            card = next(x.value for x in app.markdown if "24h · under 1 km" in x.value)
            self.assertIn('<div class="metric-value">1</div>', card)
            buckets = app.dataframe[1].value
            self.assertEqual(int(buckets["Events"].sum()), 2)
            self.assertEqual(int(buckets["Under 1 km"].sum()), 1)
        finally:
            st.cache_data.clear()

    def test_extreme_time_is_flagged_retained_and_grouped_without_overflow(self):
        sample = self.test_rows.head(2).copy()
        sample["event_id"] = ["E1", "E2"]
        sample["source_row_id"] = ["001", "002"]
        sample["time_to_tca"] = [1e300, 5.]
        upload = io.BytesIO(sample.to_csv(index=False).encode())
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.session_state["tour_dataset"] = "Upload telemetry · estimate"
        with patch("streamlit.file_uploader", return_value=upload):
            app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        queued = app.dataframe[0].value
        buckets = app.dataframe[1].value
        self.assertEqual(len(queued), 2)
        self.assertTrue(bool(queued.loc[queued["Event"] == "E1", "Review"].iloc[0]))
        self.assertIn("365+ days", buckets["Time to TCA"].tolist())
        self.assertEqual(int(buckets["Events"].sum()), 2)
        self.assertEqual(len(app.session_state["prepared_telemetry"]["df"]), 2)
        self.assertTrue(any("exact predictions" in x.proto.label for x in app.get("download_button")))


if __name__ == "__main__":
    unittest.main()
