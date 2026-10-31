"""Ensure prepared telemetry is reused only within an unchanged session."""
from pathlib import Path
from unittest.mock import patch
import io
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

import src.validation as validation
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AppTest and (ROOT / "artifacts/test_predictions.csv").exists(), "Streamlit and artifacts required")
class SessionCacheTests(unittest.TestCase):
    def test_cache_reuse_invalidation_and_session_isolation(self):
        with patch("src.validation.validate_telemetry", wraps=validation.validate_telemetry) as validator:
            first = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(first.exception, [e.message for e in first.exception])
            self.assertEqual(validator.call_count, 1)
            prepared_key = first.session_state["prepared_telemetry"]["key"]
            next(x for x in first.slider if x.label == "Replay recorded messages").set_value(0).run()
            self.assertFalse(first.exception, [e.message for e in first.exception])
            self.assertEqual(validator.call_count, 1)
            self.assertEqual(first.session_state["prepared_telemetry"]["key"], prepared_key)
            next(x for x in first.text_input if x.label == "Find event ID").input("no-such-event").run()
            self.assertEqual(validator.call_count, 1)
            # Another browser session cannot receive the first session's
            # prepared dataframe, even though bundled CSV reads may be cached.
            second = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(second.exception, [e.message for e in second.exception])
            self.assertEqual(validator.call_count, 2)
            dataset = next(x for x in second.selectbox if x.label == "Dataset")
            historical = next(x for x in dataset.options if "Held-out" in x)
            dataset.set_value(historical).run()
            self.assertFalse(second.exception, [e.message for e in second.exception])
            self.assertEqual(validator.call_count, 3)
            holdout_key = second.session_state["prepared_telemetry"]["key"]
            self.assertNotEqual(holdout_key, prepared_key)
            # Model/input/code version changes cause a key mismatch. Simulate
            # a stale saved key without modifying any source file or artifact.
            second.session_state["prepared_telemetry"]["key"] = (holdout_key[0], ("stale-code-version",))
            second.run()
            self.assertFalse(second.exception, [e.message for e in second.exception])
            self.assertEqual(validator.call_count, 4)
            self.assertEqual(second.session_state["prepared_telemetry"]["key"], holdout_key)

    def test_reprocess_upload_with_original_file_cursor_at_eof(self):
        # Reusing the same stream at EOF simulates a model/code cache-key
        # invalidation after the first render. The app must parse fresh bytes.
        rows = pd.read_csv(ROOT / "artifacts/test_predictions.csv", nrows=3)
        rows["event_id"] = ["001", "001", "001"]
        rows["source_row_id"] = ["0001", "0002", "0003"]
        uploaded = io.BytesIO(rows.to_csv(index=False).encode("utf-8"))
        uploaded.seek(0, io.SEEK_END)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.session_state["tour_dataset"] = "Upload telemetry · estimate"
        with patch("streamlit.file_uploader", return_value=uploaded):
            app.run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(app.error, [e.value for e in app.error])
            self.assertEqual(app.session_state["prepared_telemetry"]["df"].source_row_id.tolist(), ["0001", "0002", "0003"])
            # Invalidate the prepared data without touching source or model
            # files; leave the original upload's file position at EOF.
            cached = app.session_state["prepared_telemetry"]
            cached["key"] = (cached["key"][0], ("stale-code-version",))
            uploaded.seek(0, io.SEEK_END)
            app.run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(app.error, [e.value for e in app.error])
            self.assertEqual(app.session_state["prepared_telemetry"]["df"].source_row_id.tolist(), ["0001", "0002", "0003"])


if __name__ == "__main__":
    unittest.main()
