"""Streamlit smoke checks on generated artifacts; skip until trained."""
from pathlib import Path
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AppTest and (ROOT / "artifacts/test_predictions.csv").exists(), "Streamlit and trained artifacts required")
class DashboardSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def test_navigation_and_evidence(self):
        self.assertGreater(len(self.app.dataframe), 0)
        self.app.radio[0].set_value("Model evidence").run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.app.radio[0].set_value("About the data").run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def test_upload_mode(self):
        dataset = next(x for x in self.app.selectbox if x.label == "Dataset")
        upload = next(x for x in dataset.options if "Upload telemetry" in x)
        dataset.set_value(upload).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assertTrue(any("Choose a telemetry CSV" in x.value for x in self.app.info))

    def test_event_selection_and_risk_filter(self):
        event = next(x for x in self.app.selectbox if x.label == "Event to inspect")
        if len(event.options) > 1:
            event.set_value(int(event.options[1].split()[-1])).run()
            self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        risk = next(x for x in self.app.multiselect if x.label == "Risk tier")
        risk.set_value(["Critical"]).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assertGreater(len(self.app.dataframe), 0)

    def test_search_and_no_match(self):
        next(x for x in self.app.text_input if x.label == "Find event ID").input("not-a-real-event").run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assertTrue(any("No events match" in x.value for x in self.app.info))

    def test_dataset_change(self):
        dataset = next(x for x in self.app.selectbox if x.label == "Dataset")
        options = dataset.options
        historical = [x for x in options if "Held-out" in x]
        if not historical:
            self.skipTest("Holdout predictions unavailable")
        dataset.set_value(historical[0]).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        self.assertGreater(len(self.app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()


