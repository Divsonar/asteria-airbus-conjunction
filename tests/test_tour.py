"""Backend contracts for the in-context guided tour.

The tour UI runs inside a browser component, so its spotlight positioning,
navigation buttons, and browser storage need browser-level verification.
"""
import json
from pathlib import Path
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AppTest and (ROOT / "artifacts/test_predictions.csv").exists(), "Streamlit and trained artifacts required")
class GuidedTourTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def component_args(self):
        component = next(e for e in self.app.get("component_instance") if e.key == "asteria_first_visit")
        return json.loads(component.proto.json_args)

    def launch(self):
        next(e for e in self.app.button if e.key == "tour_launch").click().run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def test_replay_signal_survives_reruns(self):
        self.assertEqual(self.component_args()["launch"], 0)
        self.launch()
        self.assertEqual(self.component_args()["launch"], 1)
        self.app.run()
        self.assertEqual(self.component_args()["launch"], 1)
        self.launch()
        self.assertEqual(self.component_args()["launch"], 2)

    def test_replay_is_available_from_other_pages_and_empty_results(self):
        self.app.radio[0].set_value("Model evidence").run()
        self.launch()
        self.assertEqual(self.component_args()["launch"], 1)
        self.app.radio[0].set_value("Mission control").run()
        next(x for x in self.app.text_input if x.label == "Find event ID").input("event-does-not-exist").run()
        self.assertTrue(any("No events match" in e.value for e in self.app.info))
        self.launch()
        self.assertEqual(self.component_args()["launch"], 2)

    def test_tour_targets_are_rendered_on_their_pages(self):
        mission_keys = {getattr(e, "key", None) for e in self.app}
        self.assertTrue({"tour_navigation", "tour_dataset", "tour_risk_filter", "tour_event_select", "tour_geometry", "tour_history"}.issubset(mission_keys))
        self.assertGreater(len(self.app.dataframe), 0)
        self.app.radio[0].set_value("Model evidence").run()
        evidence_keys = {getattr(e, "key", None) for e in self.app}
        self.assertTrue({"tour_navigation", "tour_error_chart", "tour_confusion_chart"}.issubset(evidence_keys))
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])


if __name__ == "__main__":
    unittest.main()
