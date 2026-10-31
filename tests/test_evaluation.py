"""Risk-focused evaluation must not hide misses behind distance rules."""
import unittest

import numpy as np
import pandas as pd

from experiments.legacy_pipeline import (bucket_metrics, choose_weighted_candidate, distance_only_metrics,
                          latest_message, metrics, pc_driven_recall_interval)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({
            "event_id": [1, 1, 2, 3, 4, 5, 6],
            "source_row_id": range(7),
            "time_to_tca": [3.5, 2.5, 2.3, 1.2, 0.2, 3.2, -0.2],
            "risk": [-3.0, -3.0, -7.0, -7.0, -5.0, -3.0, np.nan],
            "miss_distance": [2000.0, 3000.0, 2000.0, 3000.0, 500.0, 5000.0, 2000.0],
        })
        self.predictions = np.array([-3.0, -7.0, -3.0, -7.0, -10.0, -4.0, np.nan])

    def test_pc_driven_metrics_exclude_distance_forced_and_invalid(self):
        result = metrics(self.rows, self.predictions)
        self.assertEqual((result["pc_driven_critical_tp"], result["pc_driven_critical_fn"],
                          result["pc_driven_critical_fp"]), (2, 1, 1))
        self.assertAlmostEqual(result["pc_driven_critical_recall"], 2 / 3)
        self.assertAlmostEqual(result["pc_driven_critical_precision"], 2 / 3)
        self.assertEqual(result["pc_driven_critical_events"], 2)
        self.assertEqual(result["n_risk_ge_minus6"], 4)
        self.assertAlmostEqual(result["mae_log10_risk_ge_minus6"], 2.5)

    def test_distance_baseline_and_latest_event_selection(self):
        baseline = distance_only_metrics(self.rows)
        self.assertNotIn("mae_log10", baseline)
        self.assertEqual(baseline["pc_driven_critical_tp"], 0)
        # Event 1's closest nonnegative message is its second message.
        latest = latest_message(self.rows)
        self.assertEqual(latest.set_index("event_id").loc[1, "source_row_id"], 1)
        self.assertNotIn(6, latest.event_id.tolist())
        buckets = bucket_metrics(self.rows, self.predictions)
        self.assertEqual([x["start_days"] for x in buckets], [0, 1, 2, 3])
        self.assertEqual(sum(x["model"]["n_rows"] for x in buckets), 6)

    def test_nonfinite_prediction_is_a_miss_not_a_silent_exclusion(self):
        subset = self.rows.iloc[[0, 1]]
        result = metrics(subset, np.array([-3.0, np.nan]))
        self.assertEqual(result["pc_driven_critical_true"], 2)
        self.assertEqual(result["pc_driven_critical_fn"], 1)
        self.assertEqual(result["risk_only_critical_true"], 2)
        self.assertEqual(result["risk_only_critical_fn"], 1)
        self.assertEqual(result["pc_driven_critical_recall"], .5)
        self.assertEqual(result["pc_driven_critical_unscorable"], 1)
        self.assertEqual(result["risk_only_critical_unscorable"], 1)
        self.assertEqual(result["pc_driven_predicted_critical"], 1)
        self.assertEqual(result["n_valid_truth_rows"], 2)
        self.assertEqual(result["n_scored_rows"], 1)
        self.assertEqual(result["n_unscorable_predictions_valid_truth"], 1)
        self.assertEqual(result["n_risk_ge_minus6_truth"], 2)
        self.assertEqual(result["n_risk_ge_minus6"], 1)
        self.assertEqual(result["n_risk_ge_minus6_unscorable"], 1)

    def test_weighted_candidate_must_meet_all_guardrails(self):
        original = {"pc_driven_critical_recall": .5, "pc_driven_critical_precision": .8,
                    "mae_log10": 4.0, "mae_log10_risk_ge_minus6": 2.0}
        better = {"pc_driven_critical_recall": .7, "pc_driven_critical_precision": .76,
                  "mae_log10": 4.9, "mae_log10_risk_ge_minus6": 1.5}
        self.assertTrue(choose_weighted_candidate(original, better))
        for key, value in [("pc_driven_critical_precision", .7), ("mae_log10", 5.1),
                           ("mae_log10_risk_ge_minus6", 2.1), ("pc_driven_critical_recall", .4)]:
            bad = {**better, key: value}
            self.assertFalse(choose_weighted_candidate(original, bad), key)

    def test_cluster_bootstrap_is_deterministic_and_handles_rare_events(self):
        interval = pc_driven_recall_interval(self.rows, self.predictions, iterations=100, seed=9)
        self.assertEqual(interval, pc_driven_recall_interval(self.rows, self.predictions, iterations=100, seed=9))
        self.assertEqual(interval["positive_events"], 2)
        self.assertGreater(interval["valid_resamples"], 0)
        self.assertLessEqual(interval["lower"], interval["upper"])
        self.assertGreaterEqual(interval["lower"], 0)
        self.assertLessEqual(interval["upper"], 1)
        one_event = self.rows[self.rows.event_id == 5]
        self.assertIsNone(pc_driven_recall_interval(one_event, self.predictions[[5]], iterations=100))


if __name__ == "__main__":
    unittest.main()
