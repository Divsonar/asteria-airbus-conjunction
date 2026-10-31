"""Boundary and missing-data checks for the Airbus challenge rules."""
import math
import unittest

from src.risk import classify, classify_with_rule_ids


class AirbusTierTests(unittest.TestCase):
    def tier(self, risk, distance):
        return classify([risk], [distance])[0][0]

    def test_threshold_boundaries_and_critical_precedence(self):
        cases = [
            (-4, 5001, "Critical"),
            (-4.0001, 5001, "Elevated"),
            (-6, 5001, "Elevated"),
            (-6.0001, 5001, "Safe"),
            (-7, 999.999, "Critical"),
            (-7, 1000, "Elevated"),
            (-7, 5000, "Elevated"),
            (-7, 5000.001, "Safe"),
            (-3, 2000, "Critical"),
        ]
        for risk, distance, expected in cases:
            with self.subTest(risk=risk, distance=distance):
                self.assertEqual(self.tier(risk, distance), expected)

    def test_missing_and_invalid_inputs_fail_closed(self):
        for risk, distance, expected in [
            (math.nan, 6000, "Incomplete"),
            (-7, math.nan, "Incomplete"),
            (math.nan, 999, "Critical"),
            (-3, math.nan, "Critical"),
            (1, 6000, "Incomplete"),
            (-7, -1, "Incomplete"),
            (math.inf, 6000, "Incomplete"),
        ]:
            with self.subTest(risk=risk, distance=distance):
                self.assertEqual(self.tier(risk, distance), expected)

    def test_rule_ids_preserve_precedence_and_boundary_semantics(self):
        risks = [-4, -7, -5, -7, -7, math.nan, -3]
        distances = [5001, 999, 1000, 5000, 5001, 6000, math.nan]
        tiers, rules, quality, ids = classify_with_rule_ids(risks, distances)
        self.assertEqual(ids, ["C1", "C2", "E1; E2", "E2", "S1", "", "C1"])
        self.assertEqual((tiers, rules, quality), classify(risks, distances))
        self.assertEqual(tiers[-1], "Critical")
        self.assertIn("missing miss distance", quality[-1])


if __name__ == "__main__":
    unittest.main()
