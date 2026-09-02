from __future__ import annotations

import math
import unittest

from src.analysis import (
    build_newbie_segment,
    difference_in_means,
    holm_adjust,
    standardized_mean_difference,
)


class StatisticalUtilitiesTest(unittest.TestCase):
    def test_difference_in_means_and_interval(self) -> None:
        result = difference_in_means([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        self.assertAlmostEqual(result["estimate"], 1.0)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(2.0 / 3.0))
        self.assertLess(result["ci_95_low"], result["estimate"])
        self.assertGreater(result["ci_95_high"], result["estimate"])
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_holm_adjustment_restores_original_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertTrue(all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.03])))

    def test_standardized_mean_difference(self) -> None:
        self.assertAlmostEqual(
            standardized_mean_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0
        )
        forward = standardized_mean_difference([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        reverse = standardized_mean_difference([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertAlmostEqual(forward, -reverse)

    def test_segment_inference_tests_interactions(self) -> None:
        rows = []
        patterns = {
            "No E-Mail": {0: [0, 0, 1, 1], 1: [0, 0, 0, 1]},
            "Mens E-Mail": {0: [0, 1, 1, 1], 1: [0, 1, 1, 1]},
            "Womens E-Mail": {0: [0, 0, 1, 1], 1: [0, 0, 1, 1]},
        }
        for arm, strata in patterns.items():
            for newbie, visits in strata.items():
                rows.extend(
                    {"segment": arm, "newbie": newbie, "visit": visit} for visit in visits
                )

        stratum_effects, interactions = build_newbie_segment(rows)
        self.assertEqual(len(stratum_effects), 4)
        self.assertEqual(len(interactions), 2)
        self.assertTrue(
            all(record["p_value_holm"] >= record["p_value"] for record in interactions)
        )


if __name__ == "__main__":
    unittest.main()

