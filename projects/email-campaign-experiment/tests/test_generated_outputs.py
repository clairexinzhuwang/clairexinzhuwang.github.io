from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


class GeneratedOutputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (OUTPUTS / "experiment_summary.json").open(encoding="utf-8") as handle:
            cls.summary = json.load(handle)

    def test_dataset_and_analysis_contract(self) -> None:
        self.assertEqual(self.summary["data_quality"]["row_count"], 64_000)
        self.assertFalse(self.summary["dataset"]["raw_data_redistributed"])
        self.assertEqual(len(self.summary["primary_effects"]), 6)
        self.assertEqual(len(self.summary["secondary_effects"]), 3)
        self.assertEqual(
            len(self.summary["prespecified_segment"]["interaction_family"]), 2
        )

    def test_effect_intervals_and_adjustments(self) -> None:
        effects = self.summary["primary_effects"] + self.summary["secondary_effects"]
        for effect in effects:
            self.assertLessEqual(effect["ci_95_low"], effect["estimate"])
            self.assertGreaterEqual(effect["ci_95_high"], effect["estimate"])
            self.assertGreaterEqual(effect["p_value_holm"], effect["p_value"])

    def test_balance_diagnostic_is_small(self) -> None:
        self.assertLess(
            self.summary["balance"]["maximum_absolute_standardized_mean_difference"],
            0.1,
        )

    def test_csv_is_aggregate_and_has_nine_contrasts(self) -> None:
        with (OUTPUTS / "treatment_effects.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 9)
        self.assertNotIn("customer_id", rows[0])


if __name__ == "__main__":
    unittest.main()

