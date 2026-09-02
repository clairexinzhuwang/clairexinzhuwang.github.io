from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


class AggregateOutputTest(unittest.TestCase):
    def test_csv_and_json_agree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        json_result = json.loads((root / "results" / "aggregate.json").read_text())
        with (root / "results" / "aggregate.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            csv_rows = {row["setting_id"]: row for row in csv.DictReader(handle)}

        self.assertEqual(len(csv_rows), 4)
        for setting in json_result["settings"]:
            csv_row = csv_rows[setting["setting_id"]]
            self.assertEqual(int(csv_row["updates"]), setting["updates"])
            self.assertAlmostEqual(
                float(csv_row["heldout_pairwise_accuracy_mean"]),
                setting["heldout_pairwise_accuracy"]["mean"],
            )
            self.assertAlmostEqual(
                float(csv_row["ndcg_at_10_mean"]),
                setting["ndcg_at_10"]["mean"],
            )
            self.assertGreaterEqual(float(csv_row["top10_jaccard_mean"]), 0.0)
            self.assertLessEqual(float(csv_row["top10_jaccard_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()
