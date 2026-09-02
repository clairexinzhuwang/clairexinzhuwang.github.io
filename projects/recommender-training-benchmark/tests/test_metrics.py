from __future__ import annotations

import unittest

import numpy as np

from ranking_lab.metrics import cross_seed_topk_jaccard, evaluate_model
from ranking_lab.model import BPRModel


class MetricsTest(unittest.TestCase):
    def test_predictive_metrics_have_known_values(self) -> None:
        model = BPRModel(
            user_factors=np.asarray([[1.0], [1.0]]),
            item_factors=np.asarray([[3.0], [2.0], [1.0], [0.0]]),
            item_bias=np.zeros(4),
        )
        candidates = {
            0: np.asarray([0, 1, 2, 3]),
            1: np.asarray([2, 0, 1, 3]),
        }
        metrics, rankings = evaluate_model(model, candidates, k=2)

        self.assertAlmostEqual(metrics["heldout_pairwise_accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["hit_rate_at_2"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_2"], 0.5)
        self.assertEqual(rankings[0], frozenset({0, 1}))

    def test_cross_seed_jaccard_is_descriptive_overlap(self) -> None:
        rankings = {
            1: {0: frozenset({1, 2}), 1: frozenset({3, 4})},
            2: {0: frozenset({1, 2}), 1: frozenset({3, 5})},
        }
        result = cross_seed_topk_jaccard(rankings)
        self.assertAlmostEqual(result["top10_jaccard_mean"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result["top10_jaccard_min"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

