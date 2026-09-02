from __future__ import annotations

import unittest

import numpy as np

from ranking_lab.model import train_bpr


class BPRModelTest(unittest.TestCase):
    def test_training_is_reproducible_and_counts_updates(self) -> None:
        positives = np.asarray([[0, 0], [0, 1], [1, 2], [1, 3]], dtype=np.int64)
        rated = (frozenset({0, 1}), frozenset({2, 3}))
        kwargs = dict(
            train_positives=positives,
            rated_items_by_user=rated,
            n_users=2,
            n_items=6,
            epochs=8,
            negatives_per_positive=2,
            seed=13,
            latent_dim=4,
            learning_rate=0.04,
            regularization=0.002,
            initialization_scale=0.03,
        )
        first = train_bpr(**kwargs)
        second = train_bpr(**kwargs)

        self.assertEqual(first.updates, 4 * 8 * 2)
        np.testing.assert_allclose(first.model.user_factors, second.model.user_factors)
        np.testing.assert_allclose(first.model.item_factors, second.model.item_factors)
        np.testing.assert_allclose(first.model.item_bias, second.model.item_bias)
        self.assertTrue(np.isfinite(first.model.user_factors).all())


if __name__ == "__main__":
    unittest.main()

