from __future__ import annotations

import unittest

from ranking_lab.data import (
    Rating,
    build_evaluation_candidates,
    prepare_implicit_data,
)


class DataPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = [
            Rating(1, 10, 5.0, 1),
            Rating(1, 11, 3.0, 2),
            Rating(1, 12, 4.5, 3),
            Rating(2, 11, 4.0, 1),
            Rating(2, 13, 2.0, 2),
            Rating(2, 14, 5.0, 4),
            Rating(3, 10, 4.0, 1),
            Rating(3, 15, 2.5, 2),
        ]
        self.data = prepare_implicit_data(
            self.ratings,
            movie_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18],
            positive_rating_threshold=4.0,
        )

    def test_latest_qualifying_positive_is_held_out(self) -> None:
        self.assertEqual(self.data.user_ids, (1, 2))
        self.assertEqual(len(self.data.train_positives), 2)
        item_index = {movie_id: index for index, movie_id in enumerate(self.data.item_ids)}
        self.assertEqual(self.data.holdout_positive_by_user[0], item_index[12])
        self.assertEqual(self.data.holdout_positive_by_user[1], item_index[14])

    def test_evaluation_negatives_are_unrated(self) -> None:
        candidates = build_evaluation_candidates(self.data, negatives_per_user=2, seed=7)
        for user_index, values in candidates.items():
            self.assertEqual(values[0], self.data.holdout_positive_by_user[user_index])
            self.assertTrue(
                all(value not in self.data.rated_items_by_user[user_index] for value in values[1:])
            )


if __name__ == "__main__":
    unittest.main()

