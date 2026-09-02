from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    model: "BPRModel"
    runtime_seconds: float
    updates: int


@dataclass
class BPRModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    item_bias: np.ndarray

    def scores(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        return (
            self.item_factors[item_indices] @ self.user_factors[user_index]
            + self.item_bias[item_indices]
        )


def _sample_unrated_item(
    rng: np.random.Generator,
    n_items: int,
    rated_items: frozenset[int],
) -> int:
    while True:
        candidate = int(rng.integers(n_items))
        if candidate not in rated_items:
            return candidate


def train_bpr(
    train_positives: np.ndarray,
    rated_items_by_user: tuple[frozenset[int], ...],
    n_users: int,
    n_items: int,
    epochs: int,
    negatives_per_positive: int,
    seed: int,
    latent_dim: int,
    learning_rate: float,
    regularization: float,
    initialization_scale: float,
) -> TrainingResult:
    if epochs < 1 or negatives_per_positive < 1:
        raise ValueError("epochs and negatives_per_positive must both be positive")

    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0.0, initialization_scale, size=(n_users, latent_dim))
    item_factors = rng.normal(0.0, initialization_scale, size=(n_items, latent_dim))
    item_bias = np.zeros(n_items, dtype=np.float64)

    order = np.arange(len(train_positives), dtype=np.int64)
    updates = 0
    start = time.perf_counter()

    for _ in range(epochs):
        rng.shuffle(order)
        for row_index in order:
            user_index, positive_item = train_positives[row_index]
            user_index = int(user_index)
            positive_item = int(positive_item)
            rated_items = rated_items_by_user[user_index]

            for _ in range(negatives_per_positive):
                negative_item = _sample_unrated_item(rng, n_items, rated_items)

                user_vector = user_factors[user_index].copy()
                positive_vector = item_factors[positive_item].copy()
                negative_vector = item_factors[negative_item].copy()
                margin = (
                    float(user_vector @ (positive_vector - negative_vector))
                    + item_bias[positive_item]
                    - item_bias[negative_item]
                )
                gradient_weight = 1.0 / (1.0 + np.exp(np.clip(margin, -35.0, 35.0)))

                user_factors[user_index] += learning_rate * (
                    gradient_weight * (positive_vector - negative_vector)
                    - regularization * user_vector
                )
                item_factors[positive_item] += learning_rate * (
                    gradient_weight * user_vector - regularization * positive_vector
                )
                item_factors[negative_item] += learning_rate * (
                    -gradient_weight * user_vector - regularization * negative_vector
                )
                item_bias[positive_item] += learning_rate * (
                    gradient_weight - regularization * item_bias[positive_item]
                )
                item_bias[negative_item] += learning_rate * (
                    -gradient_weight - regularization * item_bias[negative_item]
                )
                updates += 1

    runtime_seconds = time.perf_counter() - start
    return TrainingResult(
        model=BPRModel(user_factors, item_factors, item_bias),
        runtime_seconds=runtime_seconds,
        updates=updates,
    )
