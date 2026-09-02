from __future__ import annotations

from itertools import combinations

import numpy as np

from ranking_lab.model import BPRModel


def evaluate_model(
    model: BPRModel,
    candidates_by_user: dict[int, np.ndarray],
    k: int = 10,
) -> tuple[dict[str, float], dict[int, frozenset[int]]]:
    pairwise_scores: list[float] = []
    hit_rates: list[float] = []
    discounted_gains: list[float] = []
    top_k_by_user: dict[int, frozenset[int]] = {}

    for user_index, candidates in candidates_by_user.items():
        scores = model.scores(user_index, candidates)
        positive_score = scores[0]
        negative_scores = scores[1:]
        pairwise_scores.append(float(np.mean(positive_score > negative_scores)))

        rank = 1 + int(np.sum(negative_scores > positive_score))
        hit_rates.append(float(rank <= k))
        discounted_gains.append(1.0 / np.log2(rank + 1.0) if rank <= k else 0.0)

        top_count = min(k, len(candidates))
        top_positions = np.argsort(-scores, kind="stable")[:top_count]
        top_k_by_user[user_index] = frozenset(int(candidates[p]) for p in top_positions)

    metrics = {
        "heldout_pairwise_accuracy": float(np.mean(pairwise_scores)),
        f"hit_rate_at_{k}": float(np.mean(hit_rates)),
        f"ndcg_at_{k}": float(np.mean(discounted_gains)),
    }
    return metrics, top_k_by_user


def cross_seed_topk_jaccard(
    rankings_by_seed: dict[int, dict[int, frozenset[int]]],
) -> dict[str, float]:
    agreements: list[float] = []
    for seed_a, seed_b in combinations(sorted(rankings_by_seed), 2):
        rankings_a = rankings_by_seed[seed_a]
        rankings_b = rankings_by_seed[seed_b]
        for user_index in sorted(rankings_a):
            set_a = rankings_a[user_index]
            set_b = rankings_b[user_index]
            union = set_a | set_b
            agreements.append(len(set_a & set_b) / len(union) if union else 1.0)

    if not agreements:
        return {
            "top10_jaccard_mean": 1.0,
            "top10_jaccard_min": 1.0,
            "top10_jaccard_max": 1.0,
        }
    return {
        "top10_jaccard_mean": float(np.mean(agreements)),
        "top10_jaccard_min": float(np.min(agreements)),
        "top10_jaccard_max": float(np.max(agreements)),
    }
