from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ranking_lab.data import (
    build_evaluation_candidates,
    download_and_extract,
    load_movie_ids,
    load_ratings,
    prepare_implicit_data,
)
from ranking_lab.metrics import cross_seed_topk_jaccard, evaluate_model
from ranking_lab.model import train_bpr
from ranking_lab.sql_audit import run_sql_audit


TITLE = "Training-Budget Trade-offs in a Standard Recommender"
CSV_FIELDS = [
    "setting_id",
    "epochs",
    "negatives_per_positive",
    "updates",
    "optimizer_seeds",
    "heldout_pairwise_accuracy_mean",
    "heldout_pairwise_accuracy_seed_sd",
    "heldout_pairwise_accuracy_seed_range",
    "hit_rate_at_10_mean",
    "hit_rate_at_10_seed_sd",
    "hit_rate_at_10_seed_range",
    "ndcg_at_10_mean",
    "ndcg_at_10_seed_sd",
    "ndcg_at_10_seed_range",
    "runtime_seconds_mean",
    "runtime_seconds_seed_sd",
    "top10_jaccard_mean",
    "top10_jaccard_min",
]


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": _rounded(np.mean(array)),
        "seed_sd": _rounded(np.std(array, ddof=0)),
        "seed_range": _rounded(np.max(array) - np.min(array)),
        "min": _rounded(np.min(array)),
        "max": _rounded(np.max(array)),
    }


def _flatten_for_csv(setting: dict[str, Any]) -> dict[str, Any]:
    return {
        "setting_id": setting["setting_id"],
        "epochs": setting["epochs"],
        "negatives_per_positive": setting["negatives_per_positive"],
        "updates": setting["updates"],
        "optimizer_seeds": setting["optimizer_seeds"],
        "heldout_pairwise_accuracy_mean": setting["heldout_pairwise_accuracy"]["mean"],
        "heldout_pairwise_accuracy_seed_sd": setting["heldout_pairwise_accuracy"]["seed_sd"],
        "heldout_pairwise_accuracy_seed_range": setting["heldout_pairwise_accuracy"]["seed_range"],
        "hit_rate_at_10_mean": setting["hit_rate_at_10"]["mean"],
        "hit_rate_at_10_seed_sd": setting["hit_rate_at_10"]["seed_sd"],
        "hit_rate_at_10_seed_range": setting["hit_rate_at_10"]["seed_range"],
        "ndcg_at_10_mean": setting["ndcg_at_10"]["mean"],
        "ndcg_at_10_seed_sd": setting["ndcg_at_10"]["seed_sd"],
        "ndcg_at_10_seed_range": setting["ndcg_at_10"]["seed_range"],
        "runtime_seconds_mean": setting["runtime_seconds"]["mean"],
        "runtime_seconds_seed_sd": setting["runtime_seconds"]["seed_sd"],
        "top10_jaccard_mean": setting["ranking_stability"]["top10_jaccard_mean"],
        "top10_jaccard_min": setting["ranking_stability"]["top10_jaccard_min"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, settings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for setting in settings:
            writer.writerow(_flatten_for_csv(setting))
    temporary.replace(path)


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_decision_memo(path: Path, settings: list[dict[str, Any]]) -> None:
    settings_by_id = {item["setting_id"]: item for item in settings}
    routine = settings_by_id["5ep_1neg"]
    quality_focused = settings_by_id["5ep_3neg"]

    rows = []
    for item in settings:
        rows.append(
            "| {id} | {epochs} | {negatives} | {updates:,} | {accuracy:.3f} | "
            "{hr:.3f} | {ndcg:.3f} | {runtime:.2f} | {jaccard:.3f} |".format(
                id=item["setting_id"],
                epochs=item["epochs"],
                negatives=item["negatives_per_positive"],
                updates=item["updates"],
                accuracy=item["heldout_pairwise_accuracy"]["mean"],
                hr=item["hit_rate_at_10"]["mean"],
                ndcg=item["ndcg_at_10"]["mean"],
                runtime=item["runtime_seconds"]["mean"],
                jaccard=item["ranking_stability"]["top10_jaccard_mean"],
            )
        )

    ndcg_gain = quality_focused["ndcg_at_10"]["mean"] - routine["ndcg_at_10"]["mean"]
    hit_rate_gain = (
        quality_focused["hit_rate_at_10"]["mean"] - routine["hit_rate_at_10"]["mean"]
    )
    runtime_ratio = (
        quality_focused["runtime_seconds"]["mean"]
        / routine["runtime_seconds"]["mean"]
    )
    max_accuracy_span = max(
        item["heldout_pairwise_accuracy"]["seed_range"] for item in settings
    )
    memo = f"""# Decision memo: {TITLE}

## Decision

- **Routine iteration:** use **{routine['setting_id']}**. It averaged {routine['runtime_seconds']['mean']:.2f} seconds, NDCG@10 {routine['ndcg_at_10']['mean']:.3f}, and cross-seed top-10 Jaccard {routine['ranking_stability']['top10_jaccard_mean']:.3f}.
- **Final quality-focused run:** use **{quality_focused['setting_id']}** only when an observed gain of {ndcg_gain:.3f} NDCG@10 and {100.0 * hit_rate_gain:.2f} percentage points of HR@10 is worth {runtime_ratio:.2f}x the mean training time.

This is a context-specific operating choice for the fixed MovieLens protocol below. It is not a universal stopping rule: a different dataset, candidate set, hardware target, or business value for ranking quality can change the preferred setting.

## Evidence

Metrics are means across {routine['optimizer_seeds']} optimizer seeds. Runtime covers model training only.

| Setting | Epochs | Negatives / positive | Updates | Pairwise accuracy | HR@10 | NDCG@10 | Runtime (s) | Top-10 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The widest held-out pairwise-accuracy span across seeds was {_format_percent(max_accuracy_span)}. Top-10 Jaccard is a descriptive agreement score across seed pairs on the fixed evaluation candidate sets; higher values mean more overlap in the recommended items.

## Interpretation

- More training is useful only when its metric gain justifies the measured runtime increase.
- Epochs and negatives per positive change both work performed and the examples seen by the optimizer, so comparisons should use the reported update count as context.
- Cross-seed metric spans and top-10 overlap show that one reported run can hide meaningful variation in the ranked output.

## Limits

This is one chronological holdout on MovieLens latest-small with sampled evaluation negatives. It is a compact development benchmark, not a production recommendation, and CPU runtimes will differ by machine. No demographic features are present in the dataset, so this artifact does not assess subgroup performance.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(memo, encoding="utf-8")
    temporary.replace(path)


def run_experiment(
    config_path: Path,
    output_dir: Path | None = None,
    allow_insecure_download: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    model_config = config["model"]
    optimizer_seeds = [int(seed) for seed in config["optimizer_seeds"]]
    output_dir = (output_dir or (root / "results")).resolve()

    dataset_files = download_and_extract(
        root=root,
        url=dataset_config["url"],
        expected_md5=dataset_config["archive_md5"],
        allow_insecure_download=allow_insecure_download,
    )
    audit = run_sql_audit(
        dataset_files.ratings_csv,
        dataset_files.movies_csv,
        root / "sql" / "audit_ratings.sql",
    )
    ratings = load_ratings(dataset_files.ratings_csv)
    movie_ids = load_movie_ids(dataset_files.movies_csv)
    data = prepare_implicit_data(
        ratings,
        movie_ids,
        positive_rating_threshold=float(dataset_config["positive_rating_threshold"]),
    )
    evaluation_candidates = build_evaluation_candidates(
        data,
        negatives_per_user=int(dataset_config["evaluation_negatives_per_user"]),
        seed=int(dataset_config["evaluation_seed"]),
    )

    aggregate_settings: list[dict[str, Any]] = []
    for setting in config["settings"]:
        run_metrics: list[dict[str, float]] = []
        runtimes: list[float] = []
        rankings_by_seed: dict[int, dict[int, frozenset[int]]] = {}
        observed_updates: list[int] = []

        for seed in optimizer_seeds:
            training = train_bpr(
                train_positives=data.train_positives,
                rated_items_by_user=data.rated_items_by_user,
                n_users=data.n_users,
                n_items=data.n_items,
                epochs=int(setting["epochs"]),
                negatives_per_positive=int(setting["negatives_per_positive"]),
                seed=seed,
                latent_dim=int(model_config["latent_dim"]),
                learning_rate=float(model_config["learning_rate"]),
                regularization=float(model_config["regularization"]),
                initialization_scale=float(model_config["initialization_scale"]),
            )
            metrics, top_k = evaluate_model(training.model, evaluation_candidates, k=10)
            run_metrics.append(metrics)
            runtimes.append(training.runtime_seconds)
            rankings_by_seed[seed] = top_k
            observed_updates.append(training.updates)

        if len(set(observed_updates)) != 1:
            raise RuntimeError("Update counts differed across optimizer seeds.")

        stability = {
            key: _rounded(value)
            for key, value in cross_seed_topk_jaccard(rankings_by_seed).items()
        }
        aggregate_settings.append(
            {
                "setting_id": setting["id"],
                "epochs": int(setting["epochs"]),
                "negatives_per_positive": int(setting["negatives_per_positive"]),
                "updates": observed_updates[0],
                "optimizer_seeds": len(optimizer_seeds),
                "heldout_pairwise_accuracy": _summarize(
                    [item["heldout_pairwise_accuracy"] for item in run_metrics]
                ),
                "hit_rate_at_10": _summarize(
                    [item["hit_rate_at_10"] for item in run_metrics]
                ),
                "ndcg_at_10": _summarize(
                    [item["ndcg_at_10"] for item in run_metrics]
                ),
                "runtime_seconds": _summarize(runtimes),
                "ranking_stability": stability,
            }
        )

    result = {
        "title": TITLE,
        "method": "Standard implicit-feedback matrix factorization trained with BPR logistic SGD",
        "dataset": {
            "name": dataset_config["name"],
            "source_url": dataset_config["url"],
            "archive_md5": dataset_config["archive_md5"],
            "rating_rows": int(audit["rating_rows"]),
            "users": int(audit["distinct_users"]),
            "movies": int(audit["movie_rows"]),
            "eligible_users": data.eligible_user_count,
            "train_positive_interactions": int(len(data.train_positives)),
            "heldout_positive_interactions": int(len(data.holdout_positive_by_user)),
            "positive_rating_threshold": float(dataset_config["positive_rating_threshold"]),
            "evaluation_negatives_per_user": int(
                dataset_config["evaluation_negatives_per_user"]
            ),
        },
        "evaluation": {
            "split": "latest qualifying positive per eligible user",
            "candidate_set": "one held-out positive plus fixed sampled unrated items",
            "ranking_stability": "mean top-10 Jaccard across optimizer-seed pairs and users",
            "optimizer_seeds": optimizer_seeds,
        },
        "settings": aggregate_settings,
    }

    _write_json(output_dir / "aggregate.json", result)
    _write_csv(output_dir / "aggregate.csv", aggregate_settings)
    _write_json(output_dir / "data_audit.json", audit)
    _write_decision_memo(root / "DECISION_MEMO.md", aggregate_settings)
    return result
