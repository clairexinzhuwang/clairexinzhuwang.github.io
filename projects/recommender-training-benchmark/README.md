# Training-Budget Trade-offs in a Standard Recommender

A compact, reproducible benchmark of how training epochs and sampled negatives affect a standard Bayesian Personalized Ranking (BPR) recommender. The experiment reports held-out predictive metrics, CPU training time, and descriptive ranking agreement across optimizer seeds.

This is an independent implementation of a well-known public baseline. It does not introduce a new algorithm or make production-performance claims.

## Decision question

For this dataset and evaluation protocol, when does additional BPR training stop producing a practically meaningful improvement in held-out ranking quality?

## Dataset and license

The pipeline downloads **MovieLens latest-small** at runtime from the official GroupLens URL:

- Dataset page: <https://grouplens.org/datasets/movielens/latest/>
- Archive: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
- Dataset README and usage license: <https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html>

The pinned archive MD5 is `31a303aabbc519bd33d025e44d6c2570`, matching the checksum published by GroupLens. The archive describes 100,836 ratings from 610 users on 9,742 movies and was generated on September 26, 2018.

GroupLens describes latest-small as a development dataset that may change over time. Its separate usage license requires attribution, prohibits implying University of Minnesota or GroupLens endorsement, and requires permission for commercial or revenue-bearing use. The raw archive and extracted CSV files are deliberately ignored by Git and are not redistributed here. Review the official README before reuse.

Dataset citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. <https://doi.org/10.1145/2827872>

The repository's MIT license applies to this project's code only, not to MovieLens data.

## Experiment design

1. Treat ratings of 4.0 or higher as positive implicit feedback.
2. Keep users with at least two qualifying positives.
3. Hold out each eligible user's latest qualifying positive.
4. Fit 16-factor matrix-factorization models with standard BPR logistic SGD.
5. Sample negatives only from movies the user has not rated.
6. Evaluate the held-out positive against the same 99 sampled unrated movies for every optimizer seed.
7. Repeat every configuration with seeds 17, 29, and 43.

The four configurations vary only the number of epochs and negatives sampled per positive:

| Setting | Epochs | Negatives per positive |
|---|---:|---:|
| `2ep_1neg` | 2 | 1 |
| `5ep_1neg` | 5 | 1 |
| `5ep_3neg` | 5 | 3 |
| `10ep_1neg` | 10 | 1 |

## Metrics

- **Held-out pairwise accuracy:** fraction of sampled unrated candidates scored below the held-out positive, averaged across users.
- **HR@10:** fraction of users whose held-out positive appears in the top 10 of the 100-item evaluation set.
- **NDCG@10:** rank-discounted top-10 score for the single held-out positive.
- **Training runtime:** wall-clock model-training time; data loading and evaluation are excluded.
- **Top-10 Jaccard:** overlap of top-10 candidate sets across optimizer-seed pairs, averaged across users. This is a descriptive ranking-stability measure, not a guarantee.

Means, standard deviations, and ranges across the three optimizer seeds are reported for predictive metrics. Runtime and top-10 overlap are descriptive and machine-dependent.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_experiment.py
python -m unittest discover -s tests
```

The default downloader validates TLS and then enforces the pinned GroupLens MD5. If the official GroupLens endpoint has a temporary certificate problem, first verify the URL and official checksum yourself; only then use:

```bash
python scripts/run_experiment.py --allow-insecure-download
```

That option disables TLS certificate validation for the download, but checksum validation remains mandatory.

## Outputs

- `results/aggregate.json`: website-ready metadata and nested aggregate metrics.
- `results/aggregate.csv`: one flat row per training configuration.
- `results/data_audit.json`: counts and data-quality checks produced by `sql/audit_ratings.sql`.
- `DECISION_MEMO.md`: a short recommendation generated from the latest aggregate results.

No user-level MovieLens rows are written to `results/`.

## SQL audit

`sql/audit_ratings.sql` checks row counts, distinct users and movies, rating range and half-star increments, timestamps, duplicate user/movie rows, and the minimum rating count per user. The Python pipeline loads the official CSVs into an in-memory SQLite database and saves only the aggregate audit result.

## Repository layout

```text
configs/experiment.json      Experiment settings and fixed seeds
scripts/run_experiment.py    Command-line entry point
sql/audit_ratings.sql        Reusable SQLite data checks
src/ranking_lab/             Download, model, metrics, and experiment code
tests/                       Network-free unit tests
results/                     Generated aggregate outputs
DECISION_MEMO.md             Generated decision summary
```

## Scope

The benchmark uses one chronological holdout and sampled evaluation candidates on a small development dataset. It does not tune hyperparameters, compare production systems, or evaluate demographic subgroups. Runtime comparisons should be repeated on the target machine.
