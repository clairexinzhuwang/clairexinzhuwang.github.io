# Decision memo: Training-Budget Trade-offs in a Standard Recommender

## Decision

- **Routine iteration:** use **5ep_1neg**. It averaged 4.57 seconds, NDCG@10 0.388, and cross-seed top-10 Jaccard 0.898.
- **Final quality-focused run:** use **5ep_3neg** only when an observed gain of 0.016 NDCG@10 and 2.08 percentage points of HR@10 is worth 2.88x the mean training time.

This is a context-specific operating choice for the fixed MovieLens protocol below. It is not a universal stopping rule: a different dataset, candidate set, hardware target, or business value for ranking quality can change the preferred setting.

## Evidence

Metrics are means across 3 optimizer seeds. Runtime covers model training only.

| Setting | Epochs | Negatives / positive | Updates | Pairwise accuracy | HR@10 | NDCG@10 | Runtime (s) | Top-10 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2ep_1neg | 2 | 1 | 95,942 | 0.859 | 0.630 | 0.385 | 1.89 | 0.860 |
| 5ep_1neg | 5 | 1 | 239,855 | 0.863 | 0.636 | 0.388 | 4.57 | 0.898 |
| 5ep_3neg | 5 | 3 | 719,565 | 0.874 | 0.657 | 0.404 | 13.18 | 0.870 |
| 10ep_1neg | 10 | 1 | 479,710 | 0.866 | 0.641 | 0.391 | 9.21 | 0.897 |

The widest held-out pairwise-accuracy span across seeds was 0.4%. Top-10 Jaccard is a descriptive agreement score across seed pairs on the fixed evaluation candidate sets; higher values mean more overlap in the recommended items.

## Interpretation

- More training is useful only when its metric gain justifies the measured runtime increase.
- Epochs and negatives per positive change both work performed and the examples seen by the optimizer, so comparisons should use the reported update count as context.
- Cross-seed metric spans and top-10 overlap show that one reported run can hide meaningful variation in the ranked output.

## Limits

This is one chronological holdout on MovieLens latest-small with sampled evaluation negatives. It is a compact development benchmark, not a production recommendation, and CPU runtimes will differ by machine. No demographic features are present in the dataset, so this artifact does not assess subgroup performance.
