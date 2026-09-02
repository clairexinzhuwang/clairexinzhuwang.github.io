# Email Experiment: From Lift to a Launch Decision

A reproducible product data science case study using Kevin Hillstrom's public
64,000-customer randomized email experiment. The goal is not to maximize the
number of significant findings. It is to make a defensible launch decision,
show the uncertainty around it, and state what the data cannot answer.

## Decision in one minute

Use the **Mens E-Mail** treatment as the default candidate for a monitored
rollout, subject to unit economics and customer-experience guardrails.

- Versus no email, Mens E-Mail increased visits by **7.66 percentage points**
  (95% CI 7.00 to 8.32), conversions by **0.68 points** (0.50 to 0.86), and
  revenue by **0.770 per assigned customer** (0.485 to 1.055).
- Womens E-Mail also beat no email: visit **+4.52 points**, conversion **+0.31
  points**, and revenue **+0.424 per customer**. All six email-versus-control
  findings remain significant after a single Holm correction; the largest
  adjusted p-value is 0.00113.
- In the separately declared head-to-head family, Mens E-Mail generated
  **0.345 more revenue per customer** than Womens E-Mail (95% CI 0.033 to
  0.658; Holm-adjusted p = 0.0305).
- Randomization looks credible on observed baseline variables: the maximum
  absolute standardized mean difference is **0.016**, well below the common
  0.10 diagnostic threshold.
- A single planned `newbie` view finds no corrected evidence of visit-effect
  heterogeneity. It is reported, but it is not used to invent a targeting rule.

These are revenue effects, not profit effects. Campaign cost, gross margin,
unsubscribe risk, and deliverability are absent. If the gross-margin rate is
`m` and marginal send cost is `c` per assigned customer, even the lower end of
the Mens revenue interval supports a positive incremental contribution only
when `c < m × 0.485`, before accounting for longer-term customer costs.

See [decision_memo.md](decision_memo.md) for the concise recommendation.

## Experiment and estimands

The source describes three randomized arms:

| Arm | Customers | Visit rate | Conversion rate | Mean spend |
|---|---:|---:|---:|---:|
| No E-Mail | 21,306 | 10.62% | 0.57% | 0.653 |
| Mens E-Mail | 21,307 | 18.28% | 1.25% | 1.423 |
| Womens E-Mail | 21,387 | 15.14% | 0.88% | 1.077 |

The estimand is the intention-to-treat difference in mean outcome over the
two-week follow-up. Visit and conversion effects are reported as absolute
percentage-point changes; spend is the mean change per randomized customer,
including non-buyers.

### Declared inference plan

The following plan is declared in the case study before the analysis function
is run. It is **not** presented as an externally time-stamped preregistration.

1. **Primary family:** six two-sided comparisons—each email arm versus control
   for visit, conversion, and spend. Apply one Holm family-wise error-rate
   correction across all six.
2. **Secondary family:** three Mens-versus-Womens comparisons for the same
   outcomes. Apply Holm within this separately labeled family.
3. **Planned segment check:** one baseline binary variable (`newbie`) and
   one outcome (`visit`). Test the two treatment-by-newbie interactions with Holm
   adjustment. Stratum-specific estimates are descriptive; different
   significance labels across strata would not establish heterogeneity.

Effects use unpooled difference-in-means (Neyman/Welch) standard errors and
large-sample normal 95% confidence intervals. For binary outcomes this is the
risk difference. Spend is zero-inflated and right-skewed, but mean spend is the
decision-relevant estimand and each arm has more than 21,000 observations.

## Reproduce it

Requirements: Python 3.10 or newer. The analysis and tests use only the Python
standard library.

```bash
python3 -m src.download_data
python3 -m src.analysis
python3 -m unittest discover -s tests -v
```

The downloader fetches the original public CSV, verifies SHA-256
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`, and
stores it under the gitignored `data/raw/` directory. The analysis refuses a
different file by default. Generated outputs are deterministic given that
source file.

## What is in this repository

```text
data/README.md                         source, checksum, and data caveats
sql/01_cohort_and_quality_audit.sql    cohort and validation queries
sql/02_balance_and_outcomes.sql        balance and aggregate effect queries
src/download_data.py                   local-only, checksum-verified download
src/analysis.py                        validation, estimation, and exports
tests/                                 statistical and output-contract tests
outputs/experiment_summary.json        website-ready aggregate summary
outputs/arm_outcomes.csv                aggregate arm metrics
outputs/treatment_effects.csv           all primary and secondary contrasts
outputs/balance_summary.csv             covariate/level balance diagnostics
outputs/newbie_segment_effects.csv      descriptive stratum effects
outputs/newbie_segment_interactions.csv corrected heterogeneity tests
decision_memo.md                        action-oriented interpretation
```

The SQL assumes the source CSV has been loaded unchanged as `hillstrom_raw`.
It uses DuckDB/PostgreSQL-style syntax and makes the no-exclusion ITT cohort
explicit. Python is the inferential source of truth for intervals and adjusted
p-values.

## Data audit

- 64,000 rows; arm counts match the public experimental design.
- No missing cells, out-of-domain binary values, negative spend, conversions
  without visits, or inconsistencies between conversion and positive spend.
- The source value `Surburban` occurs in 28,776 rows. It is logged and mapped to
  `Suburban` in memory; the raw file is not rewritten.
- There are 6,562 rows beyond the first copy of an identical released record.
  Because the release has no customer ID and many non-buyers share all released
  values, these cannot be labeled duplicate customers. They remain in the ITT
  cohort; silently dropping them would change the randomized sample.

## Limits and next decision

This experiment identifies short-run causal effects for the randomized
population, not a universal email effect. It does not reveal why a creative
worked, whether it caused unsubscribes, or whether its incremental revenue
covered cost. Before a broad launch, attach campaign cost and gross margin,
choose unsubscribe and deliverability guardrails, and monitor the same ITT
metrics during rollout. A future test can vary creative mechanism deliberately;
this dataset does not justify searching many retrospective segments.

## Data rights

The dataset was publicly released by Kevin Hillstrom as an analytics challenge.
The original page does not state a formal data license, so the raw CSV is not
redistributed here. See [data/README.md](data/README.md) for the source URL,
checksum, and usage note.
