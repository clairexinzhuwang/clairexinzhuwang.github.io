# Decision memo: choose a default email treatment

**Decision owner:** Lifecycle / CRM lead  
**Population:** Customers eligible for the historical campaign  
**Primary decision metric:** Incremental revenue per randomized customer  
**Recommendation:** Advance **Mens E-Mail** as the default candidate for a
monitored rollout, provided incremental contribution clears the send-cost
threshold and unsubscribe/deliverability guardrails are acceptable.

## Evidence

The 64,000-customer randomized experiment is well balanced on the observed
baseline fields (maximum |SMD| = 0.016). No post-assignment records were
excluded.

| ITT contrast | Visit lift | Conversion lift | Mean spend lift |
|---|---:|---:|---:|
| Mens E-Mail − No E-Mail | +7.66 pp [7.00, 8.32] | +0.68 pp [0.50, 0.86] | +0.770 [0.485, 1.055] |
| Womens E-Mail − No E-Mail | +4.52 pp [3.89, 5.16] | +0.31 pp [0.15, 0.47] | +0.424 [0.169, 0.680] |
| Mens E-Mail − Womens E-Mail | +3.14 pp [2.43, 3.84] | +0.37 pp [0.17, 0.56] | +0.345 [0.033, 0.658] |

Brackets are 95% confidence intervals. The six email-versus-control tests were
Holm-adjusted as one primary family; all survive (largest adjusted p = 0.00113).
The three head-to-head tests were a separately declared secondary family; the
adjusted p-value for its spend contrast is 0.0305.

## Interpretation

Both emails causally improved the full funnel relative to no email in this
experiment. Mens E-Mail has the higher observed mean on every outcome and a
positive head-to-head spend interval, so it is the stronger default candidate.
The evidence is for assignment to that historical treatment—not for a general
claim that men's merchandise or the same creative will always win.

The sole planned case-study segment check compared visit effects by
baseline `newbie` status; it was not an external, time-stamped preregistration.
Neither treatment-by-newbie interaction survives Holm
correction (adjusted p = 0.708 for Mens and 0.222 for Womens). Do not introduce
a newbie targeting rule from these data.

## Launch condition and guardrails

The file contains revenue but no cost, margin, delivery, or unsubscribe data,
so it cannot establish profit. With gross-margin rate `m` and marginal send
cost `c`, the conservative lower confidence bound for Mens E-Mail implies
positive short-run contribution only if `c < m × 0.485` per assigned customer.
This still excludes longer-run customer-experience costs.

For rollout, pre-register:

- incremental contribution per eligible customer as the decision metric;
- unsubscribe/complaint rate and deliverability as guardrails;
- a holdout to detect transport drift from the historical experiment; and
- a stop/review rule if the revenue or guardrail interval crosses its threshold.

Do not optimize on open or click rate alone, and do not mine additional segments
until a future experiment is sized and designed to test heterogeneity.
