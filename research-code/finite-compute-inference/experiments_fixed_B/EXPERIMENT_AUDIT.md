# Fixed-minibatch study: what was run, and what was checked before it ran

> **Terminology note.** This frozen design audit predates the current presentation terminology. Its uses of “algorithmic” refer to the stochastic-optimization component, and internal fields such as `v_alg` and `alg_share` remain unchanged for reproducibility. The current presentation uses the single ratio `Lambda_n = BT/n_1`.

## The question the archived grid could not answer

In the archived grid the minibatch size grew with the sample size, so no two
cells differed in `B` alone and nothing in it isolated the effect of the
minibatch. This study fixes the tuple budget and varies the minibatch size
inside it.

The horizon is the schedule the paper prescribes,

    T = floor((n_total/B) log2 n_total),

so the tuple budget `BT = n_total log2 n_total` does not depend on `B`: the
minibatch size changes how one budget is spent, not how much of it there is.

At finite `n` the plug-in algorithmic covariance is
`{t_0 + f(T - t_0)} V / (B T^2)`, which under sampling with replacement is
`V/(BT)`. It is set by the tuple budget, not by how the budget is divided into
minibatches, so a common `BT` predicts a common algorithmic contribution across
`B`. That is a finite-sample prediction the design tests, not an assumption it
makes.

What this schedule does NOT instantiate. It sends `T/n_1` to infinity along
every sequence, so the grid sits in the branch of the theorem whose root-`n_1`
limit carries the data component alone. Nothing here should be described as a
nondegenerate algorithmic limit, and the data-only undercoverage the grid
reports is a finite-sample effect that shrinks as the budget outgrows the
sample, not the limiting undercoverage stated for a finite `lambda`.

`Lambda` is not `lambda`, and neither is called `lambda` in the code or the
tables. `n_1` is the theorem's reference sample size, which for two-sample AUC
is the positive sample, not the total.

## Design, frozen before any run

`code/experiment_design.py` computes every constant as a pure function of the
sample sizes. `frozen_config.json` and its SHA-256 are written on the first
run and checked on every later one: a driver that finds a different design
refuses to start.

| | |
|---|---|
| minibatch sizes | 1, 10, 100 |
| total sample sizes | 10^3, 10^4, 10^5 |
| kernels | smoothed rank, triplet metric, two-sample AUC |
| horizon | `T = floor((n/B) log2 n)`, so `BT = n log2 n` is common across `B` |
| scheme | with replacement, primary grid |
| replications | 1000 in every cell, so the coverage Monte Carlo standard error is 0.0069 throughout |
| anchor groups | `P = 16` conditional tuples per group, two groups per anchor |

The pilot budget is rounded up to a multiple of `L`, the least common multiple
of the minibatch sizes being compared, so that it divides evenly by every `B`
and `B t_0` is the same integer for each:

    L   = lcm(B_GRID) = lcm(1, 10, 100) = 100
    N0  = B t_0 = L * ceil(n_total^0.60 / L)
    t_0 = N0 // B
    T   = floor((n_total/B) log2 n_total)

`B T` is then common across `B` up to the integer part of the horizon, a
residual smaller than the largest minibatch; the recursion streams are nested,
so the shorter budgets consume prefixes of the longest.

`L` is an integer-rounding device and nothing else. It is not a statistical
tuning constant, nothing in the analysis depends on its value, and it follows
from `B_GRID`: comparing `B` in {1, 4, 8} would give `L = 8`. It happens to
equal `max(B_GRID)` for this grid, which is a coincidence of {1, 10, 100} and
not a definition; `B_MAX`, where the auxiliary budgets are evaluated, is a
separate matter.

The two budgets do two different jobs and neither substitutes for the other.
`N_0 = B t_0` controls the localization of the preliminary estimator. It does
not on its own control the accuracy of the initial Hessian: that is what the
independent Hessian sample `m_0` is for, and `m_0` is held fixed across `B`.

The exponent 0.60 is a prespecified schedule that is sufficient for all three
kernels. It is not claimed to be optimal in finite samples and nothing here is
tuned to it. The archived pilot rule is not used: its `T/2` cap makes the pilot
budget a function of the horizon, hence of `B`, which would confound the very
comparison being made. It survives as `legacy_default_t0` so that archived
results still reproduce.

Auxiliary budgets `m_0`, `m`, `q_k`, `P` are evaluated once per dataset, at
`B = 100`, and reused for every `B`. The archived rule contains a `4B` term, so
calling it per minibatch size would hand the larger minibatches more Monte
Carlo precision in the terminal Hessian and covariance, and that difference
could then be read as an effect of `B`.

## What makes the comparison paired

Within one replication the three minibatch sizes share:

- the dataset;
- the tuple budget `M = BT` and the pilot budget `N_0 = B t_0`;
- the auxiliary budgets `m_0`, `m`, `q_k`, `P`;
- the pilot tuples, the initial-Hessian tuples, the terminal tuples and the
  anchors, drawn from per-component generators seeded identically across `B`;
- one ordered recursion stream of `B(T - t_0)` tuples, cut into consecutive
  blocks of size `B`.

So the pilot estimate and the fixed preconditioner `D_0` are identical across
`B` to the last bit, and the only thing that differs is how one fixed stream of
tuples was grouped. Every summary row carries `D0_same_across_B`, computed by
hashing the matrix in each replication.

The `streams` argument that does this changes which tuples are drawn and
nothing else: with it left at its default the estimator draws from a single
generator exactly as the archived results were produced, and the default path
was checked to be bit-identical to the previously released code over 792 output
arrays spanning three kernels, three minibatch sizes, both schemes and three
seeds.

Under sampling without replacement the recursion draws per step, because
drawing without replacement is a property of a batch and a batch of size `B` is
not a sub-block of a longer stream. The without-replacement runs are therefore
not paired across `B` in the tuple stream, only in the dataset and the
auxiliary draws.

## Pre-flight checks

`code/smoke_fixed_B.py` runs twelve checks per kernel and exits non-zero on any
failure. They test the design and the pairing, not the statistics: imperfect
coverage or a finite-sample variance mismatch is not a failure and is not
tested here.

1. `T` is the prescribed schedule and `B*T` is common up to the integer part.
2. `B*t0 == N0` exactly, and `N0` is the same for every `B`.
3. `0 < t_0 < T`, and the schedule differs from the archived capped rule.
4. Pilot tuple indices identical across `B`.
5. Pilot estimates agree across `B`.
6. Initial-Hessian tuples and `D_0` identical across `B`.
7. One recursion stream, with the shorter budgets consuming prefixes of it:
   the blocks a run actually reads are reassembled and compared with that
   prefix, and no run reads past the end.
8. Terminal tuples and anchor draws shared across `B`.
9. `m_0`, `m`, `q_k`, `P` identical across `B`.
10. Under with-replacement sampling `{t_0+(T-t_0)}/(B T^2)` equals `1/(BT)`.
11. All covariance matrices finite.
12. Floor, projection and failure diagnostics all present.

## Reported quantities

Per replication: the estimate, the three variance diagonals (`Sigma_dat`,
`alg_full`, `Sigma_full`), the preconditioner hash, pilot diagnostics,
eigenvalue-floor activations and condition numbers, the PSD projection
frequency and displacement, finiteness, and wall-clock for the pilot, the
initial Hessian, the recursion, the terminal estimates, the anchors and the
total.

Per cell: empirical variance, mean plug-in data variance, mean plug-in
algorithmic variance, their sum, the empirical-to-predicted ratio, the
algorithmic share, mean absolute coordinate bias, full coverage, data-only
coverage, the coverage Monte Carlo standard error, `T`, `t_0`, `T - t_0`, `BT`,
`B t_0`, the auxiliary budgets, phase runtimes, and the conditioning and
failure diagnostics.

What the reported scalars are and are not. The proved consistency statements
are `zeta_hat_{n,k} -> zeta_k`, `V_hat_{n,T} -> V` and `D_hat_{n,T} -> A^{-1}`,
all statements about matrices. `zeta_hat_{n,k}` is what `Sigma_data` is built
from and `V_hat_{n,T}` is what `Sigma_alg` is built from; neither converges to
a scalar, and neither is `v_data` or `v_alg`. Those two columns are simulation
summaries only: the coordinate average of the diagonal of the corresponding
covariance matrix, averaged over replications, reported so that it can be set
beside `v_emp`, the coordinate average of the Monte Carlo variance of the
estimates.

The data-only interval uses `Sigma_dat`. It is not `Sigma_main`, which is the
archived audit object that drops the pilot's own sampling variance from the
algorithmic coefficient; that field is retained and reported as
`v_legacy_main`, and is not used for any interval.

## Rate schedules and the budgets on the displayed grid

The pilot and auxiliary-sample schedules satisfy the required asymptotic rate
conditions. These are properties of the design sequence as `n -> infinity` and
they hold analytically for these rules:

    N_{0,n} = L ceil(n^0.60 / L) asymp n^0.60,   h_{0,n} = n^-0.26
      =>  N_{0,n} h_{0,n}^2 asymp n^(0.60-0.52) = n^0.08 -> infinity

    m_0 >= n^0.60, that term being inside the maximum for every n
      =>  m_0 h_{0,n} >= n^(0.60-0.26) = n^0.34 -> infinity   (smoothed rank)
      =>  m_0         >= n^0.60                 -> infinity   (fixed kernels)

Fixed lower bounds stabilize the reported finite-sample computations and
determine some budgets on the displayed grid; they do not alter the asymptotic
rates and are not theorem-derived constants. On `n = 10^3, 10^4, 10^5` the
realized values are therefore floor-dominated, and `code/experiment_design.py`
records which term of each maximum binds:

| kernel | n | N_0 | N_0 h_0^2 | m_0 | m_0 h_0 | m_0 set by |
|---|---|---|---|---|---|---|
| rank | 10^3 | 100 | 2.75 | 8801 | 1461 | 1500/h floor |
| rank | 10^4 | 300 | 2.50 | 16474 | 1502 | 1500/h floor |
| rank | 10^5 | 1000 | 2.51 | 30054 | 1506 | 1500/h floor |
| auc, triplet | 10^3 | 100 | 2.75 | 8192 | 1360 | 8192 floor |
| auc, triplet | 10^4 | 300 | 2.50 | 8192 | 747 | 8192 floor |
| auc, triplet | 10^5 | 1000 | 2.51 | 8192 | 411 | 8192 floor |

Monotone growth of three finite numbers is not a requirement and is not
reported as a pass or a failure. The divergence of `N_0 h_0^2` carries exponent
0.08 and is correspondingly slow; the rest is the floors binding at these
sample sizes. Nothing here is a test of the rate conditions, which are
analytic.

The pilot tuple budget controls localization of the preliminary estimator,
whereas `m_0` controls Monte Carlo accuracy of the independently estimated
preconditioner.

## Questions this is meant to answer

1. Is the empirical variance approximately the data variance plus the
   algorithmic variance?
2. At fixed `BT` and fixed `B t_0`, are variance and coverage approximately
   invariant across `B = 1, 10, 100`?
3. Do data-only intervals under-cover when the algorithmic share is material?

## Changes to the frozen design, and when they were made

The replication count was made uniform at 1000 before any formal result was
written or inspected. The driver holds a stage's records in memory and writes
them only when the stage completes, and no stage had completed: the two runs
that had finished were the twelve pre-flight checks and a twenty-replication
runtime benchmark, neither of which reports coverage or variance for a formal
cell. The change raises the replication count for the largest sample size and
for the without-replacement runs from 500 to 1000; it does not lower it
anywhere. The configuration was re-frozen and re-hashed at that point and the
runs restarted from the beginning, so every reported cell comes from one
design and one hash.

## Old results

Untouched. The archived growing-`B` grid stays in `results/grid_*.json` and
this driver never writes there; `paths.grid` takes a tag naming the grid, and
anything other than the reported grid writes its own file. Rows from the two
grids must never appear in one table.
