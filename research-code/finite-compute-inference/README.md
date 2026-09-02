# Inference from stochastic gradient descent for U-statistics

> Browser-friendly curated derivative of the numerical package bundled on
> August 31, 2026. Executable logic and stored numerical records are unchanged,
> while presentation-only comments were edited and legacy/manuscript-sync files
> were omitted. This tree is therefore not byte-identical to the separately
> published code archive. The manuscript itself is intentionally not included.
> Public visibility does not grant a license to copy, modify, or redistribute the
> code; see `NOTICE.md`.

The shipped replication-level Parquet files require the pinned `pyarrow` dependency.

Code and result records for the numerical study. Reported simulation tables and
figures are produced from the stored results by the scripts here rather than entered
by hand. The two real-data fields identified in the verification section require a
data-backed rerun because the raw public datasets are not redistributed.

The estimation procedure appears exactly once, in `code/alg_paper.py`. Nothing
else in this repository implements it.

## Layout

    code/
      alg_paper.py                 Algorithm 1: pilot, fixed gain, 1/t recursion,
                                   post-run covariance, anchor estimator
      kernels.py                   the three kernels: AUC, smoothed rank, triplet
      dgp.py                       simulation designs and population targets
      audit_fixed_B.py             twelve checks over the fixed-minibatch cells
      sensitivity_post_run.py      how the reported errors move with P and m
      experiment_design.py         the frozen design of the fixed-minibatch study
      run_fixed_B_grid.py          the fixed-minibatch study: B = 1, 10, 100
      smoke_fixed_B.py             twelve design checks, run before any formal run
      make_fixed_B_outputs.py      inferential summary figure and fixed-B tables
      run_grid.py                  the archived grid, in which B grows with n
      time_pass.py                 timings, measured serially in one pass
      audit_cells.py               twelve checks the results must pass
      make_tables.py               the simulation tables
      make_figure.py               archived growing-B summary figure
      make_trace_figures.py        the two supplementary figures
      make_realdata_table.py       the main real-data table
      make_realdata_supp_table.py  the supplementary real-data table
      sync_figures.py              copies generated output into the manuscript
      paths.py                     where the package keeps inputs and outputs
      theta_star_p4K5_3e8.npz      triplet population target, 3e8 valid triplets
      theta_star_auc_2e8.npz       AUC population target, 2e8 draws
    real_data/
      prepare_diabetes.py          builds the readmission design from the raw file
      diabetes_auc_analysis.py     the readmission analysis
      fremtpl2_rank_analysis.py    the freMTPL2 analysis
      data_checksums.json          SHA-256 of every input file
      results/                     output of the two analyses
    experiments_fixed_B/           the fixed-minibatch study: frozen_config.json and
                                   its hash, aggregated_results/, tables/, figures/,
                                   diagnostics/, logs/, EXPERIMENT_AUDIT.md, and the
                                   per-replication records under formal_runs/,
                                   wor_robustness/ and budget_ratio_runs/, which the
                                   audit gate re-derives every summary row from
    results/                       the archived grid: grid_{rank,auc,triplet}.json,
                                   timing_pass.json
    figures/                       generated tables and figures (created on first run)

## Environment

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

The pinned versions are the ones the reported results were produced under.
NumPy 2.x changes some reduction orders, so results under it agree to about
1e-9 rather than exactly.

## Rebuilding stored simulation tables and figures

The stored result files for the simulation grids are included, so their tables
and figures can be rebuilt without rerunning the formal simulations:

    cd code
    python -u audit_fixed_B.py              # primary fixed-B gate
    python -u make_fixed_B_outputs.py       # main summaries, supplement tables and Figure 1
    python -u audit_cells.py                # archived growing-B gate
    python -u make_tables.py
    python -u make_figure.py                # archived scalability figure
    python -u make_realdata_table.py
    python -u make_realdata_supp_table.py

The trace figures are different: regenerating them performs 60 × 12 new fits
rather than reading only stored outputs:

    python -u make_trace_figures.py

## The simulations the paper reports

The primary study is the fixed-minibatch design: `B = 1, 10, 100`, each one an
asymptotic sequence along which `B` does not change and the number of iterations
grows with the sample size, at the horizon the paper prescribes,
`T = floor((n/B) log2 n)`. Three kernels, `n = 10^3, 10^4, 10^5`, `R = 1000`
replications in every cell. The separate budget-ratio experiment at $n=10^4$
and $B=10$ is now summarized in the main paper because it directly exercises the
finite-$\lambda$ regime; its fuller variance table remains in the supplement.

The revised main summary figure averages each diagnostic over all parameter
coordinates of the corresponding example. It shows B=1,10,100 separately rather
than averaging them: colour identifies the kernel and marker/line style
identifies B. The four panels give full-interval coverage, data-only coverage,
algorithmic variance share, and empirical-to-predicted variance agreement. The
main paper also contains a compact coordinate-averaged table and a finite-budget
table; complete cell-level WR/WOR tables remain in the supplement.

    cd code
    python -u smoke_fixed_B.py                 # twelve design checks; run this first
    python -u run_fixed_B_grid.py primary 1.0 8
    python -u run_fixed_B_grid.py wor     1.0 8
    python -u run_fixed_B_grid.py sweep   1.0 8
    python -u time_pass_fixed_B.py             # the Time column, serially, when idle
    python -u sensitivity_post_run.py 40       # the supplement's P and m check
    python -u audit_fixed_B.py                 # the gate; non-zero on failure
    python -u make_fixed_B_outputs.py

There are two audit gates and they read different files. `audit_fixed_B.py`
covers the fixed-minibatch cells, which is where the paper's primary tables and
figure come from; `audit_cells.py` covers the archived growing-B grid. Running
one is not running the other.

`experiment_design.py` holds every constant of that design as a pure function of
the sample sizes. The first formal run writes `experiments_fixed_B/
frozen_config.json` and its SHA-256; a later run that finds a different design
refuses to start, so the reported cells all come from one design and one hash.
`smoke_fixed_B.py` exits non-zero unless all twelve checks pass: the schedule,
the shared draws, the preconditioner being identical across `B`, the recursion
stream being consumed as nested prefixes, and the covariance coefficient
collapsing to `1/(BT)` under sampling with replacement.

The without-replacement runs cover `B = 10, 100` only. At `B = 1` a minibatch
drawn without replacement is the with-replacement draw, returning the same tuple
from the same generator state, so the cell would duplicate one already run.

## The archived grid

`run_grid.py` is the driver for the earlier grid, in which the minibatch size
grows with the sample size and which reaches `n = 10^6` and `B = 16,384`. Those
runs are retained in the supplement as scalability experiments outside the
fixed-`B` asymptotic design, and they are not the primary validation of the
theorem. They are reproduced by

    cd code
    python -u run_grid.py rank    1000 10
    python -u run_grid.py auc     1000 10
    python -u run_grid.py triplet 1000 10

`GRIDS` and `GRID_TAG` at the top of that file define which grid is run and
which file it writes. The archived grid uses an empty tag and writes
`results/grid_<kernel>.json`; any other grid must set a tag and writes its own
file, so two grids cannot accumulate in one place and be tabulated together.
Rows from the two designs must never appear in one table.

Runs are resumable: an existing result file is read back and only missing cells
are computed.

Timings are measured separately, serially, in one pass, on an idle machine, so
that the Time column is comparable across cells:

    python -u time_pass.py

Then rebuild as above, running `audit_cells.py` before anything is written into
the paper.

Expect roughly a day of wall-clock for the full grid at `R = 1000` on ten
workers, dominated by the triplet kernel at `n = 10^6`.

## Real-data analysis status

The analysis scripts and final result JSON files are included, but this public
snapshot does not yet provide an end-to-end transformation from the raw public
archives to every prepared analysis input. `prepare_diabetes.py` expects
`real_data/data/diabetes130_with_ids.csv`; `fremtpl2_rank_analysis.py` expects
`real_data/data/fremtpl2_policy.csv`. Expected SHA-256 digests are recorded in
`real_data/data_checksums.json`.

Until the preparation step is published and documented, treat the stored
real-data JSON files as inspection artifacts. Reproducing the fits requires the
prepared inputs. Once those inputs are present, the analysis commands are:

    cd real_data
    python -u prepare_diabetes.py
    python -u diabetes_auc_analysis.py --bootstrap 300 --splits 100
    python -u fremtpl2_rank_analysis.py --bootstrap 100

Both use the primary numerical schedule, `T = (n/B) log2 n`. The `--mult`
argument scales that schedule and is 1 for every reported number.

## Verification built into the scripts

* `audit_fixed_B.py` runs twelve checks over the fifty fixed-minibatch cells and
  exits non-zero on failure. Among them: that the horizon matches the frozen primary
  schedule, that the auxiliary budgets do not vary with the minibatch size, that
  the two variance estimates agree across minibatch sizes, that a cell with one
  minibatch size does not claim to demonstrate invariance, and that every
  summary row reproduces from the replication records it was computed from.
* `audit_cells.py` runs the same kind of gate over the archived growing-B grid;
  it passes with no failures and five warnings, listed in its output.
* `make_tables.py` re-reads each printed number from the result files and
  reports a mismatch rather than printing silently.
* Two real-data quantities the manuscript quotes were computed but not stored
  by the runs that produced them: the stochastic-optimization share of the plug-in variance
  for the readmission fit, and the ratio of the ordinary-least-squares residual
  scale to the robust one for freMTPL2. The analysis scripts now record
  `var_data`, `var_alg`, `alg_share` and `ols_scale` (the latter two names are retained internal fields), so a rerun makes both
  re-derivable; the prepared inputs are not redistributed here, so the rerun
  needs the inputs described above. Until then those two numbers rest on the runs that
  are reported, not on anything in `real_data/results/`.
* `sync_figures.py` verifies every target it is supposed to write and exits
  non-zero if a generated file is missing, so a stale table cannot survive in
  the manuscript unnoticed.
* The recursion's per-step score path, `score_mean` on each kernel, is a
  verbatim slice of that kernel's `stats`. It exists only to skip the loss,
  Hessian and per-tuple scores that the loop discards, and the two paths were
  checked end to end: 792 output arrays over three kernels, three minibatch
  sizes, both schemes and three seeds, all exactly equal.

## Usage and licence

No open-source licence is granted with this inspection snapshot. See `NOTICE.md`.
