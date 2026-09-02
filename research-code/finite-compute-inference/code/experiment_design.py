"""Frozen design constants and schedules for the fixed-minibatch study.

Every quantity the fixed-B experiment depends on is computed here, by pure
functions of the sample sizes, so that the design can be hashed before any run
and checked afterwards against what was actually executed.

Notation, which differs from the archived grid and must not be conflated:

    n_total = sum_k n_k          the total number of observations
    n_1                          the theorem's REFERENCE sample size, the size
                                 of sample 1; equal to n_total for the
                                 one-sample kernels, and the positive sample for
                                 two-sample AUC
    lambda  = T / n_1            the manuscript's root-T parameter
    Lambda  = B T / n_1          the tuple-budget ratio

The horizon is the schedule the paper prescribes,

    T = floor( (n_total / B) log2 n_total ),

so the tuple budget B T is n_total log2 n_total up to the integer part and does
not depend on B: the minibatch size changes how one budget is spent, not how
much of it there is.  The comparison across B is therefore made at a common
budget without holding any ratio artificially fixed.

The budget ratio that results, Lambda = B T / n_1 = (n_total/n_1) log2 n_total,
grows slowly with the sample size: it is log2 n_total for the one-sample
kernels and 2.5 log2 n_total for two-sample bipartite ranking, whose reference
stratum holds 0.4 of the data.  Do not call B T / n_1 "lambda".

What this schedule does and does not instantiate.  It sends T/n_1 to infinity
along every sequence, so the grid sits in the branch of the theorem whose
root-n_1 limit carries the data component alone; the stochastic-optimization term is
asymptotically negligible under it.  What the grid examines is that term's
FINITE-SAMPLE size.  At finite n the plug-in stochastic-optimization covariance is
{t_0 + f(T - t_0)} V / (B T^2), which under sampling with replacement is
V / (B T): it is set by the tuple budget, not by how the budget is split into
minibatches.  Since B T = n log2 n does not depend on B, the algorithmic
contribution should be common across B at each n, and that is the prediction
the design tests.  It is not a statement about a nondegenerate limit.

The pilot exponent 0.60 is a prespecified schedule that is sufficient for every
kernel here.  It is not claimed to be optimal in finite samples, and nothing in
the study is tuned to it.
"""
import math

import numpy as np

# --- frozen constants ------------------------------------------------------
B_GRID = (1, 10, 100)
N_GRID = (1_000, 10_000, 100_000)
P_ANCHOR = 16                   # conditional tuples per anchor group
PILOT_EXPONENT = 0.60
#: Replications per cell. Uniform, so that the Monte Carlo standard error of
#: a coverage near 0.95 is the same 0.0069 in every cell and cells are
#: directly comparable without carrying a per-cell error bar.
R_FORMAL = 1000
R_BY_N = {n: R_FORMAL for n in N_GRID}

#: Integer-rounding device, NOT a statistical constant.  L is the least common
#: multiple of the minibatch sizes being compared, so that a budget rounded up
#: to a multiple of L divides exactly by every B in the comparison and B t_0 and
#: B T are identical across B.  Nothing statistical depends on its value: change
#: B_GRID and L follows.  It must never be described as a tuning constant.
L_ROUND = math.lcm(*B_GRID)

#: Where the auxiliary-budget rule is evaluated, so that m_0 and m do not vary
#: with B.  This is a separate matter from L_ROUND, which happens to coincide
#: with it for this B_GRID: L_ROUND makes the budgets divisible, B_MAX fixes the
#: Monte Carlo precision of the auxiliary samples.
B_MAX = max(B_GRID)

# The budget-ratio experiment, summarized in the main paper and tabulated fully in the supplement.  It is the one place
# the tuple budget is set directly rather than by the primary numerical schedule,
# because its whole purpose is to vary the budget at a fixed sample size and a
# fixed minibatch size, so that the optimization share can be driven from a few
# per cent to a majority of the plug-in variance.  It is a slice, not a
# Cartesian product with B.
LAMBDA_TUPLE_SWEEP = (1.0, 2.0, 5.0, 10.0, 20.0)
SWEEP_KERNEL, SWEEP_N, SWEEP_B = "rank", 10_000, 10

# without-replacement robustness, likewise a slice rather than a product
# Without replacement, over the whole grid, because the tables report coverage
# and timing under both schemes side by side.  B = 1 is omitted: a size-one
# minibatch drawn without replacement is the with-replacement draw, returning
# the same tuple from the same generator state, so that cell would duplicate the
# with-replacement one exactly and the tables reuse it.
WOR_N, WOR_B = N_GRID, (10, 100)
WOR_R = R_FORMAL


def round_up(x, base=None):
    """Smallest multiple of `base` at or above x; `base` defaults to L_ROUND.

    Purely an integer-rounding device: it makes a budget divisible by every
    minibatch size in the comparison, so that one tuple stream cuts into blocks
    of any B in B_GRID with nothing left over and B t_0 and B T are the same
    number for every B.  It carries no statistical meaning.
    """
    base = L_ROUND if base is None else base
    return int(base * int(np.ceil(float(x) / base)))


def total_optimization_budget(n1, lam):
    """B T for the budget-ratio experiment, which sets the budget directly."""
    return round_up(lam * n1)


def horizon(n_total, B):
    """T = floor((n_total/B) log2 n_total), the schedule the paper prescribes.

    B T is then n_total log2 n_total up to the integer part, whatever B is, so
    the three minibatch sizes spend one common tuple budget.  The residual
    differences are the flooring alone, fewer than B tuples.
    """
    return int(np.floor(n_total * np.log2(n_total) / B))


def pilot_tuple_budget(n_total):
    """N_0 = B t_0 = L ceil(n_total^0.60 / L), a function of the sample size alone.

    What it controls: the localization of the preliminary estimator, i.e. how
    close the pilot lands to the empirical minimizer.  It does NOT on its own
    control the accuracy of the initial Hessian; that is the job of the
    independent Hessian sample size m_0, which is a separate budget and is held
    fixed across B.

    Set directly rather than through the archived pilot rule, whose T/2 cap
    would make the pilot budget depend on the horizon and therefore on B.
    """
    return round_up(n_total ** PILOT_EXPONENT)


def schedule(n_total, n1, B, lam=None):
    """(B T, N_0, T, t_0) for one cell, under the primary numerical schedule.

    With `lam` at None the horizon is the primary numerical schedule, which is what
    every reported cell of the primary study uses.  A named `lam` sets the
    tuple budget directly, B T = round_up(lam * n_1), which only the
    budget-ratio experiment does.  The pilot budget N_0 = B t_0
    is a function of the sample size alone in either case, and is divisible by
    every minibatch size in the comparison, so t_0 = N_0 / B is exact.
    """
    N0 = pilot_tuple_budget(n_total)
    assert N0 % B == 0, (N0, B)
    if lam is None:
        T = horizon(n_total, B)
    else:
        M = total_optimization_budget(n1, lam)
        assert M % B == 0, (M, B)
        T = M // B
    t0 = N0 // B
    assert 0 < t0 < T, (t0, T)
    return B * T, N0, T, t0


def auxiliary_budgets(default_m, n_total, h_eff=1.0):
    """m_0 = m, evaluated once at B_MAX and reused for every B.

    The archived rule contains a 4B term, so calling it per B would give the
    larger minibatches more Monte Carlo precision in the terminal Hessian and
    covariance, and any difference across B could then be read as an effect of
    B when it was an effect of the budget. Evaluating at B_MAX fixes the
    auxiliary precision at its most generous value for every cell.
    """
    m_shared = int(default_m(B_MAX, n_total, h_eff))
    return m_shared, m_shared


def anchor_counts(kernel):
    """q_k, one per sample, fixed across B: every anchor the sample has."""
    return [int(n_k) for n_k, _d_k in kernel.samples]


#: The deterministic bandwidth sequence of the smoothed rank kernel,
#: h_{0,n} = n^{-BANDWIDTH_EXPONENT}; the kernel's own h_n is this times a
#: consistent scale estimate.
BANDWIDTH_EXPONENT = 0.26


def h0(n_total):
    """h_{0,n}, the deterministic bandwidth sequence."""
    return float(n_total ** (-BANDWIDTH_EXPONENT))


def binding_term(B, n_total, h_eff):
    """Which term of the auxiliary-budget maximum sets m_0 at this (n, h).

    The rule is max(4B, n_total^0.60, 8192, 1500/h_eff).  Two of those are
    prespecified finite-sample floors that stabilize the reported computations,
    and one, n_total^0.60, is the term that carries the asymptotics.  Reporting
    which one binds is a description of the displayed grid, not a test.
    """
    terms = {"4B": 4.0 * B, "n^0.60": float(n_total) ** 0.60,
             "floor_8192": 8192.0, "floor_1500/h": 1500.0 / h_eff}
    return max(terms, key=terms.get), terms


def rate_schedule_report(default_m, kernels_n_total_h, lam=None):
    """Realized budgets on the displayed grid, and which term of each maximum binds.

    The asymptotic conditions are properties of the design SEQUENCE and hold
    analytically for these rules:

        N_{0,n}  = L ceil(n^0.60 / L)  asymp  n^0.60,   h_{0,n} = n^-0.26
          =>  N_{0,n} h_{0,n}^2  asymp  n^(0.60-0.52) = n^0.08  ->  infinity

        m_0 >= n^0.60  (that term is inside the maximum, always)
          =>  m_0 h_{0,n} >= n^(0.60-0.26) = n^0.34  ->  infinity   [rank]
          =>  m_0        >= n^0.60                   ->  infinity   [fixed kernels]

    The divergence in the first is slow, exponent 0.08, and on n = 10^3 to 10^5
    the realized value sits near 2.5; in the others the prespecified floors
    8192 and 1500/h_eff dominate at these sample sizes, so the realized values
    are nearly constant.  That is a statement about which term binds on the
    displayed grid.  It is not a test of the rate conditions, and monotone
    growth of three finite numbers is neither required nor reported as a pass
    or a failure.  The floors are not theorem-derived constants and do not
    change the rates.
    """
    rows = []
    for name, n_total, h_eff in kernels_n_total_h:
        N0 = pilot_tuple_budget(n_total)
        m0, _m = auxiliary_budgets(default_m, n_total, h_eff)
        hn = h0(n_total)
        which, terms = binding_term(B_MAX, n_total, h_eff)
        rows.append({
            "kernel": name, "n_total": n_total, "h0_n": hn,
            "N0": N0, "N0_h0sq": N0 * hn ** 2,
            "m0": m0, "m0_h0": m0 * hn,
            "m0_binding_term": which,
            "m0_terms": {k: round(v, 1) for k, v in terms.items()},
            "asymptotic_rate": ("N0 h0^2 ~ n^0.08; m0 h0 >= n^0.34"
                                if name == "rank" else
                                "N0 ~ n^0.60; m0 >= n^0.60"),
        })
    return rows


def frozen_config_wor(extra=None):
    """The without-replacement runs' own design record.

    Their scope is a property of that experiment, not of the primary grid, and
    recording it separately keeps the primary hash the one the reported cells
    were actually run under.
    """
    cfg = {
        "experiment": "without-replacement robustness, reported in the appendix",
        "kernels": ["rank", "auc", "triplet"],
        "N_GRID": list(WOR_N), "B": list(WOR_B), "R": WOR_R,
        "P_ANCHOR": P_ANCHOR, "B_MAX": B_MAX,
        "PILOT_EXPONENT": PILOT_EXPONENT, "L_ROUND": L_ROUND,
        "HORIZON": "T = floor((n_total/B) log2 n_total)", "scheme": "WOR",
        "note": "B = 1 is omitted: a size-one minibatch drawn without "
                "replacement is the with-replacement draw.",
    }
    if extra:
        cfg.update(extra)
    return cfg


def frozen_config_sweep(extra=None):
    """The budget-ratio experiment's own design, hashed separately.

    It is a separate experiment with a separate schedule, so it carries its own
    frozen record; folding it into the primary one would change that hash and
    invalidate the record of cells already run under it.
    """
    cfg = {
        "experiment": "budget ratio, reported in the appendix",
        "kernel": SWEEP_KERNEL, "n_total": SWEEP_N, "B": SWEEP_B,
        "LAMBDA_TUPLE_SWEEP": list(LAMBDA_TUPLE_SWEEP),
        "R": R_FORMAL, "P_ANCHOR": P_ANCHOR, "B_MAX": B_MAX,
        "PILOT_EXPONENT": PILOT_EXPONENT, "L_ROUND": L_ROUND,
        "scheme": "WR",
        "note": "B T is set directly as round_up(Lambda * n_1); the horizon of "
                "the primary study is the primary numerical schedule instead.",
    }
    if extra:
        cfg.update(extra)
    return cfg


def frozen_config(extra=None):
    """The design as a plain dict, for hashing before the formal runs."""
    cfg = {
        "B_GRID": list(B_GRID), "N_GRID": list(N_GRID), "L_ROUND": L_ROUND,
        "HORIZON": "T = floor((n_total/B) log2 n_total)",
        "P_ANCHOR": P_ANCHOR, "B_MAX": B_MAX,
        "PILOT_EXPONENT": PILOT_EXPONENT,
        "BANDWIDTH_EXPONENT": BANDWIDTH_EXPONENT,
        "R_FORMAL": R_FORMAL,
        "R_BY_N": {str(k): v for k, v in R_BY_N.items()},
        "WOR": {"n": WOR_N, "B": list(WOR_B), "R": WOR_R},
        "scheme_primary": "WR",
        "WOR_B_note": "B=1 omitted: a size-one minibatch without replacement "
                      "is the with-replacement draw",
        "note": "Lambda = BT/n_1 is the tuple-budget ratio; lambda = T/n_1. "
                "L_ROUND is an integer-rounding device, not a tuning constant.",
    }
    if extra:
        cfg.update(extra)
    return cfg
