"""Cell-by-cell audit of the fixed-minibatch study.

    python3 -u audit_fixed_B.py

The gate between "the runs finished" and "numbers go into the paper", for the
cells the paper's primary tables and figure are built from. `audit_cells.py` is
the same gate for the archived growing-B grid and reads different files; running
one is not running the other, and the README says which covers what.

It does not summarise. It tries to find a cell that should stop publication.

 C1 completeness    every declared (kernel, n, B, scheme) is present, once
 C2 schedule        T is the prescribed horizon, B t_0 is the pilot budget, and
                    both agree with experiment_design
 C3 provenance      one R, one P, one pair of auxiliary budgets per (kernel, n)
 C4 coverage        |coverage - 0.95| <= 3 Monte Carlo standard errors
 C5 decomposition   empirical variance within 10% of data plus algorithmic
 C6 invariance      at a common n the two variance estimates agree across B,
                    and the preconditioner is identical in every replication
 C7 pilot           the pilot converged in every replication
 C8 conditioning    the eigenvalue floor never acted; condition numbers finite
 C9 projection      the PSD projection is recorded, and where it acted, by how
                    much
 C10 finiteness     every reported covariance is finite
 C11 failures       the failure log is empty
 C12 agreement      the summary rows agree with the replication records they
                    were computed from
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import experiment_design as D

ROOT = os.path.join(P.ROOT, "experiments_fixed_B")
AGG = os.path.join(ROOT, "aggregated_results")
MCSE = np.sqrt(0.95 * 0.05 / D.R_FORMAL)
KERNELS = ("rank", "auc", "triplet")

FAIL, WARN, SKIPPED = [], [], []


def theta_star(kern_name):
    here = os.path.dirname(os.path.abspath(__file__))
    if kern_name == "rank":
        return np.array([0.5, 1.0])
    f = {"auc": "theta_star_auc_2e8.npz",
         "triplet": "theta_star_p4K5_3e8.npz"}[kern_name]
    return np.load(os.path.join(here, f))["theta_star"]


def fail(c, msg):
    FAIL.append(f"{c} {msg}")


def warn(c, msg):
    WARN.append(f"{c} {msg}")


def audit(tag, summary, reps, declared, by_lambda=False):
    """`by_lambda`: the budget-ratio experiment varies the tuple budget at one
    (kernel, n, B), so its cells are keyed by Lambda and its horizon is set from
    the budget rather than from the primary numerical schedule.  Auditing it under the
    primary design's assumptions reports failures that are the audit's error."""
    d = pd.read_csv(summary)
    # C12 needs the per-replication records.  If they are absent the check
    # cannot run, and saying so is the point: a gate that skips a check in
    # silence and then reports success is worse than no gate.
    r = pd.read_parquet(reps) if os.path.exists(reps) else None
    if r is None:
        SKIPPED.append(f"C12 {tag}: {os.path.basename(reps)} is not present, so "
                       f"the summary rows were NOT re-derived from the records")

    # C1 completeness
    def key(x):
        base = (x.kernel, int(x.n_total), int(x.B), x.scheme)
        return base + (round(x.Lambda, 6),) if by_lambda else base
    have = {key(x) for _, x in d.iterrows()}
    for cell in declared:
        if cell not in have:
            fail("C1", f"{tag}: missing cell {cell}")
    if len(have) != len(d):
        fail("C1", f"{tag}: {len(d)} rows for {len(have)} distinct cells")

    for _, x in d.iterrows():
        c = f"{tag} {x.kernel} n={int(x.n_total)} B={int(x.B)} {x.scheme}"
        # C2 schedule
        if by_lambda:
            # the budget is set directly: B T must be the rounded-up target
            want = D.round_up(x.Lambda * x.n1)
            if int(x.B) * int(x["T"]) != want:
                fail("C2", f"{c} Lambda={x.Lambda:g}: BT={int(x.B)*int(x['T'])} "
                           f"but the budget rule gives {want}")
        else:
            T_want = int(np.floor(x.n_total * np.log2(x.n_total) / x.B))
            if int(x["T"]) != T_want:
                fail("C2", f"{c}: T={int(x['T'])} but the schedule gives {T_want}")
        if int(x.B) * int(x.t0) != int(x.Bt0):
            fail("C2", f"{c}: B t_0 = {int(x.B) * int(x.t0)} but Bt0 = {int(x.Bt0)}")
        if int(x.Bt0) != D.pilot_tuple_budget(int(x.n_total)):
            fail("C2", f"{c}: pilot budget {int(x.Bt0)} is not the design's "
                       f"{D.pilot_tuple_budget(int(x.n_total))}")
        # C4 coverage
        if abs(x.cov_full - 0.95) > 3 * MCSE:
            warn("C4", f"{c}: coverage {x.cov_full:.4f} is "
                       f"{abs(x.cov_full - 0.95) / MCSE:.1f} MC se from nominal")
        # C5 decomposition
        if not 0.90 <= x.ratio_emp_pred <= 1.10:
            warn("C5", f"{c}: empirical/predicted = {x.ratio_emp_pred:.3f}")
        # C7 pilot
        if x.pilot_ok_frac != 1.0:
            fail("C7", f"{c}: pilot converged in {x.pilot_ok_frac:.3f} of replications")
        # C8 conditioning
        if x.floor_frac > 0:
            warn("C8", f"{c}: eigenvalue floor acted in {x.floor_frac:.3f}")
        if not np.isfinite(x.kappa_max):
            fail("C8", f"{c}: condition number not finite")
        # C9 projection
        if x.zeta_projected_frac > 0:
            warn("C9", f"{c}: PSD projection acted in {x.zeta_projected_frac:.3f}, "
                       f"moving {x.zeta_proj_moved_mean:.2e}")
        # C10 finiteness
        if x.finite_frac != 1.0:
            fail("C10", f"{c}: {1 - x.finite_frac:.3f} of replications non-finite")

    # C3 provenance, and C6 invariance, per (kernel, n, scheme)
    group_on = (["kernel", "n_total", "scheme", "Lambda"] if by_lambda
                else ["kernel", "n_total", "scheme"])
    for gk, g in d.groupby(group_on):
        k, n, sch = gk[0], gk[1], gk[2]
        c = f"{tag} {k} n={int(n)} {sch}"
        for col in ("R", "P", "m0", "m"):
            if g[col].nunique() != 1:
                fail("C3", f"{c}: {col} varies across B: {sorted(g[col].unique())}")
        if len(g) > 1:
            for col in ("v_data", "v_alg"):
                spread = (g[col].max() - g[col].min()) / g[col].mean()
                if spread > 0.02:
                    fail("C6", f"{c}: {col} varies {spread*100:.2f}% across B")
            if not bool(g.D0_same_across_B.iloc[0]):
                fail("C6", f"{c}: the preconditioner is not identical across B")
        elif g.D0_same_across_B.iloc[0] is True or g.D0_same_across_B.iloc[0] == True:
            fail("C6", f"{c}: one minibatch size cannot demonstrate invariance, "
                       f"yet the diagnostic reports True")

    # C12 the summary agrees with the records it came from
    if r is not None:
        for _, x in d.iterrows():
            sub = r[(r.kernel == x.kernel) & (r.n_total == x.n_total) &
                    (r.B == x.B) & (r.scheme == x.scheme)]
            if by_lambda:
                sub = sub[np.isclose(sub.Lambda * sub.n1, x.Lambda * x.n1, rtol=1e-6)]
            if len(sub) != x.R:
                fail("C12", f"{tag} {x.kernel} n={int(x.n_total)} B={int(x.B)}: "
                            f"{len(sub)} records for R={int(x.R)}")
                continue
            v = np.array([np.mean(a) for a in sub.v_dat])
            if not np.isclose(v.mean(), x.v_data, rtol=1e-6):
                fail("C12", f"{tag} {x.kernel} n={int(x.n_total)} B={int(x.B)}: "
                            f"v_data {x.v_data:.6e} != {v.mean():.6e} from the records")
            # Coordinate-specific columns remain in the archive and are
            # audited even though the revised Figure 1 uses coordinate averages.
            if all(c in d.columns for c in (
                    "v_emp_theta1", "v_data_theta1", "v_alg_theta1",
                    "ratio_emp_pred_theta1", "alg_share_theta1",
                    "cov_full_theta1", "cov_data_only_theta1")):
                th = np.vstack(sub.theta.to_numpy())
                vd = np.vstack(sub.v_dat.to_numpy())
                va = np.vstack(sub.v_alg.to_numpy())
                vt = np.vstack(sub.v_tot.to_numpy())
                ts = theta_star(x.kernel)
                expected = {
                    "v_emp_theta1": th[:, 0].var(ddof=1),
                    "v_data_theta1": vd[:, 0].mean(),
                    "v_alg_theta1": va[:, 0].mean(),
                    "ratio_emp_pred_theta1":
                        th[:, 0].var(ddof=1) / (vd[:, 0].mean() + va[:, 0].mean()),
                    "alg_share_theta1":
                        va[:, 0].mean() / (vd[:, 0].mean() + va[:, 0].mean()),
                    "cov_full_theta1":
                        (np.abs(th[:, 0] - ts[0]) <
                         1.96 * np.sqrt(np.maximum(vt[:, 0], 0))).mean(),
                    "cov_data_only_theta1":
                        (np.abs(th[:, 0] - ts[0]) <
                         1.96 * np.sqrt(np.maximum(vd[:, 0], 0))).mean(),
                }
                for col, val in expected.items():
                    if not np.isclose(val, x[col], rtol=1e-6, atol=1e-12):
                        fail("C12", f"{tag} {x.kernel} n={int(x.n_total)} "
                             f"B={int(x.B)}: {col} {x[col]:.6e} != "
                             f"{val:.6e} from the records")
    return len(d)


def main():
    total = 0
    prim = [(k, n, B, "WR") for k in KERNELS for n in D.N_GRID for B in D.B_GRID]
    total += audit("primary", os.path.join(AGG, "fixed_B_summary.csv"),
                   os.path.join(ROOT, "formal_runs", "fixed_B_replications.parquet"), prim)
    wor = [(k, n, B, "WOR") for k in KERNELS for n in D.WOR_N for B in D.WOR_B]
    total += audit("wor", os.path.join(AGG, "wor_robustness_summary.csv"),
                   os.path.join(ROOT, "wor_robustness", "wor_robustness_replications.parquet"), wor)
    sw = os.path.join(AGG, "budget_ratio_summary.csv")
    if os.path.exists(sw):
        total += audit("budget-ratio", sw,
                       os.path.join(ROOT, "budget_ratio_runs", "budget_ratio_replications.parquet"),
                       [(D.SWEEP_KERNEL, D.SWEEP_N, D.SWEEP_B, "WR", round(lam, 6))
                        for lam in D.LAMBDA_TUPLE_SWEEP], by_lambda=True)

    # C11 failure logs
    for f in ("fixed_B_failure_log.csv", "wor_robustness_failure_log.csv",
              "budget_ratio_failure_log.csv"):
        p = os.path.join(ROOT, "logs", f)
        if os.path.exists(p):
            log = pd.read_csv(p)
            if "error" in log.columns and len(log):
                fail("C11", f"{f}: {len(log)} failures recorded")

    print(f"\n  cells audited: {total}")
    for w in WARN:
        print(f"  warning  {w}")
    for k in SKIPPED:
        print(f"  SKIPPED  {k}")
    if FAIL:
        print()
        for f in FAIL:
            print(f"  FAILURE  {f}")
        print(f"\n  {len(FAIL)} failures. VERDICT: do not write these numbers into the paper.")
        sys.exit(1)
    print(f"  {len(WARN)} warnings, listed above; none is a failure.")
    if SKIPPED:
        print(f"\n  {len(SKIPPED)} CHECK(S) COULD NOT RUN, listed above. This archive "
              f"does not carry the per-replication records, so agreement between the "
              f"summaries and the records behind them was not re-verified here; run "
              f"this gate in the full replication archive to close C12.")
        print("\n  VERDICT: the checks that could run passed; C12 did not run.")
        sys.exit(2)
    print("\n  VERDICT: every check passed; the fixed-B numbers may be written.")


if __name__ == "__main__":
    main()
