"""The fixed-minibatch study: B in {1, 10, 100} at a fixed tuple budget.

    python3 -u run_fixed_B_grid.py primary [R_scale] [cpus]
    python3 -u run_fixed_B_grid.py sweep   [R_scale] [cpus]
    python3 -u run_fixed_B_grid.py wor     [R_scale] [cpus]
    python3 -u run_fixed_B_grid.py smoke   [R]       [cpus]
    python3 -u run_fixed_B_grid.py bench   [R]       [cpus]

The archived growing-B grid stays where it is: this driver never touches
run_grid.py's files.  Everything it writes goes under experiments_fixed_B/.

What makes the comparison across B a paired one.  Within one replication the
three minibatch sizes share the dataset, the tuple budget M = BT, the pilot
budget N_0 = B t_0, the auxiliary budgets m_0, m, q_k, P, and the tuple draws
themselves: the same pilot tuples, the same initial-Hessian tuples, the same
ordered recursion stream cut into blocks of B, the same terminal tuples and the
same anchors.  The pilot estimate and the fixed preconditioner D_0 are therefore
identical across B to the last bit, and every difference that remains is a
difference in how the recursion consumed one fixed stream.

Notation.  Lambda = BT/n_1 is the tuple-budget ratio held fixed here;
lambda = T/n_1 is the manuscript's root-T parameter, recorded per cell as
Lambda/B.  n_1 is the theorem's reference sample size, which for two-sample AUC
is the positive sample, not the total.
"""
import sys, os, json, time, hashlib, platform, subprocess
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import experiment_design as D
import run_grid as G
from alg_paper import run_algorithm1, default_m, component_streams, PHASES

OUT_ROOT = os.path.join(P.ROOT, "experiments_fixed_B")
SUBDIRS = ("smoke_tests", "formal_runs", "budget_ratio_runs", "wor_robustness",
           "aggregated_results", "figures", "tables", "logs", "diagnostics")

# Seed bases, far from the archived grid's 1000+/90000+ and the probes' 50000+.
DATA_SEED, ALG_SEED = 700_000, 800_000
KERNELS = ("rank", "auc", "triplet")


# --------------------------------------------------------------------------
# one replication: the same dataset and the same draws under every B
# --------------------------------------------------------------------------
def theta_star(kern_name):
    here = os.path.dirname(os.path.abspath(__file__))
    if kern_name == "rank":
        return np.array([0.5, 1.0])
    f = {"auc": "theta_star_auc_2e8.npz", "triplet": "theta_star_p4K5_3e8.npz"}[kern_name]
    return np.load(os.path.join(here, f))["theta_star"]


def replicate(kern_name, n_target, B_list, lam, scheme, rep):
    """Run every B on one dataset.  Returns one record per B, plus failures."""
    kern, theta0, _q_default, _rng, _rule = G.build(kern_name, n_target, DATA_SEED + rep)
    n_total, n1 = kern.n_total, kern.n_theory_ref
    h_eff = getattr(kern, "h", 1.0)
    m0, m = D.auxiliary_budgets(default_m, n_total, h_eff)   # once, at B_MAX
    q_k = D.anchor_counts(kern)                              # once
    recs, fails = [], []
    # The primary numerical schedule leaves B T equal across B only up to the integer
    # part, so the longest recursion stream is drawn once and the shorter runs
    # take prefixes of it; every B then consumes the same tuples in the same
    # order for as long as its budget lasts.
    stream_len = max(B * (D.schedule(n_total, n1, B, lam)[2]
                          - D.schedule(n_total, n1, B, lam)[3]) for B in B_list)
    for B in B_list:
        try:
            M, N0, T, t0 = D.schedule(n_total, n1, B, lam)
            o = run_algorithm1(kern, theta0, B, T, t0, m0, m, q_k,
                               scheme=scheme, k_per_stream=D.P_ANCHOR,
                               streams=component_streams(ALG_SEED + rep),
                               stream_len=stream_len)
            recs.append({
                "kernel": kern_name, "n_total": n_total, "n1": n1, "B": B, "rep": rep,
                "scheme": scheme, "Lambda": (B * T) / n1, "lambda_B": T / n1,
                "M": M, "N0": N0, "T": T, "t0": t0, "steps": T - t0,
                "m0": m0, "m": m, "q": q_k, "P": D.P_ANCHOR,
                "theta": np.asarray(o["theta"]).tolist(),
                "v_dat": np.diag(o["Sigma_dat"]).tolist(),
                "v_alg": np.diag(o["alg_full"]).tolist(),
                "v_tot": np.diag(o["Sigma_full"]).tolist(),
                "v_legacy_main": np.diag(o["Sigma_main"]).tolist(),
                "D0_hash": hashlib.sha256(np.ascontiguousarray(o["D0"]).tobytes()).hexdigest()[:16],
                "pilot_ok": bool(o["pilot_ok"]), "pilot_iters": int(o["pilot_iters"]),
                "pilot_gradnorm": float(o["pilot_gradnorm"]),
                "A0_floor": bool(o["A0_floor_active"]), "AT_floor": bool(o["AT_floor_active"]),
                "A0_kappa": float(o["A0_kappa"]), "AT_kappa": float(o["AT_kappa"]),
                "zeta_projected": float(np.mean([o.get(f"zeta_projected_k{i}", False)
                                                 for i in range(len(kern.samples))])),
                "zeta_proj_moved": float(np.mean([o.get(f"zeta_proj_moved_k{i}", 0.0)
                                                  for i in range(len(kern.samples))])),
                "finite": bool(np.all(np.isfinite(o["Sigma_full"])) and
                               np.all(np.isfinite(o["Sigma_dat"]))),
                **{f"t_{ph}": o["t_phase"][ph] for ph in PHASES},
            })
        except Exception as exc:                               # logged, never silent
            fails.append({"kernel": kern_name, "n_total": n_target, "B": B, "rep": rep,
                          "scheme": scheme, "Lambda": lam,
                          "error": f"{type(exc).__name__}: {exc}"})
    return recs, fails


@ray.remote
def replicate_remote(*a):
    return replicate(*a)


# --------------------------------------------------------------------------
# aggregation, exactly the quantities the manuscript reports
# --------------------------------------------------------------------------
def aggregate(recs, ts_by_kernel):
    """One row per (kernel, n, B, scheme, Lambda).

    On what the reported scalars are.  The estimators whose consistency is
    proved are

        zeta_hat_{n,k} -> zeta_k,   V_hat_{n,T} -> V,   D_hat_{n,T} -> A^{-1},

    and those are matrix statements about the estimators themselves.
    zeta_hat_{n,k} is what Sigma_data is built from and V_hat_{n,T} is what
    Sigma_alg is built from; neither converges to a scalar and neither is
    "v_data" or "v_alg".  The columns v_data and v_alg below are simulation
    summaries: the coordinate average of the diagonal of Sigma_data and of
    Sigma_alg, averaged over replications.  They exist to be compared with
    v_emp, the coordinate average of the Monte Carlo variance of the estimates,
    and carry no claim beyond that.
    """
    import collections
    cells = collections.defaultdict(list)
    for r in recs:
        cells[(r["kernel"], r["n_total"], r["B"], r["scheme"], r["Lambda"])].append(r)
    rows = []
    for (kern, n, B, scheme, lam), rs in sorted(cells.items()):
        ts = ts_by_kernel[kern]
        th = np.array([r["theta"] for r in rs])
        vd = np.array([r["v_dat"] for r in rs]); va = np.array([r["v_alg"] for r in rs])
        vt = np.array([r["v_tot"] for r in rs])
        R = len(rs)
        v_emp = th.var(0, ddof=1)                     # per coordinate
        v_dat, v_alg, v_tot = vd.mean(0), va.mean(0), vt.mean(0)
        cov_full = (np.abs(th - ts) < 1.96 * np.sqrt(np.maximum(vt, 0))).mean()
        cov_dat = (np.abs(th - ts) < 1.96 * np.sqrt(np.maximum(vd, 0))).mean()
        pred = v_dat + v_alg
        rows.append({
            "kernel": kern, "n_total": n, "n1": rs[0]["n1"], "B": B, "scheme": scheme,
            "Lambda": lam, "lambda_B": rs[0]["lambda_B"], "R": R,
            "T": rs[0]["T"], "t0": rs[0]["t0"], "steps": rs[0]["steps"],
            "BT": rs[0]["M"], "Bt0": rs[0]["N0"],
            "m0": rs[0]["m0"], "m": rs[0]["m"], "P": rs[0]["P"],
            # coordinate averages of matrix diagonals; see the docstring
            "v_emp": float(v_emp.mean()), "v_data": float(v_dat.mean()),
            "v_alg": float(v_alg.mean()), "v_pred": float(pred.mean()),
            # the coordinate average of the per-coordinate ratios, which is not
            # the ratio of the two coordinate averages; the manuscript names it
            "ratio_emp_pred": float(np.mean(v_emp / pred)),
            "alg_share": float(np.mean(v_alg / pred)),
            "bias_abs": float(np.abs(th.mean(0) - ts).mean()),
            "cov_full": float(cov_full), "cov_data_only": float(cov_dat),
            # Pre-specified first-coordinate summaries used only in Figure 1.
            # The tables retain the coordinate-averaged summaries above.
            "v_emp_theta1": float(v_emp[0]),
            "v_data_theta1": float(v_dat[0]),
            "v_alg_theta1": float(v_alg[0]),
            "v_pred_theta1": float(pred[0]),
            "ratio_emp_pred_theta1": float(v_emp[0] / pred[0]),
            "alg_share_theta1": float(v_alg[0] / pred[0]),
            "cov_full_theta1": float(
                (np.abs(th[:, 0] - ts[0]) <
                 1.96 * np.sqrt(np.maximum(vt[:, 0], 0))).mean()),
            "cov_data_only_theta1": float(
                (np.abs(th[:, 0] - ts[0]) <
                 1.96 * np.sqrt(np.maximum(vd[:, 0], 0))).mean()),
            "mcse_cov": float(np.sqrt(0.95 * 0.05 / R)),
            "pilot_ok_frac": float(np.mean([r["pilot_ok"] for r in rs])),
            "pilot_gradnorm_max": float(max(r["pilot_gradnorm"] for r in rs)),
            "floor_frac": float(np.mean([r["A0_floor"] or r["AT_floor"] for r in rs])),
            "kappa_max": float(max(max(r["A0_kappa"], r["AT_kappa"]) for r in rs)),
            "zeta_projected_frac": float(np.mean([r["zeta_projected"] for r in rs])),
            "zeta_proj_moved_mean": float(np.mean([r["zeta_proj_moved"] for r in rs])),
            "finite_frac": float(np.mean([r["finite"] for r in rs])),
            "D0_same_across_B": None,                 # filled in by the caller
            **{f"t_{ph}": float(np.mean([r[f"t_{ph}"] for r in rs])) for ph in PHASES},
        })
    return rows


def d0_invariance(recs):
    """Did every B in a replication produce the same preconditioner?

    The grouping key must NOT contain the realized budget ratio.  Under the
    primary numerical schedule the integer part makes B T differ slightly across B, so
    Lambda differs too; keying on it puts every B in a group of its own and the
    check passes on a single hash, verifying nothing.  Key on the replication
    and the design instead, which is what "the same run at a different B" means.
    """
    import collections
    by = collections.defaultdict(dict)
    for r in recs:
        by[(r["kernel"], r["n_total"], r["scheme"], r["rep"])][r["B"]] = r["D0_hash"]
    agree = {}
    for key, hs in by.items():
        # a group with one B cannot demonstrate invariance; record it as such
        # rather than as a pass
        agree.setdefault(key[:3] + (None,), []).append(
            len(hs) > 1 and len(set(hs.values())) == 1)
    return {k: bool(np.all(v)) for k, v in agree.items()}


# --------------------------------------------------------------------------
# output plumbing
# --------------------------------------------------------------------------
def ensure_dirs():
    for d in ("",) + SUBDIRS:
        os.makedirs(os.path.join(OUT_ROOT, d), exist_ok=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", P.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def freeze_config(tag=""):
    """Write a design and its hash before the runs it governs, once.

    The budget-ratio experiment carries its own record: it is a separate design
    with a separate schedule, and folding it into the primary one would change
    that hash and invalidate the record of the cells already run under it.
    """
    ensure_dirs()
    suffix = f"_{tag}" if tag else ""
    cfg_path = os.path.join(OUT_ROOT, f"frozen_config{suffix}.json")
    extra = {"data_seed_base": DATA_SEED, "alg_seed_base": ALG_SEED}
    cfg = (D.frozen_config_sweep(extra) if tag == "sweep" else
           D.frozen_config_wor(extra) if tag == "wor" else
           D.frozen_config({"kernels": list(KERNELS), **extra}))
    blob = json.dumps(cfg, indent=1, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    if os.path.exists(cfg_path):
        old = open(cfg_path).read()
        if old != blob:
            sys.exit(f"  frozen_config{suffix}.json differs from the current design; "
                     "the design is frozen -- revert the change, or document a "
                     "genuine bug fix and rerun every affected cell")
    else:
        open(cfg_path, "w").write(blob)
        open(os.path.join(OUT_ROOT, f"frozen_config{suffix}.sha256"), "w").write(digest + "\n")
        open(os.path.join(OUT_ROOT, "environment.txt"), "w").write(
            f"python {platform.python_version()}\nplatform {platform.platform()}\n"
            f"numpy {np.__version__}\nray {ray.__version__}\n")
    open(os.path.join(OUT_ROOT, "git_commit.txt"), "w").write(git_commit() + "\n")
    return digest


def save(recs, fails, rows, tag, subdir):
    import pandas as pd
    ensure_dirs()
    agg = os.path.join(OUT_ROOT, "aggregated_results")
    if recs:
        pd.DataFrame(recs).to_parquet(os.path.join(OUT_ROOT, subdir, f"{tag}_replications.parquet"))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(agg, f"{tag}_summary.csv"), index=False)
        pd.DataFrame(rows)[["kernel", "n_total", "B", "scheme", "Lambda"] +
                           [f"t_{ph}" for ph in PHASES]].to_csv(
            os.path.join(agg, "runtime_summary.csv"), index=False)
        pd.DataFrame(rows)[["kernel", "n_total", "B", "scheme", "Lambda", "kappa_max",
                            "floor_frac", "zeta_projected_frac", "zeta_proj_moved_mean",
                            "pilot_ok_frac", "pilot_gradnorm_max", "finite_frac"]].to_csv(
            os.path.join(OUT_ROOT, "diagnostics", f"{tag}_conditioning.csv"), index=False)
    pd.DataFrame(fails or [{"note": "no failures"}]).to_csv(
        os.path.join(OUT_ROOT, "logs", f"{tag}_failure_log.csv"), index=False)
    print(f"  wrote {tag}: {len(recs)} replication records, {len(rows)} cells, "
          f"{len(fails)} failures", flush=True)


# --------------------------------------------------------------------------
# the three studies
# --------------------------------------------------------------------------
def run_cells(cells, cpus, tag, subdir):
    """cells: list of (kernel, n_target, B_list, Lambda, scheme, R)."""
    ts = {k: theta_star(k) for k in KERNELS}
    ray.init(num_cpus=cpus, include_dashboard=False, ignore_reinit_error=True,
             log_to_driver=False)
    recs, fails = [], []
    for kern, n, B_list, lam, scheme, R in cells:
        t = time.time()
        out = ray.get([replicate_remote.remote(kern, n, B_list, lam, scheme, r)
                       for r in range(R)])
        for rr, ff in out:
            recs += rr; fails += ff
        done = [r for r in recs if r["kernel"] == kern and r["n_total"] == n
                and r["scheme"] == scheme
                and (lam is None or abs(r["Lambda"] * r["n1"] - lam * r["n1"]) < 1)]
        sched = "primary" if lam is None else f"Lambda={lam:g}"
        print(f"  {kern:<8} n={n:<7} B={B_list} {sched} {scheme} R={R}: "
              f"{len(done)} records in {time.time() - t:.0f}s", flush=True)
    rows = aggregate(recs, ts)
    inv = d0_invariance(recs)
    for row in rows:
        row["D0_same_across_B"] = inv.get((row["kernel"], row["n_total"],
                                           row["scheme"], None))
    save(recs, fails, rows, tag, subdir)
    ray.shutdown()
    return recs, rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
    # Exactly the worker count asked for.  It is an execution detail: each
    # replication is seeded by its index, so the results do not depend on how
    # many run at once, and the worker count is not part of the frozen design.
    cpus = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    digest = freeze_config(mode if mode in ("sweep", "wor") else "")
    print(f"=== fixed-B study: mode={mode} cpus={cpus} config={digest[:12]} "
          f"commit={git_commit()[:12]} ===", flush=True)

    if mode == "smoke":
        R = int(arg or 20)
        cells = [(k, 1000, list(D.B_GRID), None, "WR", R) for k in KERNELS]
        run_cells(cells, cpus, "smoke", "smoke_tests")

    elif mode == "bench":
        R = int(arg or 20)
        cells = [(k, 100_000, [1], None, "WR", R) for k in KERNELS]
        _, rows = run_cells(cells, cpus, "bench", "smoke_tests")
        print("\n  projected wall-clock for the n=1e5, B=1 formal cells:")
        for r in rows:
            per = r["t_total"]
            R_formal = D.R_BY_N[100_000]
            print(f"    {r['kernel']:<8} {per:7.1f} s/replicate -> "
                  f"{per * R_formal / 3600:6.1f} core-hours for R={R_formal} "
                  f"({per * R_formal / cpus / 3600:5.1f} h on {cpus} workers)")

    elif mode == "primary":
        scale = arg or 1.0
        cells = [(k, n, list(D.B_GRID), None, "WR",
                  max(1, int(D.R_BY_N[n] * scale)))
                 for n in D.N_GRID for k in KERNELS]
        run_cells(cells, cpus, "fixed_B", "formal_runs")

    elif mode == "sweep":
        scale = arg or 1.0
        R = max(1, int(D.R_FORMAL * scale))
        cells = [(D.SWEEP_KERNEL, D.SWEEP_N, [D.SWEEP_B], lam, "WR", R)
                 for lam in D.LAMBDA_TUPLE_SWEEP]
        run_cells(cells, cpus, "budget_ratio", "budget_ratio_runs")

    elif mode == "wor":
        scale = arg or 1.0
        R = max(1, int(D.WOR_R * scale))
        cells = [(k, n, list(D.WOR_B), None, "WOR", R)
                 for n in D.WOR_N for k in KERNELS]
        run_cells(cells, cpus, "wor_robustness", "wor_robustness")

    else:
        sys.exit(f"  unknown mode {mode!r}")


if __name__ == "__main__":
    main()
