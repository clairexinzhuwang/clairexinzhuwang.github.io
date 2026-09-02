"""Sensitivity of the reported standard errors to the post-run budgets.

    python3 -u sensitivity_post_run.py [draws] [cpus]

The partner count P and the terminal sample size m are Monte Carlo precision
budgets, not model parameters, so the question is how much the reported standard
errors move when they change, not how coverage responds.

Design of the check. One dataset and one final iterate are held fixed per
kernel, at a cell of the reported grid, and only the POST-RUN sampling is
repeated: the terminal tuples, the anchors and their partners are redrawn
`draws` times at each setting, while the data, the pilot, the preconditioner and
the recursion stay exactly as the reported run left them. Everything that moves
between two settings is therefore the post-run budget and nothing else.

q_k is held fixed, so raising P raises the total number of partner draws. This
is a numerical-stability check, not a statement about how a fixed budget is best
divided between P and q_k.

Writes experiments_fixed_B/aggregated_results/sensitivity_post_run.json.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import experiment_design as D
import run_grid as G
from alg_paper import (run_algorithm1, default_m, component_streams,
                       anchor_zeta, stabilized_inverse, psd_project)

OUT = os.path.join(P.ROOT, "experiments_fixed_B", "aggregated_results",
                   "sensitivity_post_run.json")

CELL_N, CELL_B = 10_000, 10          # a cell of the reported grid
P_GRID = (1, 4, 16, 32)
M_FRACTIONS = (1.0, 0.5, 0.25)
KERNELS = ("rank", "auc", "triplet")
SEED_DATA, SEED_ALG, SEED_POST = 700_000, 800_000, 950_000


def reported_run(kern_name):
    """The run whose terminal iterate the sensitivity check freezes."""
    kern, theta0, _q, _r, _rule = G.build(kern_name, CELL_N, SEED_DATA)
    n_total, n1 = kern.n_total, kern.n_theory_ref
    h_eff = getattr(kern, "h", 1.0)
    m0, m = D.auxiliary_budgets(default_m, n_total, h_eff)
    q_k = D.anchor_counts(kern)
    _BT, _N0, T, t0 = D.schedule(n_total, n1, CELL_B)
    out = run_algorithm1(kern, theta0, CELL_B, T, t0, m0, m, q_k, scheme="WR",
                         k_per_stream=D.P_ANCHOR,
                         streams=component_streams(SEED_ALG))
    return kern, out, m, q_k, n1


def post_run_once(kern, theta, m_use, q_k, n1, P_use, rng):
    """Redraw the terminal and anchor samples only, and rebuild the interval."""
    eps = 1.0 / kern.n_ref
    _, _, AT, ST = kern.stats(kern.draw(m_use, rng), theta)
    DT = stabilized_inverse(AT, eps)
    VT = np.cov(ST, rowvar=False, bias=False)
    p = len(theta)
    data_term = np.zeros((p, p))
    projected = False
    for kidx, (n_k, d_k) in enumerate(kern.samples):
        Zk, dz = anchor_zeta(kern, kidx, theta, int(q_k[kidx]), rng, P_use)
        projected |= bool(dz["zeta_projected"])
        data_term += (d_k ** 2) * Zk / n_k
    Sigma = DT @ data_term @ DT.T
    return np.sqrt(np.maximum(np.diag(Sigma), 0)), projected


def unprojected_negative(kern, theta, q_k, P_use, rng):
    """Did the raw cross-covariance leave the positive semidefinite cone?"""
    neg = False
    for kidx, _ in enumerate(kern.samples):
        Y, Z = kern.anchor_streams(kidx, rng.integers(0, kern.samples[kidx][0],
                                                      int(q_k[kidx])),
                                   rng, theta, P_use)
        Yc, Zc = Y - Y.mean(0), Z - Z.mean(0)
        C = (Yc.T @ Zc) / (len(Y) - 1)
        C = (C + C.T) / 2
        neg |= float(np.linalg.eigvalsh(C)[0]) < 0
    return neg


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    res = {"cell": {"n_total": CELL_N, "B": CELL_B, "draws": draws,
                    "P_grid": list(P_GRID), "m_fractions": list(M_FRACTIONS)},
           "kernels": {}}
    for kern_name in KERNELS:
        t0 = time.time()
        kern, run, m, q_k, n1 = reported_run(kern_name)
        theta = run["theta"]
        rec = {"m": int(m), "q": [int(x) for x in q_k], "P_reported": D.P_ANCHOR,
               "by_P": {}, "by_m": {}}
        for P_use in P_GRID:
            ses, negs, projs = [], 0, 0
            for r in range(draws):
                rng = np.random.default_rng(SEED_POST + 1000 * P_use + r)
                se, proj = post_run_once(kern, theta, m, q_k, n1, P_use, rng)
                ses.append(se); projs += proj
                negs += unprojected_negative(
                    kern, theta, q_k, P_use,
                    np.random.default_rng(SEED_POST + 7 + 1000 * P_use + r))
            ses = np.array(ses)
            rec["by_P"][str(P_use)] = {
                "se_mean": ses.mean(0).tolist(),
                "se_sd": ses.std(0, ddof=1).tolist(),
                "unprojected_negative": int(negs), "projection_acted": int(projs)}
        for frac in M_FRACTIONS:
            m_use = max(2, int(round(m * frac)))
            ses = []
            for r in range(draws):
                rng = np.random.default_rng(SEED_POST + 500_000 + r)
                se, _ = post_run_once(kern, theta, m_use, q_k, n1, D.P_ANCHOR, rng)
                ses.append(se)
            ses = np.array(ses)
            rec["by_m"][f"{frac:g}"] = {"m": m_use, "se_mean": ses.mean(0).tolist(),
                                        "se_sd": ses.std(0, ddof=1).tolist()}
        rec["wall_s"] = time.time() - t0
        res["kernels"][kern_name] = rec
        base = np.array(rec["by_P"][str(D.P_ANCHOR)]["se_mean"])
        print(f"  {kern_name:<8} m={m:<6} q={q_k}", flush=True)
        for P_use in P_GRID:
            e = rec["by_P"][str(P_use)]
            ch = 100 * np.max(np.abs(np.array(e["se_mean"]) - base) / base)
            print(f"    P={P_use:<3} se[0]={e['se_mean'][0]:.5f}  "
                  f"spread={np.mean(e['se_sd']):.5f}  "
                  f"unprojected negative in {e['unprojected_negative']}/{draws}  "
                  f"max change vs P={D.P_ANCHOR}: {ch:.2f}%", flush=True)
        for frac in M_FRACTIONS:
            e = rec["by_m"][f"{frac:g}"]
            ch = 100 * np.max(np.abs(np.array(e["se_mean"]) - base) / base)
            print(f"    m x{frac:<5g} ({e['m']:>6}) se[0]={e['se_mean'][0]:.5f}  "
                  f"spread={np.mean(e['se_sd']):.5f}  max change: {ch:.2f}%", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
