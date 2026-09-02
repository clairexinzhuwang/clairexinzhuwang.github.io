"""30-day readmission AUC analysis with Algorithm 1 (frozen, code/alg_paper.py).

Replaces the Taiwan-credit analysis of the earlier draft.  The estimand is the
coefficient vector of a linear risk score fitted by AUC maximisation on the
training split; the paper's Wald intervals for those coefficients are the object
of interest, and the held-out AUC with its DeLong interval is the external
check.

What is reused from the old script: the split protocol, the bootstrap
comparison, the robustness splits, the DeLong benchmark.
What is replaced: the fit itself.  The old code ran a same-batch running Hessian
with a fixed ridge, refreshed geometrically from t = 1, and estimated the
first-projection covariances with in-run half-stream accumulation.  Algorithm 1
as the paper now defines it is: pilot on N_0 = B t_0 tuples -> fixed gain
D_0 = S_n(A_0) from m_0 fresh tuples -> 1/t recursion on the original clock ->
post-run A_T, V_T from m fresh tuples -> anchor estimator for zeta_k.

Usage:
  python3 -u diabetes_auc_analysis.py [--bootstrap 300] [--splits 100] [--B 256]
"""
import argparse, json, os, sys, time

# Single-threaded BLAS, set BEFORE numpy is imported.  The recursion is tens of
# thousands of small (B x p) products; threaded BLAS spends more on
# synchronisation than it saves (measured 1.75x slower at 4 threads), and the
# reported timings are only meaningful if one fit uses one core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from alg_paper import run_algorithm1, default_t0, default_m   # noqa: E402
from kernels import AUCKernel                                  # noqa: E402

DESIGN = os.path.join(HERE, "data", "diabetes_design.npz")
OUTDIR = os.path.join(HERE, "results")
PRIMARY_SPLIT_SEED = 20260817
FIT_SEED = 50001
K_ANCHOR = 16          # conditional tuples per anchor stream, as in the simulations


def load_design():
    d = np.load(DESIGN, allow_pickle=True)
    return d["X"].astype(float), d["y"].astype(int), [str(s) for s in d["names"]]


def split(X, y, seed, frac=0.6):
    """Stratified train/test split, so the positive fraction matches in both."""
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1); idx_neg = np.flatnonzero(y == 0)
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    ntr_p = int(round(frac * len(idx_pos))); ntr_n = int(round(frac * len(idx_neg)))
    tr = np.concatenate([idx_pos[:ntr_p], idx_neg[:ntr_n]])
    te = np.concatenate([idx_pos[ntr_p:], idx_neg[ntr_n:]])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


def standardise(Xtr, Xte):
    """Training-split statistics only."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def fit(Xtr, ytr, B, seed, mult=1.0, want_zeta=True):
    """One Algorithm-1 fit on a training split, at the schedule the paper
    prescribes and the simulations use: T = (n/B) log2 n, so BT = n log2 n and
    lambda = BT/n_1 = log2 n.  `mult` exists only for the sensitivity check
    reported in the supplement and is 1 for every headline number."""
    Xp = Xtr[ytr == 1]; Xn = Xtr[ytr == 0]
    kern = AUCKernel(Xp, Xn)
    n_ref = kern.n_ref
    T = int(mult * (n_ref / B) * np.log2(n_ref))
    t0 = default_t0(T, rule="log")
    m0 = m = default_m(B, n_ref)
    q = [kern.m, kern.nn]
    return kern, run_algorithm1(kern, np.zeros(Xtr.shape[1]), B, T, t0, m0, m, q,
                                scheme="WR", rng=np.random.default_rng(seed),
                                want_zeta=want_zeta, k_per_stream=K_ANCHOR), T, t0, m0, m


def auc_of(scores_pos, scores_neg, chunk=4096):
    """Mann-Whitney AUC with ties counted as 1/2, chunked over positives."""
    tot = 0.0
    s_neg = np.sort(scores_neg)
    for a in range(0, len(scores_pos), chunk):
        sp = scores_pos[a:a + chunk]
        lo = np.searchsorted(s_neg, sp, side="left")
        hi = np.searchsorted(s_neg, sp, side="right")
        tot += (lo + hi).sum() / 2.0
    return tot / (len(scores_pos) * len(scores_neg))


def delong_var(scores_pos, scores_neg):
    """DeLong's two-stratum variance of the empirical AUC (Sen/Hanley form)."""
    m, n = len(scores_pos), len(scores_neg)
    s_neg = np.sort(scores_neg); s_pos = np.sort(scores_pos)
    v10 = (np.searchsorted(s_neg, scores_pos, "left") +
           np.searchsorted(s_neg, scores_pos, "right")) / (2.0 * n)
    v01 = 1.0 - (np.searchsorted(s_pos, scores_neg, "left") +
                 np.searchsorted(s_pos, scores_neg, "right")) / (2.0 * m)
    return v10.var(ddof=1) / m + v01.var(ddof=1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--out", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, names = load_design()
    print(f"design: n={len(y):,}  p={X.shape[1]}  positives={y.sum():,} ({y.mean():.2%})", flush=True)

    # ---------------- primary split ----------------
    tr, te = split(X, y, PRIMARY_SPLIT_SEED)
    Xtr, Xte, mu, sd = standardise(X[tr], X[te]); ytr, yte = y[tr], y[te]
    npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
    print(f"train {len(tr):,} ({npos:,} pos)   test {len(te):,} ({int(yte.sum()):,} pos)"
          f"   training pairs {npos*nneg:,}", flush=True)

    t_start = time.time()
    kern, res, T, t0, m0, m = fit(Xtr, ytr, args.B, FIT_SEED, args.mult)
    fit_s = time.time() - t_start
    theta = res["theta"]; se = np.sqrt(np.maximum(np.diag(res["Sigma_full"]), 0))
    print(f"fit: B={args.B} T={T:,} t0={t0} N0={args.B*t0:,} m0=m={m0:,} q=(n+,n-)  "
          f"{fit_s:.1f}s   pilot_ok={res['pilot_ok']} iters={res['pilot_iters']}", flush=True)
    print(f"     floor active: A0={res['A0_floor_active']} AT={res['AT_floor_active']}   "
          f"kappa(A_T)={res['AT_kappa']:.1f}   PSD projection: "
          f"k0={res.get('zeta_projected_k0')} k1={res.get('zeta_projected_k1')}", flush=True)

    # ---------------- held-out AUC vs DeLong ----------------
    s_te = Xte @ theta
    auc = auc_of(s_te[yte == 1], s_te[yte == 0])
    v_del = delong_var(s_te[yte == 1], s_te[yte == 0])
    lo, hi = auc - 1.96 * np.sqrt(v_del), auc + 1.96 * np.sqrt(v_del)
    print(f"held-out AUC {auc:.4f}  DeLong 95% CI [{lo:.4f}, {hi:.4f}]", flush=True)

    # ---------------- bootstrap SE for the coefficients ----------------
    print(f"bootstrap: {args.bootstrap} stratified refits", flush=True)
    boots = []
    t_b = time.time()
    for b in range(args.bootstrap):
        rb = np.random.default_rng(70000 + b)
        ip = rb.integers(0, npos, npos); jn = rb.integers(0, nneg, nneg)
        Xb = np.vstack([Xtr[ytr == 1][ip], Xtr[ytr == 0][jn]])
        yb = np.concatenate([np.ones(npos, int), np.zeros(nneg, int)])
        _, rb_res, *_ = fit(Xb, yb, args.B, 90000 + b, args.mult, want_zeta=False)
        boots.append(rb_res["theta"])
        if (b + 1) % 25 == 0:
            print(f"  {b+1}/{args.bootstrap}  ({(time.time()-t_b)/(b+1):.1f}s/refit)", flush=True)
    boots = np.array(boots)
    se_boot = boots.std(0, ddof=1)
    ratio = se / se_boot
    print(f"online/bootstrap SE ratio: median {np.median(ratio):.3f}  "
          f"IQR [{np.percentile(ratio,25):.3f}, {np.percentile(ratio,75):.3f}]", flush=True)
    nsig = int((np.abs(theta) > 1.96 * se).sum())
    print(f"coefficients with 95% interval excluding zero: {nsig}/{len(theta)}", flush=True)

    # ---------------- robustness over splits ----------------
    print(f"robustness: {args.splits} independent splits", flush=True)
    aucs = []
    for s in range(args.splits):
        tr2, te2 = split(X, y, 80000 + s)
        Xt2, Xe2, *_ = standardise(X[tr2], X[te2])
        _, r2, *_ = fit(Xt2, y[tr2], args.B, 91000 + s, args.mult, want_zeta=False)
        s2 = Xe2 @ r2["theta"]
        aucs.append(auc_of(s2[y[te2] == 1], s2[y[te2] == 0]))
        if (s + 1) % 25 == 0:
            print(f"  {s+1}/{args.splits}", flush=True)
    aucs = np.array(aucs)

    out = {
        "dataset": "UCI 296 diabetes 130-US hospitals, first encounter per patient",
        "n": int(len(y)), "p": int(X.shape[1]), "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "split": {"seed": PRIMARY_SPLIT_SEED, "n_train": int(len(tr)), "n_test": int(len(te)),
                  "train_pairs": npos * nneg},
        "algorithm": {"B": args.B, "T": T, "t0": t0, "N0": args.B * t0, "m0": m0, "m": m,
                      "k_anchor": K_ANCHOR, "q": "n_k per sample", "mult": args.mult},
        "fit_seconds": fit_s,
        "diagnostics": {k: res[k] for k in res if isinstance(res[k], (int, float, bool, str))},
        "auc_heldout": float(auc), "delong_var": float(v_del),
        "delong_ci": [float(lo), float(hi)],
        "coef": theta.tolist(), "se_online": se.tolist(), "se_bootstrap": se_boot.tolist(),
        # The two covariance components separately, not only their sum.  The
        # manuscript quotes the stochastic-optimization share of the plug-in variance, and
        # a run that keeps se_online alone leaves that number unverifiable.
        "var_data": np.diag(res["Sigma_dat"]).tolist(),
        "var_alg": np.diag(res["alg_full"]).tolist(),
        "alg_share": float(np.mean(np.diag(res["alg_full"]) /
                                   (np.diag(res["Sigma_dat"]) + np.diag(res["alg_full"])))),
        "se_ratio_median": float(np.median(ratio)),
        "se_ratio_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
        "n_significant": nsig, "names": names,
        "robustness_auc_mean": float(aucs.mean()), "robustness_auc_sd": float(aucs.std(ddof=1)),
        "robustness_auc_range": [float(aucs.min()), float(aucs.max())],
    }
    p = os.path.join(args.out, "diabetes_auc_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
