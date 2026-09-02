"""freMTPL2 claim-severity rank regression with Algorithm 1 (frozen).

Appendix analysis: a variance sanity check for the smoothed rank kernel on a
heavy-tailed response, the object the kernel is designed for.  The response is
the total claim amount per policy; least squares on such a response is
dominated by a handful of extreme claims, which is the standard motivation for
rank regression.

Design is unchanged from the earlier draft (same policy aggregation, same
covariates, same standardisation, verified by the SHA-256 of the aggregated
CSV).  What is replaced is the fit: the earlier script ran a same-batch running
Hessian with an absolute eigenvalue floor and refreshed geometrically from
t = 1; Algorithm 1 as the paper now defines it is pilot -> fixed gain ->
1/t recursion on the original clock -> post-run A_T, V_T -> anchor estimator.

Two things about the rank kernel need care here and are handled explicitly:

* BANDWIDTH SCALE.  h_n = sigma_hat n^{-beta} needs a residual-scale estimate
  from a preliminary consistent fit.  On this response the OLS residual scale
  used in the simulations is itself dominated by the extreme claims, so we take
  sigma_hat from a Huber pre-fit (as the earlier script did) and pass it to the
  kernel rather than letting the kernel compute an OLS scale.

* WHAT IS COMPARED.  theta is identified up to the response scale, so the
  reported comparison is the ONLINE standard error against a policy-level
  bootstrap standard error, coefficient by coefficient; there is no ground
  truth, so we report interval agreement, not coverage.

Usage:
  python3 -u fremtpl2_rank_analysis.py [--bootstrap 100] [--B 256] [--beta 0.26]
"""
import argparse, json, os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from alg_paper import run_algorithm1, default_t0, default_m   # noqa: E402
from kernels import RankKernel                                 # noqa: E402

POLICY_CSV = os.path.join(HERE, "data", "fremtpl2_policy.csv")
OUTDIR = os.path.join(HERE, "results")
CATEGORICAL = ("Area", "VehBrand", "VehGas", "Region")
FIT_SEED = 50011
K_ANCHOR = 16


def robust_mad(v):
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def load_design():
    df = pd.read_csv(POLICY_CSV)
    # the earlier draft's filters: positive exposure and a positive claim total
    df = df[(df["Exposure"] > 0) & (df["TotalClaim"] > 0)].reset_index(drop=True)
    num = pd.DataFrame({
        "log_exposure": np.log(df["Exposure"].to_numpy(float)),
        "veh_power": df["VehPower"].to_numpy(float),
        "log1p_veh_age": np.log1p(df["VehAge"].to_numpy(float)),
        "driver_age": df["DrivAge"].to_numpy(float),
        "bonus_malus": df["BonusMalus"].to_numpy(float),
        "log1p_density": np.log1p(df["Density"].to_numpy(float)),
    })
    cat = df[list(CATEGORICAL)].astype(str)
    dummies = pd.get_dummies(cat, drop_first=True, dtype=float)
    keep = dummies.columns[(dummies.mean() >= 0.02) & (dummies.mean() <= 0.98)]
    design = pd.concat([num, dummies[keep]], axis=1)
    names = design.columns.tolist()
    Z = design.to_numpy(float)
    mu, sd = Z.mean(0), Z.std(0)
    sd[sd < 1e-12] = 1.0
    X = (Z - mu) / sd
    y = df["TotalClaim"].to_numpy(float)
    return X, y, names, df


def huber_scale(X, y):
    """Residual scale from a Huber pre-fit: the OLS scale is dominated by the
    extreme claims this kernel exists to handle."""
    from sklearn.linear_model import HuberRegressor
    y0 = np.median(y); s0 = robust_mad(y)
    mdl = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True,
                         max_iter=2000, tol=1e-8).fit(X, (y - y0) / s0)
    coef = mdl.coef_ * s0
    resid = y - (mdl.intercept_ * s0 + y0 + X @ coef)
    return robust_mad(resid), coef


def fit(X, y, B, seed, beta=0.26, mult=1.0, sigma=None, want_zeta=True):
    kern = RankKernel(X, y, beta=beta)
    if sigma is not None:                      # override the OLS-based bandwidth
        kern.h = float(sigma * len(y) ** (-beta))
    n = len(y)
    T = int(mult * (n / B) * np.log2(n))
    t0 = default_t0(T, rule="rank", n=n)
    m0 = m = default_m(B, n, kern.h)
    return kern, run_algorithm1(kern, kern.theta_ols, B, T, t0, m0, m, n,
                                scheme="WR", rng=np.random.default_rng(seed),
                                want_zeta=want_zeta, k_per_stream=K_ANCHOR), T, t0, m0, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.26)
    ap.add_argument("--mult", type=float, default=1.0)
    # mult scales the primary numerical schedule T = (n/B) log2 n and is 1 for every
    # reported number; the earlier draft inherited 6 from the previous code base.
    ap.add_argument("--out", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, names, df = load_design()
    n, p = X.shape
    print(f"policies n={n:,}  p={p}  pairs={n*(n-1)//2:,}", flush=True)
    print(f"  response: median {np.median(y):,.0f}  mean {y.mean():,.0f}  "
          f"max {y.max():,.0f}  skew {float(pd.Series(y).skew()):.1f}", flush=True)

    sigma, huber_coef = huber_scale(X, y)
    print(f"  Huber residual scale {sigma:,.0f}   (OLS residual scale "
          f"{np.std(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]):,.0f})", flush=True)

    t0_ = time.time()
    kern, res, T, t0, m0, m = fit(X, y, args.B, FIT_SEED, args.beta, args.mult, sigma)
    fit_s = time.time() - t0_
    theta = res["theta"]; se = np.sqrt(np.maximum(np.diag(res["Sigma_full"]), 0))
    print(f"fit: B={args.B} T={T:,} t0={t0} N0={args.B*t0:,} m0=m={m0:,} h={kern.h:.4f}  "
          f"{fit_s:.1f}s  pilot_ok={res['pilot_ok']} iters={res['pilot_iters']}", flush=True)
    print(f"     floor A0={res['A0_floor_active']} AT={res['AT_floor_active']}  "
          f"kappa(A_T)={res['AT_kappa']:.1f}  PSD proj k0={res.get('zeta_projected_k0')}", flush=True)

    print(f"bootstrap: {args.bootstrap} policy-level resamples", flush=True)
    boots = []
    tb = time.time()
    for b in range(args.bootstrap):
        rb = np.random.default_rng(60000 + b)
        idx = rb.integers(0, n, n)
        sb, _ = huber_scale(X[idx], y[idx])
        _, rb_res, *_ = fit(X[idx], y[idx], args.B, 95000 + b, args.beta, args.mult, sb,
                            want_zeta=False)
        boots.append(rb_res["theta"])
        if (b + 1) % 10 == 0:
            print(f"  {b+1}/{args.bootstrap}  ({(time.time()-tb)/(b+1):.1f}s each)", flush=True)
    boots = np.array(boots)
    se_boot = boots.std(0, ddof=1)
    ratio = se / se_boot
    print(f"online/bootstrap SE ratio: median {np.median(ratio):.3f}  "
          f"range [{ratio.min():.3f}, {ratio.max():.3f}]", flush=True)

    out = {"dataset": "freMTPL2 policy-level claim severity (CASdatasets)",
           "n": int(n), "p": int(p), "pairs": int(n * (n - 1) // 2),
           "response_median": float(np.median(y)), "response_max": float(y.max()),
           "response_skew": float(pd.Series(y).skew()),
           "huber_scale": float(sigma), "bandwidth": float(kern.h), "beta": args.beta,
           # The ordinary-least-squares residual scale, for the comparison the
           # manuscript draws with the robust one; printing it to stdout left
           # the ratio with no stored source.
           "ols_scale": float(np.std(y - X @ np.linalg.lstsq(X, y, rcond=None)[0], ddof=1)),
           "var_data": np.diag(res["Sigma_dat"]).tolist(),
           "var_alg": np.diag(res["alg_full"]).tolist(),
           "alg_share": float(np.mean(np.diag(res["alg_full"]) /
                                      (np.diag(res["Sigma_dat"]) + np.diag(res["alg_full"])))),
           "algorithm": {"B": args.B, "T": T, "t0": t0, "N0": args.B * t0,
                         "m0": m0, "m": m, "k_anchor": K_ANCHOR, "mult": args.mult},
           "fit_seconds": fit_s,
           "diagnostics": {k: res[k] for k in res if isinstance(res[k], (int, float, bool, str))},
           "coef": theta.tolist(), "se_online": se.tolist(), "se_bootstrap": se_boot.tolist(),
           "se_ratio_median": float(np.median(ratio)),
           "se_ratio_range": [float(ratio.min()), float(ratio.max())],
           "n_significant": int((np.abs(theta) > 1.96 * se).sum()),
           "names": names, "bootstrap": args.bootstrap}
    p_out = os.path.join(args.out, "fremtpl2_rank_results.json")
    json.dump(out, open(p_out, "w"), indent=1)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
