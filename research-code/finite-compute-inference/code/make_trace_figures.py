"""Two supplementary figures, drawn with Algorithm 1 at reported grid cells.

The two figures are deliberately about different things.

Figure S1 separates the two error sources.  Several runs are made on each of a
few independently generated datasets: runs sharing a dataset converge towards
that dataset's own complete-U minimizer, drawn as a dashed line in the
dataset's colour, while the dashed lines themselves scatter about the
population value.  Within-colour spread is optimization error and
between-colour spread is sampling error, which no other display in the paper
shows separately.

Figure S2 checks the normal approximation the theorem asserts.  Coverage alone
only probes one quantile pair, so the standardized last iterate is shown
against normal quantiles for all three kernels, over the same replications and
seeds as the reported cells.

Run:  python3 -u make_trace_figures.py
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alg_paper import run_algorithm1, default_t0, default_m   # noqa: E402
from kernels import RankKernel                                 # noqa: E402
import run_grid as G                                           # noqa: E402

import paths as P

OUT = P.FIGURES
THETA_STAR = np.array([0.5, 1.0])

# Every configuration below is a cell of the reported grid, so the schedule is
# the prescribed T = (n/B) log2 n and the seeds match the reported coverage.
DECOMP = dict(n=10 ** 3, B=128, datasets=60, runs=12)
QQ = [("rank", 10 ** 4, 1024), ("triplet", 10 ** 4, 1024), ("auc", 10 ** 4, 1024)]
TRACE = dict(n=10 ** 4, B=256, datasets=6, runs=10)   # a reported cell
R_QQ = 1000
LABEL = {"rank": "Rank ($d=2$)", "triplet": "Triplet ($d=3$)", "auc": "AUC ($K=2$)"}
COLOUR = {"rank": "#4477AA", "triplet": "#EE6677", "auc": "#228833"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = X @ THETA_STAR + rng.normal(size=n)
    return X, y


def figure_decomposition():
    """Check the two variance components separately, not just their sum.

    Independent datasets each carry several independent runs.  The scatter of
    runs within a dataset estimates the optimization component and the scatter
    of the datasets' own complete-U minimizers estimates the data component, so
    the two halves of the sandwich can be compared with the theory one at a
    time rather than only through their total.
    """
    n, B = DECOMP["n"], DECOMP["B"]
    nd, nr = DECOMP["datasets"], DECOMP["runs"]
    T = int((n / B) * np.log2(n))
    t0 = default_t0(T, rule="rank", n=n)
    finals, hats, v_dat, v_sgd = [], [], [], []
    for d in range(nd):
        X, y = make_data(n, 40000 + d)
        hats.append(RankKernel(X, y, beta=0.26).full_minimizer())
        row = []
        for r in range(nr):
            kern = RankKernel(X, y, beta=0.26)
            m0 = m = default_m(B, kern.n_ref, kern.h)
            # the plug-in components are averaged over one run per dataset,
            # which is already 60 independent evaluations of them
            want = (r == 0)
            res = run_algorithm1(kern, kern.theta_ols, B, T, t0, m0, m, n,
                                 scheme="WR",
                                 rng=np.random.default_rng(50000 + 100 * d + r),
                                 want_zeta=want, k_per_stream=16)
            row.append(res["theta"])
            if want:
                v_dat.append(np.diag(res["Sigma_dat"]))
                v_sgd.append(np.diag(res["Sigma_full"] - res["Sigma_dat"]))
        finals.append(row)
        if (d + 1) % 10 == 0:
            print(f"  decomposition: dataset {d + 1}/{nd}", flush=True)
    F = np.array(finals)                      # (datasets, runs, 2)
    H = np.array(hats)                        # (datasets, 2)
    v_dat = np.array(v_dat).mean(0)
    v_sgd = np.array(v_sgd).mean(0)
    within = F.var(axis=1, ddof=1).mean(0)    # optimization scatter
    between = H.var(axis=0, ddof=1)           # sampling scatter
    # a variance estimate on nu degrees of freedom has relative standard error
    # (2/nu)^{1/2}; without it a ratio of 0.8 cannot be told from noise
    se_within = np.sqrt(2.0 / (nd * (nr - 1)))
    se_between = np.sqrt(2.0 / (nd - 1))

    show = min(nd, 10)                        # plot a legible subset
    colours = plt.get_cmap("tab10")(np.linspace(0, 0.9, show))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    xs = np.arange(1, show + 1)
    for j, ax in enumerate(axes):
        for d in range(show):
            jitter = (np.arange(nr) - (nr - 1) / 2) * 0.045
            ax.plot(xs[d] + jitter, F[d, :, j], marker="o", markersize=2.6,
                    linestyle="none", color=colours[d], zorder=2)
            ax.hlines(H[d, j], xs[d] - 0.34, xs[d] + 0.34, color=colours[d],
                      linewidth=1.4, zorder=3)
        ax.axhline(THETA_STAR[j], color="black", linewidth=0.9, zorder=1)
        ax.set_xlabel("dataset")
        ax.set_ylabel(rf"$\widetilde\theta_{{n,T,{j + 1}}}$")
        ax.set_xticks(xs)
    p = os.path.join(OUT, "figS_decomposition.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}   ({nd} datasets used, {show} drawn)")
    for j in range(2):
        rw, rb = within[j] / v_sgd[j], between[j] / v_dat[j]
        print(f"    coord {j + 1}: optimization ratio {rw:.2f} "
              f"(+/- {1.96 * rw * se_within:.2f});  "
              f"sampling ratio {rb:.2f} (+/- {1.96 * rb * se_between:.2f})")
    return dict(n=n, B=B, T=int(T), t0=int(t0), datasets=nd, runs=nr,
                se_within=float(se_within), se_between=float(se_between),
                within=within.tolist(), var_sgd_pred=v_sgd.tolist(),
                between=between.tolist(), var_dat_pred=v_dat.tolist())


def figure_trace():
    """The recursion against the target each run is actually converging to.

    Several independent datasets, each carrying several runs.  A dataset's own
    complete-U minimizer is drawn as a dashed line in its colour: the runs of
    that dataset converge to it, not to theta_star, so the offsets of the
    dashed lines are sampling error and the scatter within a colour is
    optimization error.  The right panel measures each run's distance to its
    own dataset's minimizer, a fixed target, rather than to the run's own
    endpoint, which would be zero at the horizon by construction.
    """
    n, B = TRACE["n"], TRACE["B"]
    nd, nr = TRACE["datasets"], TRACE["runs"]
    T = int((n / B) * np.log2(n))
    t0 = default_t0(T, rule="rank", n=n)
    steps = np.arange(t0, T + 1)
    colours = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA4499", "#999933"]
    paths, hats = [], []
    for d in range(nd):
        X, y = make_data(n, 20000 + d)
        hats.append(RankKernel(X, y, beta=0.26).full_minimizer())
        row = []
        for r in range(nr):
            kern = RankKernel(X, y, beta=0.26)
            m0 = m = default_m(B, kern.n_ref, kern.h)
            res = run_algorithm1(kern, kern.theta_ols, B, T, t0, m0, m, n,
                                 scheme="WR",
                                 rng=np.random.default_rng(30000 + 100 * d + r),
                                 want_zeta=False, trace=True)
            row.append(res["path"])
        paths.append(row)
        print(f"  trace: dataset {d + 1}/{nd}  theta_hat_n = "
              f"{np.round(hats[-1], 4)}", flush=True)
    P = np.array(paths)                       # (datasets, runs, steps, 2)
    H = np.array(hats)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
    for d in range(nd):
        c = colours[d % len(colours)]
        for r in range(nr):
            axes[0].plot(steps, P[d, r, :, 0], color=c, linewidth=0.45,
                         alpha=0.65, zorder=1)
        axes[0].axhline(H[d, 0], linestyle="--", color=c, linewidth=1.1, zorder=3)
    axes[0].axhline(THETA_STAR[0], color="black", linewidth=1.3, zorder=4)
    axes[0].set_ylabel(r"$\widetilde\theta_{n,t,1}$")
    axes[0].set_xlabel("step $t$")
    axes[0].set_title(rf"{nd} datasets $\times$ {nr} runs; dashed "
                      rf"$\widehat\theta_{{n,1}}$, solid $\theta_1^\star$",
                      pad=4, fontsize=8)

    for d in range(nd):
        c = colours[d % len(colours)]
        dist = np.linalg.norm(P[d] - H[d], axis=2)
        for r in range(nr):
            axes[1].plot(steps, dist[r], color=c, linewidth=0.45, alpha=0.6, zorder=1)
    allrun = np.linalg.norm(P - H[:, None, None, :], axis=3).reshape(-1, len(steps))
    axes[1].plot(steps, np.exp(np.log(allrun).mean(0)), color="black",
                 linewidth=1.3, zorder=3)
    axes[1].axhline((B * T) ** -0.5, linestyle=":", color="black", linewidth=1.0,
                    zorder=2)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("step $t$")
    axes[1].set_ylabel(r"$\|\widetilde\theta_{n,t}-\widehat\theta_n\|$")
    axes[1].set_title(r"dotted: $(BT)^{-1/2}$", pad=4, fontsize=8)
    p = os.path.join(OUT, "figS_rank_traces.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    spread = float(H[:, 0].std(ddof=1))
    final = float(allrun[:, -1].mean())
    print(f"    n={n:,} B={B} t0={t0} T={T}  {nd} datasets x {nr} runs")
    print(f"    spread of theta_hat_n over datasets, coord 1: {spread:.4f}")
    print(f"    mean distance at the horizon: {final:.4f}  "
          f"against (BT)^-1/2 = {(B * T) ** -0.5:.4f}")
    return dict(n=n, B=B, t0=int(t0), T=int(T), datasets=nd, runs=nr,
                theta_hat=H.tolist(), hat_spread_coord1=spread,
                mean_final_distance=final, bt_root=float((B * T) ** -0.5))


def figure_qq():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    meta = []
    pp = (np.arange(1, R_QQ + 1) - 0.5) / R_QQ
    zq = norm.ppf(pp)
    for ax, (kern, n, B) in zip(axes, QQ):
        # the same seeds as the reported grid cell, so these are the very
        # replications whose coverage the table reports
        res = [G.one(kern, n, B, "WR", 1000 + s) for s in range(R_QQ)]
        th = np.array([r["theta"] for r in res])
        vf = np.array([r["v_full"] for r in res])
        star = target(kern)
        z = (th[:, 0] - star[0]) / np.sqrt(np.maximum(vf[:, 0], 0))
        zs = np.sort(z)
        ax.plot([-3.6, 3.6], [-3.6, 3.6], color="0.5", linestyle="--",
                linewidth=0.8, zorder=1)
        ax.plot(zq, zs, color=COLOUR[kern], marker="o", markersize=1.4,
                linestyle="none", zorder=2)
        ax.set_title(LABEL[kern], pad=4)
        ax.set_xlabel("Normal quantile")
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        meta.append(dict(kernel=kern, n=n, B=B, R=R_QQ,
                         mean=float(z.mean()), sd=float(z.std(ddof=1)),
                         shapiro_free=True))
        print(f"  qq {kern}: mean {z.mean():+.3f}  sd {z.std(ddof=1):.3f}",
              flush=True)
    axes[0].set_ylabel("Standardized last iterate")
    p = os.path.join(OUT, "figS_normal_qq.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return meta


def target(kern):
    """The population target, taken exactly as run_grid defines it.

    The AUC target is a 2 x 10^8 Monte Carlo evaluation, so it is cached on
    first use; the triplet target is the stored 3 x 10^8 stream that the grid
    itself loads, and the rank target is analytic.
    """
    if kern == "rank":
        return THETA_STAR
    if kern == "triplet":
        return np.load(os.path.join(HERE, "theta_star_p4K5_3e8.npz"))["theta_star"]
    cache = os.path.join(HERE, "theta_star_auc_2e8.npz")
    if not os.path.exists(cache):
        print("  computing the AUC population target (2e8 draws), once", flush=True)
        val = G.AU.compute_theta_star_auc(*G.AU.make_dgp(3), M=int(2e8))[0]
        np.savez(cache, theta_star=val)
    return np.load(cache)["theta_star"]


def main():
    os.makedirs(OUT, exist_ok=True)
    a = figure_decomposition()
    b = figure_qq()
    c = figure_trace()
    p = os.path.join(OUT, "trace_figures_meta.json")
    json.dump({"decomposition": a, "normal_qq": b, "trace": c}, open(p, "w"), indent=1)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
