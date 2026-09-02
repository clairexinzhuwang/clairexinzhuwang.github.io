"""Simulation-summary figure, drawn from grid_{rank,auc,triplet}.json.

Same three panels as before (coverage, absolute bias, empirical-to-predicted
variance ratio) and the same visual style, but every point is read from the
result files rather than pasted in as a literal array, which is how the old
version of this figure drifted out of step with the runs it described.

For each kernel and each n the panels show the largest-B cell, and the shaded
band is the Monte Carlo interval implied by R = 1000.
"""
import json, os
import numpy as np
import paths as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = P.RESULTS
# Drawn once per target text block rather than drawn once and scaled: scaling a
# figure changes its apparent type size, which is what made the type here read
# smaller than the body text in the single-column layout.
WIDTHS = {"fig_combined.pdf": 7.00,        # two-column working copy
          "fig_combined_bka.pdf": 5.46}    # .94 of the 5.81in journal block

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COLOUR = {"rank": "#4477AA", "triplet": "#EE6677", "auc": "#228833"}
LABEL = {"rank": "Rank ($d=2$)", "triplet": "Triplet ($d=3$)",
         "auc": "AUC ($K=2$)"}
GRAY = "#888888"
R = 1000
Z95 = 1.96
MCSE = np.sqrt(0.95 * 0.05 / R)


N_MIN = 1000        # the paper reports 10^3 to 10^6


def series(kern):
    """One point per n: the largest-B cell under with-replacement sampling."""
    recs = [r for r in json.load(open(os.path.join(ROOT, f"grid_{kern}.json")))
            if r["scheme"] == "WR" and r["n"] >= N_MIN]
    by_n = {}
    for r in recs:
        if r["n"] not in by_n or r["B"] > by_n[r["n"]]["B"]:
            by_n[r["n"]] = r
    ns = sorted(by_n)
    return (np.array(ns),
            np.array([by_n[n]["coverage"] for n in ns]),
            np.array([by_n[n]["bias"] for n in ns]),
            np.array([by_n[n]["VR"] for n in ns]),
            np.array([by_n[n]["var_emp"] for n in ns]))


def draw(width_in):
    # constrained layout, so the panels, their labels and the legend are placed
    # from their measured sizes; hand-tuned fractions only hold at one figure
    # size and ran the legend into the axis labels at another.
    fig, axes = plt.subplots(1, 3, figsize=(width_in, width_in * 2.35 / 7.0),
                             layout="constrained")
    for kern in ("rank", "triplet", "auc"):
        n, cov, bias, vr, vemp = series(kern)
        c = COLOUR[kern]
        axes[0].errorbar(n, cov, yerr=Z95 * MCSE, color=c, marker="o",
                         capsize=2, elinewidth=0.7, label=LABEL[kern])
        # Bias is a mean over coordinates with Monte Carlo error sd/sqrt(R).
        # On a log axis a symmetric interval reaches zero, so the lower arm is
        # clipped to the plotted range; the upper arm carries the information.
        se = Z95 * np.sqrt(vemp / R)
        lo = np.minimum(se, bias * 0.9)          # keep the bar inside the axis
        axes[1].errorbar(n, bias, yerr=np.vstack([lo, se]), color=c,
                         marker="o", capsize=2, elinewidth=0.7, label=LABEL[kern])
        axes[2].plot(n, vr, color=c, marker="o", label=LABEL[kern])

    axes[0].axhline(0.95, linestyle="--", color=GRAY, linewidth=0.7, zorder=0)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0.905, 0.985)
    axes[1].set_ylabel(r"$|\mathrm{Bias}|$")
    axes[1].set_yscale("log")
    axes[2].axhline(1.0, linestyle="--", color=GRAY, linewidth=0.7, zorder=0)
    axes[2].set_ylabel(r"$\var_{\mathrm{emp}}/(\var_{\mathrm{dat}}+\var_{\mathrm{sgd}})$"
                       .replace(r"\var", r"\mathrm{var}"))
    axes[2].set_ylim(0.85, 1.15)
    for ax, t in zip(axes, ("(a) Coverage", "(b) Bias decay", "(c) Variance match")):
        ax.set_xscale("log")
        ax.set_xlabel("Sample size $n$")
        ax.set_title(t, pad=4)
    # One legend for the whole figure, below the panels. The space it needs is
    # a fixed number of inches, not a fixed fraction of the height, so the
    # fractions are computed from the figure size; a fixed fraction let the
    # legend run into the axis labels once the figure was drawn narrower.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               handlelength=1.2, borderpad=0.15, columnspacing=1.6,
               handletextpad=0.4)
    return fig


def main():
    out_dir = P.FIGURES
    os.makedirs(out_dir, exist_ok=True)
    for name, w in WIDTHS.items():
        fig = draw(w)
        p = os.path.join(out_dir, name)
        fig.savefig(p)
        plt.close(fig)
        print(f"  wrote {p}  ({w:.2f} in wide, drawn at that size)")
    for kern in ("rank", "triplet", "auc"):
        n, cov, bias, vr, _ = series(kern)
        print(f"    {kern:<8} n={list(n)}")
        print(f"      coverage {np.round(cov,3)}   VR {np.round(vr,2)}")


if __name__ == "__main__":
    main()
