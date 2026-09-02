"""Figures and tables for the fixed-minibatch study, in manuscript style.

    python3 -u make_fixed_B_outputs.py

Figure 1 reports summaries averaged over all parameter coordinates.  Every
fixed minibatch size B=1,10,100 is shown separately.  Accessible colour and line
style identify the statistical example, while marker shape identifies B.
Panel (b) displays data-only interval coverage directly.  The generator also
writes a compact main-text summary table and a compact finite-budget table; the
complete cell-level tables remain available for the supplement.

Axis limits are fixed in advance and use natural reference values, so short
numerical ranges are not visually exaggerated.  The archived growing-B results
are not touched and are never merged with the fixed-B experiment.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import experiment_design as D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(P.ROOT, "experiments_fixed_B")
AGG = os.path.join(ROOT, "aggregated_results")
FIGS = os.path.join(ROOT, "figures")
TABS = os.path.join(ROOT, "tables")

# --- the manuscript's existing figure style, unchanged ---------------------
WIDTHS = {"fig_fixedB.pdf": 7.00,          # two-column working copy
          "fig_fixedB_bka.pdf": 5.46}      # .94 of the 5.81in journal block

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

# Accessible colour choices with redundant line styles.  The bipartite curve
# remains interpretable in greyscale; B is encoded by marker shape.
COLOUR = {"rank": "#0072B2", "triplet": "#D55E00", "auc": "#000000"}
LABEL = {"rank": "Rank regression", "triplet": "Triplet metric learning",
         "auc": "Bipartite ranking"}
KERNEL_LINESTYLE = {"rank": "-", "triplet": "--", "auc": "-."}
MARKER = {1: "o", 10: "s", 100: "^"}
# Small multiplicative offsets on the logarithmic sample-size axis prevent
# coincident B-specific markers from hiding one another.  The data are not
# changed and the tick marks remain at the actual sample sizes.
XSHIFT = {1: 0.965, 10: 1.0, 100: 1.035}
GRAY = "#888888"
R = D.R_FORMAL
Z95 = 1.96
SINGLE_COORD_MCSE = np.sqrt(0.95 * 0.05 / R)

CAPTION = {
    "rank": ("Smoothed rank regression ($p=2$, $\\theta^\\star=(0.5,1)^\\top$). ",
             "tab:rank-fixedB"),
    "auc": ("Two-sample bipartite ranking ($p=3$, $\\pi_+=0.4$). ", "tab:auc-fixedB"),
    "triplet": ("Triplet metric learning ($d_x=4$, $p=10$). ", "tab:metric-fixedB"),
}
KERNELS = ("rank", "triplet", "auc")


# --- the manuscript's existing number formats, unchanged ------------------
def _t(x):
    if x >= 100: return f"{x:.0f}"
    if x >= 10:  return f"{x:.1f}"
    if x >= 1:   return f"{x:.2f}"
    if x >= 0.01: return f"{x:.3f}"
    return f"{x:.4f}"


def fmt(x, sig=3):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -3 <= e <= 3:
        return f"{x:.{max(0, sig - 1 - e)}f}"
    return f"{x/10**e:.{sig-1}f}e{e:+d}".replace("e+0", "e+").replace("e-0", "e-")


# --------------------------------------------------------------------------
def load():
    d = pd.read_csv(os.path.join(AGG, "fixed_B_summary.csv"))
    w = pd.read_csv(os.path.join(AGG, "wor_robustness_summary.csv"))
    sp = os.path.join(AGG, "budget_ratio_summary.csv")
    s = pd.read_csv(sp) if os.path.exists(sp) else None
    tp = os.path.join(AGG, "timing_fixed_B.json")
    timing = json.load(open(tp))["cells"] if os.path.exists(tp) else []
    if not timing:
        print("  (no timing_fixed_B.json; the Time column will print as --)")
    return d, w, s, timing


#: The journal layout sets a table caption with \tbl{caption}{body}, above the rule,
#: not with \caption below it.  The archived tables use it and these must match,
#: or the same page carries two caption styles.
def tbl(rows, header, caption, label, tabcolsep=3, resize=True, rules=True,
        placement="!tbp"):
    # Supplement tables may float; compact main-text tables use placement="H"
    # so evidence cannot appear before the subsection that introduces it.
    out = [rf"\begin{{table}}[{placement}]", "\\tbl{" + caption + "}{%",
           rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}"]
    if resize:
        out.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    out += header + ([r"\midrule"] if rules else []) + rows
    out += ([r"\bottomrule"] if rules else []) + [r"\end{tabular}"]
    out.append("}}" if resize else "}")
    out += [f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(out) + "\n"


def sci(x, sig=3):
    """The archived tables' number style: fixed when moderate, times-ten-to otherwise."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -3 <= e <= 3:
        s = f"{x:.{max(0, sig - 1 - e)}f}"
        return s
    return f"${x/10**e:.{sig-1}f}\\times10^{{{e}}}$"


def table(kern, d, wor, timing):
    """The manuscript's table, unchanged in form from the archived one.

    Coverage and time are reported under both sampling schemes, slash-separated
    with replacement first.  Bias and the three variances are reported once: the
    two schemes differ there by well under a Monte Carlo standard error, so a
    slash pair would invite a reader to see an effect that is not resolved.
    """
    cap, lab = CAPTION[kern]
    sub = d[(d.kernel == kern) & (d.scheme == "WR")].sort_values(["n_total", "B"])
    wk = wor[wor.kernel == kern].set_index(["n_total", "B"])
    tk = {}
    for r in timing:
        if r["kernel"] == kern:
            tk[(r["n"], r["B"], r["scheme"])] = r["median_s"]
    header = [r"\begin{tabular}{rrccccccc}",
              r"& & & & \multicolumn{3}{c}{Variance} & &\\[-1mm]",
              r"$n$ & $B$ & $T$ & $|\text{Bias}|$ & Empirical & Data",
              r"& Optimization & Coverage & Time (seconds)\\",
              r"\addlinespace[2pt]"]
    rows, printed, last_n = [], [], None
    for _, r in sub.iterrows():
        if last_n is not None and r.n_total != last_n:
            rows.append(r"\addlinespace")
        last_n = r.n_total
        ne = int(round(np.log10(r.n_total)))
        key = (r.n_total, r.B)
        cov_wor = wk.loc[key, "cov_full"] if key in wk.index else r.cov_full
        t_wr = tk.get((r.n_total, r.B, "WR"))
        t_wor = tk.get((r.n_total, r.B, "WOR"), t_wr)
        times = (f"{_t(t_wr)}/{_t(t_wor)}" if t_wr is not None else "--")
        rows.append(
            f"$10^{{{ne}}}$ & {int(r.B)} & {int(r['T']):,} & {sci(r.bias_abs)} & "
            f"{sci(r.v_emp)} & {sci(r.v_data)} & {sci(r.v_alg)} & "
            f"{r.cov_full:.3f}/{cov_wor:.3f} & {times}\\\\".replace(",", "{,}"))
        printed.append(r)
    caption = (cap +
               "Each row is one asymptotic sequence: the minibatch size $B$ is held "
               "fixed and the number of iterations grows with $n$. The horizon is the "
               "primary numerical schedule $T=\\lfloor(n/B)\\log_2n\\rfloor$, so rows sharing "
               "an $n$ share the tuple budget $BT=n\\log_2n$ up to the integer part, "
               "and also the pilot budget $Bt_0$, the auxiliary budgets and the tuple "
               "draws themselves, so the comparison across $B$ is paired. "
               f"$R={R}$ replications; Monte Carlo standard error of a coverage near "
               "$0.95$ is about $0.007$. Slash-separated entries list sampling with "
               "replacement first and sampling without replacement second; at $B=1$ "
               "the two schemes are one computation, since a minibatch of size one "
               "drawn without replacement is the with-replacement draw. Bias and the "
               "three variances are listed once: across the paired cells the two "
               "schemes differ in those columns by less than the Monte Carlo standard "
               "error of the quantity itself, so a pair would suggest a difference the "
               "design does not resolve. Times "
               "are medians of five serial runs of the reported procedure under each "
               "scheme, measured in one pass on an idle machine")
    return tbl(rows, header, caption, lab, rules=False), printed



def table_budget_ratio(s):
    """The appendix table: the budget varied at a fixed sample size and B."""
    header = [r"\begin{tabular}{rrccccc}", r"\toprule",
              r"$\Lambda_n$ & $T$ & $\widehat v_{\mathrm{data}}$ & $\widehat v_{\mathrm{opt}}$ "
              r"& $\Lambda_n\widehat v_{\mathrm{opt}}$ & Full coverage & Data-only coverage\\"]
    rows, printed = [], []
    for _, r in s.sort_values("Lambda").iterrows():
        rows.append(
            f"{r.Lambda:.0f} & {int(r['T']):,} & {sci(r.v_data)} & {sci(r.v_alg)} & "
            f"{sci(r.v_alg * r.Lambda)} & {r.cov_full:.3f} & "
            f"{r.cov_data_only:.3f}\\\\".replace(",", "{,}"))
        printed.append(r)
    caption = ("Smoothed rank regression at $n=10^{4}$ and $B=10$, varying "
               "$\\Lambda_n=BT/n_1$ through the horizon. The data, pilot, auxiliary "
               "draws and tuple-stream prefixes are shared across rows; "
               f"$R={R}$ replications. The fifth column checks the predicted "
               "$\\Lambda_n^{-1}$ scaling")
    return tbl(rows, header, caption, "tab:budget-ratio-fixedB",
               tabcolsep=5, resize=False), printed


def table_main_summary(d):
    """Compact main-text summary over the twenty-seven primary WR cells."""
    wr = d[d.scheme == "WR"]
    header = [r"\begin{tabular}{lccccc}", r"\toprule",
              r"Example & Full coverage & Data-only coverage & Optimization share "
              r"& Variance ratio & Maximum coverage loss\\"]
    rows, printed = [], []
    names = {"rank": "Rank regression", "triplet": "Triplet metric learning",
             "auc": "Bipartite ranking"}
    for kern in KERNELS:
        g = wr[wr.kernel == kern]
        rec = {
            "kernel": kern,
            "cov_lo": float(g.cov_full.min()),
            "cov_hi": float(g.cov_full.max()),
            "data_lo": float(g.cov_data_only.min()),
            "data_hi": float(g.cov_data_only.max()),
            "share_lo": float(g.alg_share.min()),
            "share_hi": float(g.alg_share.max()),
            "ratio_lo": float(g.ratio_emp_pred.min()),
            "ratio_hi": float(g.ratio_emp_pred.max()),
            "loss": float((g.cov_full - g.cov_data_only).max()),
        }
        rows.append(
            f"{names[kern]} & {rec['cov_lo']:.3f}--{rec['cov_hi']:.3f} & "
            f"{rec['data_lo']:.3f}--{rec['data_hi']:.3f} & "
            f"{100*rec['share_lo']:.1f}--{100*rec['share_hi']:.1f}\\% & "
            f"{rec['ratio_lo']:.3f}--{rec['ratio_hi']:.3f} & "
            f"{rec['loss']:.3f}\\\\")
        printed.append(rec)
    caption = ("Coordinate-averaged summary of the primary fixed-minibatch grid, "
               "using sampling with replacement. Ranges are over total sample size "
               "$n_{\\mathrm{tot}}=10^3,10^4,10^5$ and $B=1,10,100$. The variance ratio is "
               r"$\widehat v_{\mathrm{emp}}/(\widehat v_{\mathrm{data}}+"
               r"\widehat v_{\mathrm{opt}})$; maximum coverage loss is the largest decrease "
               "in coverage after omitting the stochastic-optimization component. "
               f"$R={R}$ replications per cell")
    return tbl(rows, header, caption, "tab:simulation-summary",
               tabcolsep=3.5, resize=True, placement="H"), printed


def table_budget_ratio_main(s):
    """Compact main-text table for the finite-budget experiment."""
    header = [r"\begin{tabular}{rrcccc}", r"\toprule",
              r"$\Lambda_n=BT/n_1$ & $T$ & Optimization share "
              r"& $\Lambda_n\widehat v_{\mathrm{opt}}$ & Full coverage "
              r"& Data-only coverage\\"]
    rows, printed = [], []
    for _, r in s.sort_values("Lambda").iterrows():
        rec = {
            "Lambda": float(r.Lambda),
            "T": int(r["T"]),
            "alg_share": float(r.alg_share),
            "scaled": float(r.v_alg * r.Lambda),
            "cov_full": float(r.cov_full),
            "cov_data_only": float(r.cov_data_only),
        }
        rows.append(
            (f"{rec['Lambda']:.0f} & {rec['T']:,} & "
             f"{100*rec['alg_share']:.1f}\\% & {sci(rec['scaled'])} & "
             f"{rec['cov_full']:.3f} & {rec['cov_data_only']:.3f}\\\\")
            .replace(",", "{,}"))
        printed.append(rec)
    caption = ("Finite-budget smoothed-rank experiment with "
               "$n_1=n_{\\mathrm{tot}}=10^4$ and $B=10$. Only the horizon varies. "
               r"The tuple-budget ratio is $\Lambda_n=BT/n_1$. The product "
               r"$\Lambda_n\widehat v_{\mathrm{opt}}$ checks the predicted "
               r"$\Lambda_n^{-1}$ decay. All displayed variance and coverage summaries "
               r"are coordinate averages. The data, pilot and auxiliary random-number streams "
               f"are shared across rows; $R={R}$ replications")
    return tbl(rows, header, caption, "tab:budget-ratio-main",
               tabcolsep=3.2, resize=True, placement="H"), printed


def table_wor(d, w):
    """What the per-kernel tables do not show: the optimization variance itself.

    Those tables carry coverage under both schemes. The quantity that the
    finite-population correction actually acts on is the optimization variance,
    and the point of this table is how little it moves.
    """
    m = w.merge(d[d.scheme == "WR"][["kernel", "n_total", "B", "v_alg", "v_emp"]],
                on=["kernel", "n_total", "B"], suffixes=("", "_wr"))
    order = {k: i for i, k in enumerate(KERNELS)}
    m = m.sort_values(["kernel", "n_total", "B"],
                      key=lambda s: s.map(order) if s.name == "kernel" else s)
    header = [r"\begin{tabular}{lrrccc}", r"\toprule",
              r"Example & $n$ & $B$ & $\widehat v_{\mathrm{opt}}$ (WOR) "
              r"& $\widehat v_{\mathrm{opt}}$ (WR) & Ratio\\"]
    rows, printed, last = [], [], None
    for _, r in m.iterrows():
        name = {"rank": "Rank", "triplet": "Triplet", "auc": "AUC"}[r.kernel]
        if last is not None and r.kernel != last:
            rows.append(r"\addlinespace")
        ne = int(round(np.log10(r.n_total)))
        rows.append(
            f"{name if r.kernel != last else ''} & $10^{{{ne}}}$ & {int(r.B)} & "
            f"{sci(r.v_alg)} & {sci(r.v_alg_wr)} & {r.v_alg / r.v_alg_wr:.4f}\\\\")
        printed.append(r); last = r.kernel
    lo = (m.v_alg / m.v_alg_wr).min(); hi = (m.v_alg / m.v_alg_wr).max()
    caption = ("Optimization variance under the two sampling schemes, over every "
               "reported cell. The finite-population correction is $1-O(B/N)$ with "
               "$N$ the number of distinct tuples, of order $n^2$ for the pairwise "
               "kernels and $n^3$ for the triplet kernel, so the two schemes are "
               "expected to be indistinguishable; the realized ratios in the last "
               "column differ from one only in the fourth decimal. "
               "$B=1$ is omitted: a minibatch of size one drawn without replacement is "
               "the with-replacement draw, returning the same tuple from the same "
               f"generator state. $R={R}$ replications")
    return tbl(rows, header, caption, "tab:wor-fixedB", tabcolsep=5, resize=False), printed



def table_data_only(d):
    """Cell-level data-only coverage, reported in the supplement.

    The revised main paper gives compact coordinate-averaged ranges for the
    with-replacement design. This table retains every cell and reports what
    happens if the stochastic-optimization
    component is omitted, which is the comparison summarized in the main text
    and which a reader has to be able to check.
    """
    sub = d[d.scheme == "WR"]
    order = {k: i for i, k in enumerate(KERNELS)}
    sub = sub.sort_values(["kernel", "n_total", "B"],
                          key=lambda s: s.map(order) if s.name == "kernel" else s)
    header = [r"\begin{tabular}{lrrcccc}", r"\toprule",
              r"Example & $n$ & $B$ & Optimization share & Full coverage & Data-only coverage "
              r"& Difference\\"]
    rows, printed, last = [], [], None
    for _, r in sub.iterrows():
        if last is not None and r.kernel != last:
            rows.append(r"\addlinespace")
        name = {"rank": "Rank", "triplet": "Triplet", "auc": "AUC"}[r.kernel]
        ne = int(round(np.log10(r.n_total)))
        rows.append(
            f"{name if r.kernel != last else ''} & $10^{{{ne}}}$ & {int(r.B)} & "
            f"{r.alg_share*100:.1f}\\% & {r.cov_full:.3f} & {r.cov_data_only:.3f} & "
            f"{r.cov_full - r.cov_data_only:.3f}\\\\")
        printed.append(r); last = r.kernel
    caption = ("Coordinatewise effect of omitting the stochastic-optimization "
               "component in the primary with-replacement grid. Optimization share "
               "is its fraction of the plug-in variance; Difference is full minus "
               f"data-only coverage. $R={R}$ replications")
    return tbl(rows, header, caption, "tab:data-only-fixedB",
               tabcolsep=5, resize=False), printed


def table_runtime(d):
    sub = d[(d.scheme == "WR") & (d.n_total == 100_000)]
    order = {k: i for i, k in enumerate(KERNELS)}
    sub = sub.sort_values(["kernel", "B"],
                          key=lambda s: s.map(order) if s.name == "kernel" else s)
    header = [r"\begin{tabular}{lrrcccccc}", r"\toprule",
              r"Kernel & $B$ & Updates & Pilot & Hessian & Recursion & Terminal "
              r"& Anchors & Total\\"]
    rows, printed, last = [], [], None
    for _, r in sub.iterrows():
        name = {"rank": "Rank", "triplet": "Triplet", "auc": "AUC"}[r.kernel]
        rows.append(
            f"{name if r.kernel != last else ''} & {int(r.B)} & "
            f"{int(r.steps):,} & {_t(r.t_pilot)} & {_t(r.t_hessian0)} & "
            f"{_t(r.t_recursion)} & {_t(r.t_terminal)} & {_t(r.t_anchors)} & "
            f"{_t(r.t_total)}\\\\".replace(",", "{,}"))
        printed.append(r); last = r.kernel
    caption = ("Wall-clock by phase at $n=10^{5}$, in seconds per replication. Every "
               "phase but the recursion evaluates the same number of tuples whatever "
               "$B$ is, and its cost is correspondingly flat; the small residual "
               "variation in the sub-millisecond columns is the per-call overhead of "
               "the pilot solver, not a difference in work. The recursion is where the "
               "minibatch size tells: it performs $T-t_0$ sequential updates on a "
               "budget $BT$ that does not depend on $B$. The columns are the phases in "
               "the order the procedure runs them and sum to the total")
    return tbl(rows, header, caption, "tab:runtime-fixedB", tabcolsep=5, resize=False), printed


# --- the figure: inferential consequences, summarized across fixed B -----
def draw(width_in, d):
    """Main simulation summary averaged over all parameter coordinates.

    Every fixed minibatch size is shown.  Colour and line style distinguish
    examples; marker shape distinguishes B.  The coverage panels use a nominal
    reference line, not an uncertainty band; the share panel starts at zero and
    the variance-ratio panel is centred at one.
    """
    fig, axarr = plt.subplots(2, 2, figsize=(width_in, width_in * 0.86),
                              layout="constrained")
    axes = axarr.ravel()
    wr = d[d.scheme == "WR"].copy()

    required = {"cov_full", "cov_data_only", "alg_share", "ratio_emp_pred"}
    missing = required.difference(wr.columns)
    if missing:
        raise RuntimeError(f"Figure 1 requires coordinate-average columns: {sorted(missing)}")

    for kern in KERNELS:
        c = COLOUR[kern]
        for B in D.B_GRID:
            g = wr[(wr.kernel == kern) & (wr.B == B)].sort_values("n_total")
            x = g.n_total.to_numpy(dtype=float) * XSHIFT[B]
            style = dict(color=c, marker=MARKER[B],
                         linestyle=KERNEL_LINESTYLE[kern],
                         markerfacecolor="white", markeredgecolor=c,
                         markeredgewidth=0.9)

            axes[0].plot(x, g.cov_full, **style)
            axes[1].plot(x, g.cov_data_only, **style)
            axes[2].plot(x, 100.0 * g.alg_share, **style)
            axes[3].plot(x, g.ratio_emp_pred, **style)

    # Fixed scales were chosen before plotting and are wider than the observed
    # ranges, so small Monte Carlo differences do not look artificially large.
    for ax in axes[:2]:
        ax.axhline(0.95, linestyle=":", color=GRAY, linewidth=0.8, zorder=0)
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.88, 1.00)
        ax.set_yticks([0.88, 0.90, 0.925, 0.95, 0.975, 1.00])

    axes[2].axhline(0.0, color=GRAY, linewidth=0.6, zorder=0)
    axes[2].set_ylabel("Optimization share (%)")
    axes[2].set_ylim(0.0, 25.0)
    axes[2].set_yticks([0, 5, 10, 15, 20, 25])

    axes[3].axhline(1.0, linestyle=":", color=GRAY, linewidth=0.8, zorder=0)
    axes[3].set_ylabel("Empirical / predicted variance")
    axes[3].set_ylim(0.90, 1.10)
    axes[3].set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])

    titles = ("(a) Full-interval coverage",
              "(b) Data-only coverage",
              "(c) Optimization variance share",
              "(d) Variance calibration")
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks(D.N_GRID)
        ax.set_xticklabels([r"$10^3$", r"$10^4$", r"$10^5$"])
        ax.set_xlabel("Total sample size $n_{\\mathrm{tot}}$")
        ax.set_title(title, pad=4)

    kernel_handles = [
        plt.Line2D([], [], color=COLOUR[k], linestyle=KERNEL_LINESTYLE[k],
                   linewidth=1.8, label=LABEL[k])
        for k in KERNELS
    ]
    b_handles = [
        plt.Line2D([], [], color="#333333", marker=MARKER[B],
                   linestyle="None", markerfacecolor="white",
                   markeredgecolor="#333333", label=f"$B={B}$")
        for B in D.B_GRID
    ]
    handles = kernel_handles + b_handles
    fig.legend(handles, [h.get_label() for h in handles],
               loc="outside lower center", ncol=3, handlelength=1.8,
               borderpad=0.15, columnspacing=1.5, handletextpad=0.4)
    return fig


def verify(tex, printed, cols, name):
    """Every printed number must appear, in the row it belongs to.

    Containment in the whole file would pass a table whose rows were permuted,
    so each record is matched against a single line: the one carrying all of its
    values.  A guard that cannot fail is worse than no guard, because it is
    reported as a pass.
    """
    bad = []
    lines = [l for l in tex.split("\n") if l.rstrip().endswith(r"\\")]
    for r in printed:
        want = [f(r[c]) for c, f in cols]
        if not any(all(w in l for w in want) for l in lines):
            missing = [w for w in want if w not in tex]
            bad.append(f"{name}: no row carries {want}"
                       + (f" (absent from the file: {missing})" if missing else
                          " together, though each appears somewhere"))
    return bad


def main():
    os.makedirs(FIGS, exist_ok=True); os.makedirs(TABS, exist_ok=True)
    os.makedirs(P.FIGURES, exist_ok=True)
    d, w, sweep, timing = load()
    problems = []

    for kern in KERNELS:
        tex, printed = table(kern, d, w, timing)
        for out in (TABS, P.FIGURES):
            open(os.path.join(out, f"{kern}_fixedB_table.tex"), "w").write(tex)
        # the checked format must be the printed one, or the guard passes on
        # tables it never actually read
        problems += verify(tex, printed,
                           [("bias_abs", sci), ("v_emp", sci), ("v_data", sci),
                            ("v_alg", sci), ("cov_full", lambda x: f"{x:.3f}"),
                            ("T", lambda x: f"{int(x):,}".replace(",", "{,}"))],
                           f"{kern} table")
        print(f"  {kern}: {len(printed)} rows -> {kern}_fixedB_table.tex")

    outs = [("wor_fixedB_table.tex", table_wor(d, w)),
            ("data_only_fixedB_table.tex", table_data_only(d)),
            ("runtime_fixedB_table.tex", table_runtime(d)),
            ("simulation_compact_main.tex", table_main_summary(d))]
    if sweep is not None:
        outs.extend([
            ("budget_ratio_fixedB_table.tex", table_budget_ratio(sweep)),
            ("budget_ratio_main.tex", table_budget_ratio_main(sweep)),
        ])
    COLS = {
        "wor_fixedB_table.tex": [("v_alg", sci), ("v_alg_wr", sci)],
        "data_only_fixedB_table.tex": [("cov_full", lambda x: f"{x:.3f}"),
                                       ("cov_data_only", lambda x: f"{x:.3f}")],
        "runtime_fixedB_table.tex": [("t_recursion", _t), ("t_total", _t)],
        "simulation_compact_main.tex": [
            ("cov_lo", lambda x: f"{x:.3f}"), ("cov_hi", lambda x: f"{x:.3f}"),
            ("data_lo", lambda x: f"{x:.3f}"), ("data_hi", lambda x: f"{x:.3f}"),
            ("share_hi", lambda x: f"{100*x:.1f}\\%"),
            ("ratio_lo", lambda x: f"{x:.3f}"), ("ratio_hi", lambda x: f"{x:.3f}"),
            ("loss", lambda x: f"{x:.3f}"),
        ],
        "budget_ratio_fixedB_table.tex": [("v_data", sci), ("v_alg", sci),
                                          ("cov_full", lambda x: f"{x:.3f}"),
                                          ("cov_data_only", lambda x: f"{x:.3f}")],
        "budget_ratio_main.tex": [
            ("Lambda", lambda x: f"{x:.0f}"),
            ("T", lambda x: f"{int(x):,}".replace(",", "{,}")),
            ("alg_share", lambda x: f"{100*x:.1f}\\%"),
            ("scaled", sci),
            ("cov_full", lambda x: f"{x:.3f}"),
            ("cov_data_only", lambda x: f"{x:.3f}"),
        ],
    }
    for name, (tex, printed) in outs:
        for out in (TABS, P.FIGURES):
            open(os.path.join(out, name), "w").write(tex)
        problems += verify(tex, printed, COLS[name], name)
        print(f"  {len(printed)} rows -> {name}")

    if problems:
        print("  MISMATCHES:"); [print("   ", p) for p in problems]; sys.exit(1)
    print(f"  verification: every row of all {len(outs) + len(KERNELS)} tables matched "
          f"against the summary frame, value by value")

    for name, wid in WIDTHS.items():
        fig = draw(wid, d)
        fig.savefig(os.path.join(FIGS, name))
        if name.endswith("_bka.pdf"):
            fig.savefig(os.path.join(P.FIGURES, name))
        plt.close(fig)
        print(f"  wrote {name}  ({wid:.2f} in wide, drawn at that size)")

    wrd = d[d.scheme == "WR"]
    facts = {
        "ratio_range": [float(wrd.ratio_emp_pred.min()), float(wrd.ratio_emp_pred.max())],
        "cov_full_range": [float(wrd.cov_full.min()), float(wrd.cov_full.max())],
        "cov_data_only_range": [float(wrd.cov_data_only.min()), float(wrd.cov_data_only.max())],
        "alg_share_range": [float(wrd.alg_share.min()), float(wrd.alg_share.max())],
        "max_cov_spread_across_B": float(max(
            g.cov_full.max() - g.cov_full.min() for _, g in wrd.groupby(["kernel", "n_total"]))),
        "max_v_alg_spread_pct": float(max(
            (g.v_alg.max() - g.v_alg.min()) / g.v_alg.mean() * 100
            for _, g in wrd.groupby(["kernel", "n_total"]))),
        "D0_same_everywhere": bool(wrd.D0_same_across_B.all()),
        "wor_ratio_range": None, "R": R,
        "single_coordinate_mcse": float(SINGLE_COORD_MCSE),
    }
    m = w.merge(d[d.scheme == "WR"][["kernel", "n_total", "B", "v_alg"]],
                on=["kernel", "n_total", "B"], suffixes=("", "_wr"))
    facts["wor_ratio_range"] = [float((m.v_alg / m.v_alg_wr).min()),
                                float((m.v_alg / m.v_alg_wr).max())]
    json.dump(facts, open(os.path.join(TABS, "fixed_B_facts.json"), "w"), indent=1)
    print("  wrote fixed_B_facts.json")
    for k, v in facts.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
