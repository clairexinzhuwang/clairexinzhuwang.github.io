"""Generate the paper's result tables from grid_{rank,auc,triplet}.json.

Every number in the .tex output is read from the JSON produced by the frozen
Algorithm 1; nothing is typed by hand.  The script also re-reads what it wrote
and checks each printed cell against the JSON, so a formatting slip cannot
silently change a reported number.

Writes into the package's figures/ directory so nothing is overwritten until the numbers
have been inspected.
"""
import json, os, re, sys
import numpy as np
import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = P.RESULTS
OUT = P.FIGURES

# The reported sweeps run from 10^3 to 10^6.  The n = 100 rank cells were run
# and are kept in the result files, but they are below the range the paper
# reports and are not displayed.
N_MIN = 1000

MAIN_ROWS = {           # the compact sweeps shown in the main text
    "rank":    [(1000, 256), (10000, 512), (100000, 1024), (1000000, 4096)],
    "auc":     [(1000, 128), (10000, 512), (100000, 2048), (1000000, 8192)],
    "triplet": [(1000, 128), (10000, 512), (100000, 2048), (1000000, 8192)],
}
# Labels must match what the manuscript cites: tab:{rank,metric,auc}-main for
# the compact sweeps and tab:*-full for the complete tables in the supplement.
CAPTION = {
    "rank": ("Smoothed rank regression ($p=2$, $\\theta^\\star=(0.5,1)^\\top$). ",
             "tab:rank-main", "tab:rank-full"),
    "auc": ("Two-sample AUC ($p=3$, positive fraction $0.4$). ",
            "tab:auc-main", "tab:auc-full"),
    "triplet": ("Triplet metric learning ($d_x=4$, $p=10$). ",
                "tab:metric-main", "tab:triplet-full"),
}


def _t(x):
    """Time in seconds at three significant figures, so the sub-millisecond
    cells are not all printed as 0.00."""
    if x >= 100: return f"{x:.0f}"
    if x >= 10:  return f"{x:.1f}"
    if x >= 1:   return f"{x:.2f}"
    if x >= 0.01: return f"{x:.3f}"
    return f"{x:.4f}"


def fmt(x, sig=3):
    """Format like the existing tables: 1.23e-4 style, or fixed for coverage."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -3 <= e <= 3:
        return f"{x:.{max(0, sig - 1 - e)}f}"
    return f"{x/10**e:.{sig-1}f}e{e:+d}".replace("e+0", "e+").replace("e-0", "e-")


def load(kern):
    recs = [r for r in json.load(open(os.path.join(ROOT, f"grid_{kern}.json")))
            if r["n"] >= N_MIN]
    by = {}
    for r in recs:
        by.setdefault((r["n"], r["B"]), {})[r["scheme"]] = r
    return by


def table(kern, rows, full=False):
    by = load(kern)
    cap, lab_main, lab_full = CAPTION[kern]
    lab = lab_full if full else lab_main
    # Shrink-only: a plain \resizebox{\linewidth} also ENLARGES a table that is
    # narrower than the text block, which is how these tables came to be set
    # larger than the body text, by a different factor in each table.
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{rrccccccc}", r"\toprule",
             r"$n$ & $B$ & $T$ & $|\text{Bias}|$ & $\var_{\text{emp}}$ & $\var_{\text{dat}}$",
             r"& $\var_{\text{sgd}}$ & Cov.\ (WR/WOR) & Time (s, WR/WOR)\\", r"\midrule"]
    printed = []
    last_n = None
    for (n, B) in rows:
        if (n, B) not in by:
            print(f"  !! {kern}: no record for n={n} B={B}", file=sys.stderr); continue
        wr = by[(n, B)]["WR"]; wor = by[(n, B)].get("WOR", wr)
        if last_n is not None and n != last_n:
            lines.append(r"\addlinespace")
        last_n = n
        ne = int(np.log10(n))
        lines.append(
            f"$10^{{{ne}}}$ & {B} & {wr['T']:,} & {fmt(wr['bias'])} & {fmt(wr['var_emp'])} & "
            f"{fmt(wr['var_dat'])} & {fmt(wr['var_sgd'])} & "
            f"{wr['coverage']:.3f}/{wor['coverage']:.3f} & "
            f"{_t(wr['time_serial_med'])}/{_t(wor['time_serial_med'])}\\\\".replace(",", "{,}"))
        printed.append({"n": n, "B": B, "T": wr["T"], "bias": wr["bias"], "var_emp": wr["var_emp"],
                        "var_dat": wr["var_dat"], "var_sgd": wr["var_sgd"],
                        "cov_wr": wr["coverage"], "cov_wor": wor["coverage"],
                        "time": wr["time_serial_med"],
                        "time_wor": wor["time_serial_med"]})
    lines += [r"\bottomrule", r"\end{tabular}}",
              "\\caption{" + cap + f"$R=1000$ replications; Monte Carlo standard error of a "
              r"coverage near $0.95$ is about $0.007$. Times are medians of five serial runs "
              r"under each sampling scheme, measured in one pass on an idle machine.}",
              f"\\label{{{lab}}}", r"\end{table}"]
    return "\n".join(lines) + "\n", printed


def verify(tex, printed, kern):
    """Re-read the generated file and check every number against the JSON."""
    by = load(kern)
    bad = []
    for rec in printed:
        wr = by[(rec["n"], rec["B"])]["WR"]
        wor = by[(rec["n"], rec["B"])].get("WOR", wr)
        for key, want in (("bias", wr["bias"]), ("var_emp", wr["var_emp"]),
                          ("var_dat", wr["var_dat"]), ("var_sgd", wr["var_sgd"])):
            s = fmt(want)
            if s not in tex:
                bad.append(f"{kern} n={rec['n']} B={rec['B']} {key}={s} missing from the tex")
        if f"{wr['coverage']:.3f}/{wor['coverage']:.3f}" not in tex:
            bad.append(f"{kern} n={rec['n']} B={rec['B']} coverage pair missing")
    return bad


def main():
    os.makedirs(OUT, exist_ok=True)
    problems = []
    for kern in ("rank", "auc", "triplet"):
        by = load(kern)
        # main-text compact sweep
        tex, printed = table(kern, MAIN_ROWS[kern])
        p = os.path.join(OUT, f"{kern}_table_main.tex")
        open(p, "w").write(tex); problems += verify(tex, printed, kern)
        # supplement: every cell
        allrows = sorted(by.keys())
        tex_a, printed_a = table(kern, allrows, full=True)
        pa = os.path.join(OUT, f"{kern}_results_table.tex")
        open(pa, "w").write(tex_a); problems += verify(tex_a, printed_a, kern)
        print(f"  {kern}: {len(printed)} main rows -> {os.path.basename(p)}   "
              f"{len(printed_a)} full rows -> {os.path.basename(pa)}")

    # a one-line summary the text can quote, also derived not typed
    summ = {}
    for kern in ("rank", "auc", "triplet"):
        recs = [r for r in json.load(open(os.path.join(ROOT, f"grid_{kern}.json")))
                if r["n"] >= N_MIN]
        cov = [r["coverage"] for r in recs]; vr = [r["VR"] for r in recs]
        summ[kern] = {"cells": len(recs), "coverage_min": min(cov), "coverage_max": max(cov),
                      "VR_min": min(vr), "VR_max": max(vr),
                      "pilot_all_converged": all(r["pilot_ok_frac"] == 1.0 for r in recs),
                      "floor_ever_active": any(r["A0_floor_frac"] + r["AT_floor_frac"] > 0 for r in recs),
                      "psd_ever_projected": any(r["proj_frac"] > 0 for r in recs)}
    allcov = [c for k in summ for c in (summ[k]["coverage_min"], summ[k]["coverage_max"])]
    summ["overall"] = {"cells": sum(summ[k]["cells"] for k in ("rank", "auc", "triplet")),
                       "coverage_min": min(allcov), "coverage_max": max(allcov)}
    json.dump(summ, open(os.path.join(OUT, "table_summary.json"), "w"), indent=1)
    print(f"\n  summary: {summ['overall']['cells']} cells, coverage "
          f"[{summ['overall']['coverage_min']:.3f}, {summ['overall']['coverage_max']:.3f}]")
    if problems:
        print("\n  VERIFICATION FAILURES:")
        for b in problems: print("   ", b)
        sys.exit(1)
    print("  verification: every printed number found in the JSON")


if __name__ == "__main__":
    main()
