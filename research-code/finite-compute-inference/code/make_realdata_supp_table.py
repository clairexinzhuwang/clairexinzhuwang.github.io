"""Supplementary real-data table, generated from the stored result files.

The main text reports the agreement between the single-run plug-in standard error
and its resampling counterpart; this table adds the scale of the problem, how
little of the tuple set is touched, and the cost of a single fit.  Every entry
is read from the analysis output rather than typed in, and the script fails
loudly if a field it needs is absent.

Run:  python3 -u make_realdata_supp_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
RES = P.REAL_RESULTS
OUT = os.path.join(P.FIGURES, "real_data_supp_table.tex")


def thin(x):
    return f"{x:,}".replace(",", "{,}")


def main():
    d = json.load(open(os.path.join(RES, "diabetes_auc_results.json")))
    f = json.load(open(os.path.join(RES, "fremtpl2_rank_results.json")))

    # readmission: the pair set is positives x negatives on the training split
    n_tr = d["split"]["n_train"]
    pairs_d = d["split"]["train_pairs"]
    drawn_d = d["algorithm"]["B"] * d["algorithm"]["T"]
    pairs_f = f["pairs"]
    drawn_f = f["algorithm"]["B"] * f["algorithm"]["T"]

    rows = [
        ("30-day readmission", "AUC, $K=2$", n_tr, d["p"], pairs_d, drawn_d,
         d["se_ratio_median"], d["se_ratio_iqr"], "interquartile range",
         d["fit_seconds"]),
        ("freMTPL2 severity", "Rank, $d=2$", f["n"], f["p"], pairs_f, drawn_f,
         f["se_ratio_median"], f["se_ratio_range"], "range",
         f["fit_seconds"]),
    ]

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llrrrrccr}", r"\toprule",
             r"Data & Kernel & $n$ & $p$ & Tuples & Drawn & \multicolumn{2}{c}{SE ratio}"
             r" & Fit (s)\\",
             r"\cmidrule(lr){7-8}",
             r" & & & & & & median & spread & \\",
             r"\midrule"]
    for nm, kn, n, p, poss, drawn, med, spread, _lab, secs in rows:
        lines.append(
            f"{nm} & {kn} & {thin(n)} & {p} & {thin(poss)} & {thin(drawn)} & "
            f"{med:.3f} & $[{spread[0]:.3f},{spread[1]:.3f}]$ & {secs:.1f}\\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        "\\caption{Real-data summary. Tuples is the size of the complete tuple "
        "set and Drawn is the number the run evaluates, $BT$ at the prescribed "
        "schedule $T=\\lfloor (n/B)\\log_2 n\\rfloor$. The SE ratio is the "
        "single-run plug-in standard error divided by its resampling counterpart, "
        "over the coefficients; the spread is the interquartile range for the "
        "readmission fit and the full range for freMTPL2. Fit is the single "
        "run, excluding the resampling used only for the comparison.}",
        r"\label{tab:real-supp}", r"\end{table}"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"  wrote {OUT}")
    for nm, _k, n, p, poss, drawn, med, spread, lab, secs in rows:
        print(f"    {nm:<20} n={n:,} p={p} drawn={drawn:,} "
              f"({100 * drawn / poss:.2f}% of {poss:,})  "
              f"SE ratio {med:.3f} {lab} [{spread[0]:.3f},{spread[1]:.3f}]  "
              f"{secs:.1f}s")


if __name__ == "__main__":
    main()
