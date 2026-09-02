"""Main-paper data-analysis table generated from the two result JSON files."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
import paths as P
R = P.REAL_RESULTS

d = json.load(open(os.path.join(R, "diabetes_auc_results.json")))
f = json.load(open(os.path.join(R, "fremtpl2_rank_results.json")))

def num(x):
    return f"{int(x):,}".replace(",", "{,}")

rows = []
a = d["algorithm"]; drawn = a["B"] * a["T"]
rows.append(("30-day readmission", "Bipartite ranking", d["split"]["n_train"], d["p"],
             drawn, d["split"]["train_pairs"],
             f'{d["se_ratio_median"]:.3f} $[{d["se_ratio_iqr"][0]:.3f},'
             f'{d["se_ratio_iqr"][1]:.3f}]$', f'{d["fit_seconds"]:.2f}'))
a = f["algorithm"]; drawn = a["B"] * a["T"]
rows.append(("freMTPL2", "Rank regression", f["n"], f["p"],
             drawn, f["pairs"],
             f'{f["se_ratio_median"]:.3f} $[{f["se_ratio_range"][0]:.3f},'
             f'{f["se_ratio_range"][1]:.3f}]$', f'{f["fit_seconds"]:.2f}'))

L = [r"\begin{table}[H]",
     r"\tbl{Data analyses. The standard-error ratio is the single-run plug-in standard error divided by its resampling counterpart, summarized over coefficients. The bracketed spread is the interquartile range for 300 stratified bootstrap refits in the readmission analysis and the full range for 100 policy resamples in freMTPL2. The sample size is the fitting-sample size}{%",
     r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
     r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
     r"\begin{tabular}{llrrccc}", r"\toprule",
     r"Data set & Method & Fitting $n$ & $p$ & Tuples drawn / possible & SE ratio, median [spread] & Fit time (s)\\",
     r"\midrule"]
for nm, method, n, p_dim, drawn, possible, se, t in rows:
    L.append(f"{nm} & {method} & {num(n)} & {p_dim} & {num(drawn)} / {num(possible)} & {se} & {t}\\\\")
L += [r"\bottomrule", r"\end{tabular}}}", r"\label{tab:real-main}", r"\end{table}"]
out = os.path.join(P.FIGURES, "real_data_table_main.tex")
open(out, "w").write("\n".join(L) + "\n")
print("  wrote", os.path.basename(out))
for row in rows:
    print("   ", row[0], row[6])
