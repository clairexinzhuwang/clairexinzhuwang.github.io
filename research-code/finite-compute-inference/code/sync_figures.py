"""Copy generated tables and figures into the manuscript tree.

The manuscript inputs some files under names that differ from the generator's
output (the triplet table is input as metric_table_main.tex), so the mapping is
written out explicitly here and every target is verified to exist afterwards.
A silent miss leaves a stale table in the paper, which is the failure this
script is meant to make impossible.

The manuscript lives outside this package, so its location is taken from the
command line, or from $MANUSCRIPT_DIR, or found by looking for a sibling
directory that holds a main.tex.  No path is written into this file: one would
be wrong on any other machine, and it would carry its author's name into a
repository that is meant to be anonymous.
"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
SRC = P.FIGURES


def manuscript_dir(argv):
    """Where the .tex sources are: argument, environment, then a sibling."""
    if len(argv) > 1:
        return os.path.abspath(argv[1])
    env = os.environ.get("MANUSCRIPT_DIR")
    if env:
        return os.path.abspath(env)
    parent = os.path.dirname(P.ROOT)
    for name in sorted(os.listdir(parent)):
        cand = os.path.join(parent, name)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "main.tex")):
            return cand
    sys.exit("  no manuscript directory found: pass one as an argument, "
             "set MANUSCRIPT_DIR, or place this package beside the sources")

MAP = {
    # The fixed-minibatch study.  These are the only files this step owns.
    "rank_fixedB_table.tex":    "rank_fixedB_table.tex",
    "triplet_fixedB_table.tex": "triplet_fixedB_table.tex",
    "auc_fixedB_table.tex":     "auc_fixedB_table.tex",
    "wor_fixedB_table.tex":     "wor_fixedB_table.tex",
    "data_only_fixedB_table.tex": "data_only_fixedB_table.tex",
    "runtime_fixedB_table.tex": "runtime_fixedB_table.tex",
    "simulation_compact_main.tex": "simulation_compact_main.tex",
    "budget_ratio_main.tex":    "budget_ratio_main.tex",
    "budget_ratio_fixedB_table.tex": "budget_ratio_fixedB_table.tex",
    "fig_fixedB_bka.pdf":       "fig_fixedB_bka.pdf",
}

# NOT synced, deliberately.  The archived growing-B tables and the real-data
# tables in the typeset source are written with the journal's \tbl macro,
# which puts the caption above the rule; make_tables.py emits the \caption form
# of the older working copy.  Copying one over the other silently changes the
# house style of a table that is already typeset correctly, which is what
# happened once.  Regenerate those in the working copy and convert, or edit the
# typeset source directly; do not route them through here.
NOT_SYNCED = (
    "rank_table_main.tex", "auc_table_main.tex", "metric_table_main.tex",
    "real_data_table_main.tex", "rank_results_table.tex", "auc_results_table.tex",
    "triplet_results_table.tex", "real_data_supp_table.tex",
    "fig_combined.pdf", "fig_combined_bka.pdf",
    "figS_decomposition.png", "figS_normal_qq.png", "figS_rank_traces.png",
)


def main():
    dst_root = manuscript_dir(sys.argv)
    DST = os.path.join(dst_root, "figures")
    if not os.path.isdir(DST):
        sys.exit(f"  no figures directory under {dst_root}")
    print(f"  manuscript: {dst_root}")
    missing, copied = [], []
    for src_name, dst_name in MAP.items():
        s = os.path.join(SRC, src_name)
        if not os.path.exists(s):
            missing.append(src_name); continue
        d = os.path.join(DST, dst_name)
        before = hashlib.sha256(open(d, "rb").read()).hexdigest() if os.path.exists(d) else None
        shutil.copy(s, d)
        after = hashlib.sha256(open(d, "rb").read()).hexdigest()
        copied.append((dst_name, before != after))
    for name, changed in copied:
        print(f"  {'updated' if changed else 'same   '}  {name}")
    if missing:
        print(f"  MISSING from {SRC}: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"  {sum(c for _, c in copied)} of {len(copied)} files changed")


if __name__ == "__main__":
    main()
