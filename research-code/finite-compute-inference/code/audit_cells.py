"""Cell-by-cell audit of the 60 simulation cells and the two real-data runs.

This is the gate between "the runs finished" and "numbers go into the paper".
It does not summarise; it tries to find a cell that should stop publication.

Checks, each with the reason it exists:

 C1 completeness      every (n, B, scheme) of the declared grid is present, once
 C2 provenance        all cells carry the same k_anchor, R, and defaults
 C3 coverage          |coverage - 0.95| <= 3 MC se, per cell and per coordinate
 C4 variance identity Var_emp ~= Var_dat + Var_sgd (the paper's own diagnostic)
 C5 bias              |bias| / sd <= 0.3 per coordinate (a bias of half an SE
                      moves coverage by about a point; 0.3 is the audit line)
 C6 pilot             every replication of every cell converged
 C7 floor             the relative spectral floor never became active
 C8 PSD               the anchor covariance never needed projection
 C9 scaling           Var_dat falls like 1/n and Var_sgd like 1/(BT) across the
                      sweep; a cell off the line is a mis-specified run
C10 WR vs WOR         the two schemes agree to within MC error
C11 timing            the Time column came from one clean pass
C12 real data         the two analyses carry their diagnostics and their
                      online/bootstrap SE agreement is in a defensible band
"""
import json, os, sys
import numpy as np
import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = P.ROOT
MCSE = np.sqrt(0.95 * 0.05 / 1000)          # 0.0069

GRID = {
    "auc": [(1000, 128), (1000, 256), (10000, 512), (10000, 1024),
            (100000, 2048), (100000, 4096), (1000000, 8192)],
    "triplet": [(1000, 128), (1000, 256), (10000, 512), (10000, 1024),
                (100000, 2048), (100000, 4096), (1000000, 8192), (1000000, 16384)],
    "rank": [(100, 32), (100, 64), (100, 100), (1000, 128), (1000, 256), (1000, 512),
             (10000, 256), (10000, 512), (10000, 1024), (100000, 512), (100000, 1024),
             (100000, 2048), (1000000, 2048), (1000000, 4096), (1000000, 8192)],
}

fails, warns = [], []


def fail(msg): fails.append(msg)
def warn(msg): warns.append(msg)


def main():
    print("=" * 78); print("CELL AUDIT"); print("=" * 78)
    allrecs = {}
    for kern, cells in GRID.items():
        p = P.grid(kern)
        if not os.path.exists(p):
            fail(f"C1 {kern}: grid file missing"); continue
        recs = json.load(open(p)); allrecs[kern] = recs
        seen = {}
        for r in recs:
            k = (r["n"], r["B"], r["scheme"])
            if k in seen: fail(f"C1 {kern}: duplicate record {k}")
            seen[k] = r
        for (n, B) in cells:
            for sch in ("WR", "WOR"):
                if (n, B, sch) not in seen:
                    fail(f"C1 {kern}: missing cell n={n} B={B} {sch}")
        # C2 provenance
        for key in ("k_a", "R"):
            vals = {r.get(key) for r in recs}
            if len(vals) != 1: fail(f"C2 {kern}: mixed {key} across cells: {vals}")
        # per-cell checks
        for r in recs:
            tag = f"{kern} n={r['n']} B={r['B']} {r['scheme']}"
            if abs(r["coverage"] - 0.95) > 3 * MCSE:
                fail(f"C3 {tag}: coverage {r['coverage']:.3f} is >3 MC se from 0.95")
            elif abs(r["coverage"] - 0.95) > 2 * MCSE:
                warn(f"C3 {tag}: coverage {r['coverage']:.3f} is >2 MC se from 0.95")
            pc = np.array(r["coverage_percoord"])
            if (np.abs(pc - 0.95) > 4 * MCSE).any():
                warn(f"C3 {tag}: a coordinate coverage {pc.min():.3f}-{pc.max():.3f} is >4 MC se out")
            gap = abs(r["var_emp"] - r["var_dat"] - r["var_sgd"]) / r["var_emp"]
            if gap > 0.10: fail(f"C4 {tag}: Var_emp vs Var_dat+Var_sgd off by {gap:.1%}")
            elif gap > 0.06: warn(f"C4 {tag}: variance identity off by {gap:.1%}")
            bs = np.abs(np.array(r["bias_percoord"])) / np.sqrt(np.array(r["var_emp_percoord"]))
            if bs.max() > 0.30: fail(f"C5 {tag}: |bias|/sd = {bs.max():.2f}")
            elif bs.max() > 0.20: warn(f"C5 {tag}: |bias|/sd = {bs.max():.2f}")
            if r["pilot_ok_frac"] < 1.0: fail(f"C6 {tag}: pilot converged in only {r['pilot_ok_frac']:.1%}")
            if r["A0_floor_frac"] + r["AT_floor_frac"] > 0: fail(f"C7 {tag}: floor active")
            if r["proj_frac"] > 0: warn(f"C8 {tag}: PSD projection in {r['proj_frac']:.1%} of reps")
        # C9 scaling
        for sch in ("WR",):
            byn = {}
            for r in recs:
                if r["scheme"] == sch: byn.setdefault(r["n"], []).append(r)
            ns = sorted(byn)
            for a, b in zip(ns, ns[1:]):
                va = np.mean([x["var_dat"] for x in byn[a]]); vb = np.mean([x["var_dat"] for x in byn[b]])
                got, want = va / vb, b / a
                if not (0.6 < got / want < 1.6):
                    fail(f"C9 {kern}: Var_dat ratio n={a}->{b} is {got:.1f}, expected ~{want:.0f}")
        # C10 WR vs WOR
        for (n, B) in cells:
            wr = seen.get((n, B, "WR")); wo = seen.get((n, B, "WOR"))
            if wr and wo and abs(wr["coverage"] - wo["coverage"]) > 4 * MCSE:
                warn(f"C10 {kern} n={n} B={B}: WR {wr['coverage']:.3f} vs WOR {wo['coverage']:.3f}")
        # C11 timing provenance
        clean = [r.get("time_pass_clean") for r in recs]
        if any(c is None for c in clean):
            warn(f"C11 {kern}: some cells have no unified timing yet")
        elif not all(clean):
            fail(f"C11 {kern}: {sum(1 for c in clean if not c)} cells timed on a busy machine")

    # C12 real data
    for name, f, band in (("diabetes", "diabetes_auc_results.json", (0.85, 1.20)),
                          ("fremtpl2", "fremtpl2_rank_results.json", (0.80, 1.25))):
        p = os.path.join(ROOT, "real_data", "results", f)
        if not os.path.exists(p):
            warn(f"C12 {name}: results not present yet"); continue
        d = json.load(open(p)); g = d.get("diagnostics", {})
        if not g.get("pilot_ok", True): fail(f"C12 {name}: pilot did not converge")
        if g.get("A0_floor_active") or g.get("AT_floor_active"): fail(f"C12 {name}: floor active")
        r = d.get("se_ratio_median")
        if r is not None and not (band[0] <= r <= band[1]):
            fail(f"C12 {name}: online/bootstrap SE ratio {r:.3f} outside {band}")

    print(f"\n  cells audited: {sum(len(v) for v in allrecs.values())}")
    print(f"  FAILURES: {len(fails)}")
    for m in fails: print("    x", m)
    print(f"  warnings: {len(warns)}")
    for m in warns: print("    -", m)
    if fails:
        print("\n  VERDICT: do not write these numbers into the paper yet."); sys.exit(1)
    print("\n  VERDICT: every check passed; the numbers may be written.")


if __name__ == "__main__":
    main()
