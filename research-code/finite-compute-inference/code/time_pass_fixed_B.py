"""Serial timing pass for the fixed-minibatch grid, both sampling schemes.

    python3 -u time_pass_fixed_B.py [reps] [--force]

Separate from the grid for the reason the archived pass is separate: coverage,
bias and variance are load-independent and may be computed in parallel, but a
wall-clock number is comparable only with another measured under the same
conditions.  This measures every reported cell one after another on an idle
machine, in one pass, under one code version, and refuses to start if the
machine is busy.

At B = 1 the two schemes are one computation -- a size-one minibatch drawn
without replacement is the with-replacement draw -- so that cell is timed once
and the same number stands for both.

Writes experiments_fixed_B/aggregated_results/timing_fixed_B.json.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import experiment_design as D
import run_grid as G
import run_fixed_B_grid as F
from time_pass import machine_state, idle_enough, IDLE_CPU_MAX, IDLE_FREE_GB_MIN

OUT = os.path.join(P.ROOT, "experiments_fixed_B", "aggregated_results",
                   "timing_fixed_B.json")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 5
    force = "--force" in sys.argv
    st = machine_state()
    print(f"  cpu {st['cpu_percent']:.0f}%  load {st['load1']:.1f}  "
          f"free {st['ram_free_gb']:.1f} GB", flush=True)
    if not idle_enough(st) and not force:
        print(f"  NOT IDLE (need cpu<={IDLE_CPU_MAX}% and free>={IDLE_FREE_GB_MIN} GB); "
              f"pass --force to time anyway"); sys.exit(1)

    cells = [(k, n, B, s) for k in ("rank", "auc", "triplet") for n in D.N_GRID
             for B in D.B_GRID for s in (("WR",) if B == 1 else ("WR", "WOR"))]
    # one dataset build per timed replication is part of the cost the tables
    # describe, so it is inside the timed region, exactly as in the runs
    print(f"  timing {len(cells)} cells x {reps} reps, serial\n", flush=True)
    print(f"  {'kernel':<8}{'n':>8}{'B':>6}{'sch':>5}{'median s':>11}{'min':>9}{'max':>9}", flush=True)
    rows = []
    for kern, n, B, scheme in cells:
        t = []
        for r in range(reps):
            # Time the run the tables report.  Calling run_grid.one would time
            # the ARCHIVED driver instead: it sets the pilot from the legacy
            # rule and the auxiliary budgets per B, so at n = 10^3, B = 100 it
            # performs 50 updates where the reported run performs 98.  The
            # replication path below is the one every other column comes from.
            s0 = time.perf_counter()
            F.replicate(kern, n, [B], None, scheme, 900_000 + r)
            t.append(time.perf_counter() - s0)
        rows.append({"kernel": kern, "n": n, "B": B, "scheme": scheme, "reps": reps,
                     "median_s": float(np.median(t)), "min_s": float(min(t)),
                     "max_s": float(max(t))})
        print(f"  {kern:<8}{n:>8}{B:>6}{scheme:>5}{np.median(t):>11.3f}"
              f"{min(t):>9.3f}{max(t):>9.3f}", flush=True)
    json.dump({"machine": st, "reps": reps, "cells": rows}, open(OUT, "w"), indent=1)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
