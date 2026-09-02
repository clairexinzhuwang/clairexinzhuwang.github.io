"""Timing pass: re-measure the Time column for EVERY cell in one go, serially,
on an idle machine, under one code version.

Why separate from run_grid.py: the statistics (coverage, bias, variance) are
load-independent and may be computed in parallel and resumed after
interruptions, but a wall-clock number is only comparable to another wall-clock
number measured under the same conditions.  The per-cell timings collected
during the parallel grid were taken across several days, worker counts, a
machine crash and a code change, so they are not mutually comparable.  This
script replaces all of them with one internally consistent set.

It refuses to run if the machine is busy, records the machine state next to the
numbers, and re-checks that state after every cell so a background job that
starts mid-pass is visible in the output rather than silently inflating it.

Usage:  python3 -u time_pass.py [reps] [--force]
Writes ../timing_pass.json and patches time_serial_med in the grid files
(keeping the original value as time_serial_med_gridpass).
"""
import json, os, sys, time, platform, subprocess
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grid as G

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths as P
OUT = os.path.join(P.RESULTS, "timing_pass.json")
IDLE_CPU_MAX = 25.0        # % system-wide CPU over a 3 s sample
IDLE_FREE_GB_MIN = 4.0


def machine_state(sample_s=3.0):
    cpu = psutil.cpu_percent(interval=sample_s)
    vm = psutil.virtual_memory(); sw = psutil.swap_memory()
    load1, load5, load15 = os.getloadavg()
    busy = [p.info["name"] for p in psutil.process_iter(["name", "cpu_percent"])
            if (p.info.get("cpu_percent") or 0) > 40][:6]
    return {"cpu_percent": cpu, "load1": load1, "load5": load5, "load15": load15,
            "ram_free_gb": vm.available / 2**30, "ram_percent": vm.percent,
            "swap_used_gb": sw.used / 2**30, "busy_processes": busy,
            "cores": os.cpu_count()}


def idle_enough(st):
    return st["cpu_percent"] <= IDLE_CPU_MAX and st["ram_free_gb"] >= IDLE_FREE_GB_MIN


def code_fingerprint():
    fp = {}
    for f in ("alg_paper.py", "kernels.py", "run_grid.py"):
        p = os.path.join(HERE, f)
        fp[f] = subprocess.run(["shasum", "-a", "256", p], capture_output=True,
                               text=True).stdout.split()[0][:16]
    return fp


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 5
    force = "--force" in sys.argv

    start_state = machine_state()
    print(f"machine: {platform.platform()}  {start_state['cores']} cores", flush=True)
    print(f"  cpu {start_state['cpu_percent']:.0f}%  load {start_state['load1']:.1f}  "
          f"free {start_state['ram_free_gb']:.1f} GB  swap {start_state['swap_used_gb']:.1f} GB", flush=True)
    if not idle_enough(start_state):
        print(f"  NOT IDLE (need cpu<={IDLE_CPU_MAX}% and free>={IDLE_FREE_GB_MIN} GB).", flush=True)
        if start_state["busy_processes"]:
            print("  busy:", ", ".join(start_state["busy_processes"]), flush=True)
        if not force:
            print("  aborting; pass --force to time anyway (numbers will not be comparable).")
            sys.exit(1)
        print("  --force given: proceeding on a LOADED machine; results flagged.", flush=True)

    cells = []
    for kern in ("rank", "auc", "triplet"):
        p = P.grid(kern)
        if not os.path.exists(p):
            print(f"  (no grid_{kern}.json, skipping)"); continue
        for rec in json.load(open(p)):
            cells.append((kern, rec["n"], rec["B"], rec["scheme"]))
    print(f"timing {len(cells)} cells x {reps} reps, serial\n", flush=True)

    rows = []
    print(f"{'kernel':<8}{'n':>8}{'B':>7}{'sch':>5}{'median s':>10}{'min':>8}{'max':>8}"
          f"{'cpu%':>7}{'free GB':>9}", flush=True)
    for kern, n, B, scheme in cells:
        t = []
        for r in range(reps):
            s = time.perf_counter()
            G.one(kern, n, B, scheme, 90000 + r)
            t.append(time.perf_counter() - s)
        st = machine_state(sample_s=0.5)
        rows.append({"kernel": kern, "n": n, "B": B, "scheme": scheme, "reps": reps,
                     "median_s": float(np.median(t)), "min_s": float(min(t)), "max_s": float(max(t)),
                     "all_s": [float(x) for x in t],
                     "cpu_percent_after": st["cpu_percent"], "ram_free_gb_after": st["ram_free_gb"],
                     "clean": bool(idle_enough(st))})
        flag = "" if rows[-1]["clean"] else "   <- MACHINE BUSY DURING THIS CELL"
        print(f"{kern:<8}{n:>8}{B:>7}{scheme:>5}{np.median(t):>10.3f}{min(t):>8.3f}{max(t):>8.3f}"
              f"{st['cpu_percent']:>7.0f}{st['ram_free_gb']:>9.1f}{flag}", flush=True)

    end_state = machine_state()
    out = {"reps": reps, "n_cells": len(rows),
           "code_sha256_16": code_fingerprint(),
           "machine_start": start_state, "machine_end": end_state,
           "all_cells_clean": all(r["clean"] for r in rows) and idle_enough(start_state),
           "timings": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    dirty = [r for r in rows if not r["clean"]]
    print(f"  clean: {len(rows)-len(dirty)}/{len(rows)} cells" +
          (f"   DIRTY: {[(r['kernel'],r['n'],r['B'],r['scheme']) for r in dirty]}" if dirty else ""))

    # patch the grid files, keeping the original grid-pass timing
    for kern in ("rank", "auc", "triplet"):
        p = P.grid(kern)
        if not os.path.exists(p): continue
        recs = json.load(open(p)); byk = {(r["kernel"], r["n"], r["B"], r["scheme"]): r for r in rows}
        for rec in recs:
            k = (kern, rec["n"], rec["B"], rec["scheme"])
            if k in byk:
                rec.setdefault("time_serial_med_gridpass", rec.get("time_serial_med"))
                rec["time_serial_med"] = byk[k]["median_s"]
                rec["time_pass_clean"] = byk[k]["clean"]
        json.dump(recs, open(p, "w"), indent=1)
        print(f"  patched grid_{kern}.json")


if __name__ == "__main__":
    main()
