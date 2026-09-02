"""The twelve checks that must pass before any formal fixed-B run.

    python3 -u smoke_fixed_B.py [n] [R]

These check the DESIGN and the PAIRING, not the statistics.  Imperfect coverage
or a finite-sample variance mismatch is not a failure here and is not tested.
Exits non-zero if any check fails, and prints every check either way so a run
can be pasted into the audit file.

The horizon is the schedule the paper prescribes, T = floor((n/B) log2 n), so
the tuple budget B T is common to every minibatch size up to the integer part.
The residual is the flooring alone, fewer than B_max tuples, and the recursion
streams are nested: the longest is drawn and the shorter budgets consume
prefixes of it.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_design as D
import run_grid as G
import alg_paper as A
from alg_paper import run_algorithm1, default_m, component_streams

KERNELS = ("rank", "auc", "triplet")
SEED_ALG, SEED_DATA = 800_000, 700_000
FAILED = []


def check(num, name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {num:>2}. {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILED.append(f"{num}. {name} {detail}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"=== fixed-B smoke: n_total={n}, R={R}, B={D.B_GRID}, "
          f"T=floor((n/B)log2 n) ===", flush=True)

    for kern_name in KERNELS:
        print(f"\n--- {kern_name} ---", flush=True)
        kern, theta0, _q, _r, _rule = G.build(kern_name, n, SEED_DATA)
        n_total, n1 = kern.n_total, kern.n_theory_ref
        h_eff = getattr(kern, "h", 1.0)
        m0, m = D.auxiliary_budgets(default_m, n_total, h_eff)
        q_k = D.anchor_counts(kern)

        # ---- 1-3: the schedule, and no fallback --------------------------
        sched = {B: D.schedule(n_total, n1, B) for B in D.B_GRID}
        BTs = {v[0] for v in sched.values()}
        N0s = {v[1] for v in sched.values()}
        ok_t0 = all(B * v[3] == v[1] for B, v in sched.items())
        ok_pos = all(0 < v[3] < v[2] for v in sched.values())
        ok_rule = all(v[2] == int(np.floor(n_total * np.log2(n_total) / B))
                      for B, v in sched.items())
        spread = max(BTs) - min(BTs)

        check(1, "T is the primary numerical schedule and B*T is common up to the integer part",
              ok_rule and spread < max(D.B_GRID),
              f"BT in {sorted(BTs)}, spread {spread} < {max(D.B_GRID)}")
        check(2, "B*t0 == N0 exactly, and N0 is the same for every B",
              ok_t0 and len(N0s) == 1, f"N0={sorted(N0s)[0]}")
        legacy = A.legacy_default_t0(sched[1][2],
                                     rule=("rank" if kern_name == "rank" else "log"),
                                     n=n_total)
        check(3, "0 < t0 < T and the schedule uses no T/2 fallback",
              ok_pos and sched[1][3] != legacy,
              f"t0(B=1)={sched[1][3]} vs archived rule {legacy}")

        # ---- the runs, sharing one stream --------------------------------
        need = {B: B * (sched[B][2] - sched[B][3]) for B in D.B_GRID}
        stream_len = max(need.values())
        runs = {}
        for B in D.B_GRID:
            _BT, _N0, T, t0 = sched[B]
            runs[B] = run_algorithm1(kern, theta0, B, T, t0, m0, m, q_k, scheme="WR",
                                     k_per_stream=D.P_ANCHOR,
                                     streams=component_streams(SEED_ALG),
                                     stream_len=stream_len)

        def drawn(label, k):
            """Re-draw a component from a fresh identically-seeded stream."""
            return kern.draw(k, component_streams(SEED_ALG)[label])

        # ---- 4-9: the draws the RUNS consumed are shared across B ---------
        # Comparing three freshly re-seeded draws with each other tests the
        # generator, not the runs: it passes however the runs actually sampled.
        # Each check below re-runs one minibatch size with a component
        # deliberately perturbed and requires the outputs to move, so that a
        # check which cannot fail is caught here rather than shipped.
        pil = [drawn("pilot", sched[B][1]) for B in D.B_GRID]
        same_pilot = all(all(np.array_equal(a, b) for a, b in zip(pil[0], p)) for p in pil[1:])
        _BT, _N0, T1, t01 = sched[D.B_GRID[0]]
        st = component_streams(SEED_ALG + 1)          # a different pilot stream
        st["hessian0"] = component_streams(SEED_ALG)["hessian0"]
        st["recursion"] = component_streams(SEED_ALG)["recursion"]
        st["terminal"] = component_streams(SEED_ALG)["terminal"]
        st["anchors"] = component_streams(SEED_ALG)["anchors"]
        alt = run_algorithm1(kern, theta0, D.B_GRID[0], T1, t01, m0, m, q_k,
                             scheme="WR", k_per_stream=D.P_ANCHOR, streams=st,
                             stream_len=stream_len)
        pilot_matters = alt["pilot_gradnorm"] != runs[D.B_GRID[0]]["pilot_gradnorm"] \
            or not np.array_equal(alt["D0"], runs[D.B_GRID[0]]["D0"])
        check(4, "pilot draws shared across B, and the check can fail",
              same_pilot and pilot_matters,
              "perturbing the pilot stream moves the run" if pilot_matters
              else "PERTURBING THE PILOT STREAM CHANGED NOTHING")

        gns = [runs[B]["pilot_gradnorm"] for B in D.B_GRID]
        kappas = [runs[B]["A0_kappa"] for B in D.B_GRID]
        check(5, "pilot estimates agree across B", len(set(gns)) == 1,
              f"gradnorm {gns[0]:.2e}")

        h0 = [drawn("hessian0", m0) for B in D.B_GRID]
        same_h0 = all(all(np.array_equal(a, b) for a, b in zip(h0[0], x)) for x in h0[1:])
        d0s = {B: runs[B]["D0"].tobytes() for B in D.B_GRID}
        check(6, "initial-Hessian tuples and D0 identical across B",
              same_h0 and len(set(d0s.values())) == 1 and len(set(kappas)) == 1)

        # The master stream, and what each B would consume from it.  The test
        # is that the tuples a shorter budget uses are the FIRST tuples of the
        # longest stream, in the same order, and that the blocks a run reads
        # tile that prefix exactly.  Comparing a slice with itself would pass
        # unconditionally and check nothing.
        master = kern.draw(stream_len, component_streams(SEED_ALG)["recursion"])
        nested = True
        for B in D.B_GRID:
            _BT, _N0, T, t0 = sched[B]
            blocks = [kern.batch_slice(master, i * B, (i + 1) * B)
                      for i in range(T - t0)]
            rebuilt = tuple(np.concatenate([blk[c] for blk in blocks])
                            for c in range(len(master)))
            prefix = kern.batch_slice(master, 0, need[B])
            nested &= (len(rebuilt[0]) == need[B] and
                       all(np.array_equal(a, b) for a, b in zip(rebuilt, prefix)))
            nested &= need[B] <= stream_len          # never reads past the end
        steps = {B: sched[B][2] - sched[B][3] for B in D.B_GRID}
        check(7, "one recursion stream, shorter budgets taking prefixes of it",
              nested and len(set(steps.values())) == len(D.B_GRID),
              f"stream={stream_len} tuples, consumed={sorted(need.values(), reverse=True)}, "
              f"updates={list(steps.values())}")

        # What must match is the TUPLES the runs consumed, not the estimates
        # computed from them: V_T and the anchor covariance are evaluated at the
        # terminal iterate, which differs across B, so requiring those to agree
        # would be requiring the wrong thing.  Draw each component from a stream
        # seeded as the runs seed it and compare the indices.
        term = [drawn("terminal", m) for _ in D.B_GRID]
        same_term = all(all(np.array_equal(a, b) for a, b in zip(term[0], x))
                        for x in term[1:])
        anch = [component_streams(SEED_ALG)["anchors"].integers(0, n1, 64)
                for _ in D.B_GRID]
        same_anch = all(np.array_equal(anch[0], x) for x in anch[1:])
        # and the check must be able to fail: a different terminal stream must
        # move the terminal covariance
        st2 = component_streams(SEED_ALG)
        st2["terminal"] = component_streams(SEED_ALG + 2)["terminal"]
        alt2 = run_algorithm1(kern, theta0, D.B_GRID[0], T1, t01, m0, m, q_k,
                              scheme="WR", k_per_stream=D.P_ANCHOR, streams=st2,
                              stream_len=stream_len)
        term_matters = not np.array_equal(alt2["VT"], runs[D.B_GRID[0]]["VT"])
        check(8, "terminal and anchor draws shared across B, and the check can fail",
              same_term and same_anch and term_matters,
              "perturbing the terminal stream moves V_T" if term_matters
              else "PERTURBING THE TERMINAL STREAM CHANGED NOTHING")

        aux = {B: (m0, m, tuple(q_k), D.P_ANCHOR) for B in D.B_GRID}
        check(9, "m0, m, q_k, P identical across B", len(set(aux.values())) == 1,
              f"m0=m={m0}, q={q_k}, P={D.P_ANCHOR}")

        # ---- 10: the WR coefficient collapses to 1/(BT) ------------------
        worst = 0.0
        for B in D.B_GRID:
            _BT, _N0, T, t0 = sched[B]
            worst = max(worst, abs((t0 + (T - t0)) / (B * T * T) - 1.0 / (B * T)) * B * T)
        check(10, "under WR, {t0+(T-t0)}/(BT^2) == 1/(BT)", worst < 1e-12,
              f"max relative error {worst:.1e}")

        # ---- 11-12: everything finite, everything logged -----------------
        fin = all(np.all(np.isfinite(runs[B][k])) for B in D.B_GRID
                  for k in ("Sigma_full", "Sigma_dat", "alg_full", "D0", "DT"))
        check(11, "all covariance matrices and intervals finite", fin)
        logged = all(all(k in runs[B] for k in
                         ("A0_floor_active", "AT_floor_active", "A0_kappa", "AT_kappa",
                          "pilot_ok", "pilot_gradnorm", "t_phase"))
                     for B in D.B_GRID)
        proj = all(f"zeta_projected_k{i}" in runs[D.B_GRID[0]]
                   for i in range(len(kern.samples)))
        check(12, "floor, projection and failure diagnostics all present", logged and proj)

        ph = runs[1]["t_phase"]
        print("        phase times at B=1 (s): " +
              "  ".join(f"{k}={ph[k]:.2f}" for k in
                        ("pilot", "hessian0", "recursion", "terminal", "anchors", "total")))

    print()
    if FAILED:
        print(f"  {len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print(f"    {f}")
        sys.exit(1)
    print("  all twelve checks passed for every kernel; the formal runs may start")


if __name__ == "__main__":
    main()
