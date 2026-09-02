#!/usr/bin/env python3
"""ONE streaming pass over the 720 formal jsonl files -> canonical per-(cell,route) table.

Conventions inherited from independent_audit_v26_1f/analyze_formal_grid.py and from the
producer (work/v26_1f/run_formal_experiment.py, work/v26_1f/formal_contract.py):

  * status in {ok, selection_failure_empty, numerical_failure, program_or_schema_error};
    status == 'ok'  <=>  interval_reportable is True  (formal_contract.outcome_taxonomy).
  * NON-OK RECORDS ARE NEITHER CONTAINED NOR RECOVERED.  A naive
    set(sel or []) == set(sup or []) returns True for numerical_failure records
    (selected=None, support=None) and is therefore WRONG; we count it separately
    (n_naive_recovery_bug) purely to quantify the size of that error.
  * The producer boolean 'exact_recovery' is the authority; we recompute it on every
    ok record and record any disagreement (n_recovery_field_vs_recomputed_mismatch).
  * Route is taken from the exact field r['route'] with ==.  NEVER endswith('known_s'),
    which also matches 'unknown_s'.
  * interior/boundary is the pre-locked C2 scope: a CELL-level frozen classification
    (work/v26_1f/C2_BOUNDARY_CELLS.json, 23 boundary cells), not a per-record field.
"""
import json, os, sys, glob, math
from collections import defaultdict
from multiprocessing import Pool

import sys as _sys
if len(_sys.argv) < 4:
    _sys.exit("usage: python3 build_canonical_cell_table.py <records_root> <out_dir> <c2_boundary_json>\n"
              "  <records_root>: directory holding the 720 task_*.jsonl formal record files\n"
              "  <out_dir>: where canonical_cell_table.json/.csv are written\n"
              "  <c2_boundary_json>: work tree's C2_BOUNDARY_CELLS.json")
ROOT, OUTDIR, C2 = _sys.argv[1], _sys.argv[2], _sys.argv[3]


FIELDS_INT = [
    "n_records", "n_ok", "n_numerical_failure", "n_selection_failure_empty",
    "n_program_or_schema_error", "n_other_status",
    "n_empty_selection", "n_selection_nonempty_field",
    "n_containment_candidate", "n_containment_candidate_raw_field",
    "n_containment_selected", "n_containment_selected_raw_naive",
    "n_exact_recovery", "n_exact_recovery_recomputed_ok_only",
    "n_recovery_field_vs_recomputed_mismatch", "n_naive_recovery_bug",
    "n_strict_simultaneous_coverage", "n_coverage_conditional_numerator",
    "n_coverage_numerator_not_subset_of_recovery",
    "n_computational_failure", "n_interval_reportable", "n_total_failure",
    "false_negative_count_reportable", "false_positive_count_reportable",
    "n_records_with_any_false_negative", "n_records_with_any_false_positive",
    "n_width_used", "n_width_missing_among_ok",
    "n_tail_boundary_hits_positive", "n_active_joint_coverage_field",
]


def blank():
    d = {k: 0 for k in FIELDS_INT}
    d["sum_width"] = 0.0
    d["sumsq_width"] = 0.0
    d["meta"] = None
    return d


def process_file(path):
    acc = defaultdict(blank)
    with open(path, "r") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            route = r["route"]
            assert route in ("known_s", "unknown_s"), route
            key = (r["cell_id"], route)
            a = acc[key]
            if a["meta"] is None:
                a["meta"] = dict(cell_id=r["cell_id"], route=route, problem=r["problem"],
                                 panel=r.get("panel"), N=r["N"], p=r["p"], s=r["s"],
                                 K=r.get("K"), s_upper_bound=r.get("s_upper_bound"),
                                 design=r["design"], law=r["law"],
                                 sparsity_information=r.get("sparsity_information"),
                                 cell_index=r.get("cell_index"))
            a["n_records"] += 1
            status = r.get("status")
            ok = (status == "ok")
            if ok:
                a["n_ok"] += 1
            elif status == "numerical_failure":
                a["n_numerical_failure"] += 1
            elif status == "selection_failure_empty":
                a["n_selection_failure_empty"] += 1
            elif status == "program_or_schema_error":
                a["n_program_or_schema_error"] += 1
            else:
                a["n_other_status"] += 1

            if r.get("selection_empty") is True:
                a["n_empty_selection"] += 1
            if r.get("selection_nonempty") is True:
                a["n_selection_nonempty_field"] += 1
            if r.get("interval_reportable") is True:
                a["n_interval_reportable"] += 1
            if r.get("total_failure") is True:
                a["n_total_failure"] += 1
            if r.get("computational_failure") is True:
                a["n_computational_failure"] += 1

            sel, sup, cand = r.get("selected"), r.get("support"), r.get("candidate")
            # --- containment on the HTP candidate set (producer field) ---
            if r.get("candidate_contains_truth") is True:
                a["n_containment_candidate_raw_field"] += 1
                if ok:
                    a["n_containment_candidate"] += 1
            # --- containment on the reported selected set ---
            if sel is not None and sup is not None and set(sup).issubset(set(sel)):
                a["n_containment_selected_raw_naive"] += 1
                if ok:
                    a["n_containment_selected"] += 1
            # --- recovery ---
            if r.get("exact_recovery") is True:
                a["n_exact_recovery"] += 1
            naive = (set(sel or []) == set(sup or []))          # the KNOWN-WRONG computation
            if naive:
                a["n_naive_recovery_bug"] += 1
            if ok:
                rec = (sel is not None and sup is not None and set(sel) == set(sup))
                if rec:
                    a["n_exact_recovery_recomputed_ok_only"] += 1
                if rec != bool(r.get("exact_recovery")):
                    a["n_recovery_field_vs_recomputed_mismatch"] += 1

            # --- coverage ---
            strict = bool(r.get("strict_simultaneous_coverage"))
            if strict:
                a["n_strict_simultaneous_coverage"] += 1
                if not (r.get("exact_recovery") is True and ok):
                    a["n_coverage_numerator_not_subset_of_recovery"] += 1
            if r.get("coverage_conditional_numerator") is True:
                a["n_coverage_conditional_numerator"] += 1

            mi = r.get("method_intervals") or {}
            if mi.get("active_joint_coverage") is True:
                a["n_active_joint_coverage_field"] += 1
            if ok:
                w = mi.get("mean_interval_width")
                if w is not None and isinstance(w, (int, float)) and math.isfinite(w):
                    a["n_width_used"] += 1
                    a["sum_width"] += float(w)
                    a["sumsq_width"] += float(w) * float(w)
                else:
                    a["n_width_missing_among_ok"] += 1
                fn = r.get("false_negatives") or []
                fp = r.get("false_positives") or []
                a["false_negative_count_reportable"] += len(fn)
                a["false_positive_count_reportable"] += len(fp)
                if fn:
                    a["n_records_with_any_false_negative"] += 1
                if fp:
                    a["n_records_with_any_false_positive"] += 1

            cth = r.get("clean_tail_boundary_hits")
            fth = r.get("final_tail_boundary_hits")
            if (isinstance(cth, int) and cth > 0) or (isinstance(fth, int) and fth > 0):
                a["n_tail_boundary_hits_positive"] += 1
    return {("%s|%s" % k): v for k, v in acc.items()}


def merge(dst, src):
    for k, v in src.items():
        d = dst[k]
        if d["meta"] is None:
            d["meta"] = v["meta"]
        for f in FIELDS_INT:
            d[f] += v[f]
        d["sum_width"] += v["sum_width"]
        d["sumsq_width"] += v["sumsq_width"]


def se(p, n):
    if n <= 0:
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "**", "task_*.jsonl"), recursive=True))
    assert len(files) == 720, len(files)
    boundary = set(json.load(open(C2))["cell_ids"])
    total = defaultdict(blank)
    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, files, chunksize=1)):
            merge(total, res)
            if (i + 1) % 60 == 0:
                print("  %d/720 files" % (i + 1), file=sys.stderr, flush=True)

    rows = []
    for key in sorted(total):
        a = total[key]
        m = a["meta"]
        n = a["n_records"]
        nrec = a["n_exact_recovery"]
        s = m["s"]
        scope = "boundary" if m["cell_id"] in boundary else "interior"
        cont = a["n_containment_candidate"]
        contsel = a["n_containment_selected"]
        strict = a["n_strict_simultaneous_coverage"]
        cf = a["n_computational_failure"]
        emp = a["n_empty_selection"]
        nw = a["n_width_used"]
        mw = (a["sum_width"] / nw) if nw else None
        if nw > 1:
            var = max(a["sumsq_width"] / nw - mw * mw, 0.0) * nw / (nw - 1)
            mw_se = math.sqrt(var / nw)
        else:
            mw_se = None
        row = dict(
            cell_id=m["cell_id"], route=m["route"], c2_scope=scope,
            c2_scope_level="cell (frozen C2_BOUNDARY_CELLS.json); records carry no "
                           "per-record interior/boundary field",
            problem=m["problem"], panel=m["panel"], N=m["N"], p=m["p"], s=s, K=m["K"],
            s_upper_bound=m["s_upper_bound"], design=m["design"], law=m["law"],
            sparsity_information=m["sparsity_information"], cell_index=m["cell_index"],
            n_records=n, n_ok=a["n_ok"],
            n_numerical_failure=a["n_numerical_failure"],
            n_selection_failure_empty=a["n_selection_failure_empty"],
            n_program_or_schema_error=a["n_program_or_schema_error"],
            n_other_status=a["n_other_status"],
            n_empty_selection=emp,
            n_interval_reportable=a["n_interval_reportable"],
            n_total_failure=a["n_total_failure"],

            n_containment=cont, containment=cont / n, containment_se=se(cont / n, n),
            n_containment_selected=contsel, containment_selected=contsel / n,
            containment_selected_se=se(contsel / n, n),
            n_containment_candidate_raw_field=a["n_containment_candidate_raw_field"],
            n_containment_selected_raw_naive=a["n_containment_selected_raw_naive"],

            n_exact_recovery=nrec, exact_recovery=nrec / n,
            exact_recovery_se=se(nrec / n, n),
            n_exact_recovery_recomputed_ok_only=a["n_exact_recovery_recomputed_ok_only"],
            n_recovery_field_vs_recomputed_mismatch=a["n_recovery_field_vs_recomputed_mismatch"],
            n_naive_recovery_bug=a["n_naive_recovery_bug"],
            naive_recovery_bug_excess=a["n_naive_recovery_bug"] - nrec,

            n_strict_simultaneous_coverage=strict,
            strict_simultaneous_coverage=strict / n,
            strict_simultaneous_coverage_se=se(strict / n, n),
            n_coverage_conditional_numerator=a["n_coverage_conditional_numerator"],
            n_coverage_numerator_not_subset_of_recovery=a["n_coverage_numerator_not_subset_of_recovery"],
            coverage_conditional_on_exact_recovery=(strict / nrec) if nrec else None,
            coverage_conditional_on_exact_recovery_se=(se(strict / nrec, nrec) if nrec else None),
            coverage_conditional_denominator=nrec,
            n_active_joint_coverage_field=a["n_active_joint_coverage_field"],

            mean_interval_width=mw, mean_interval_width_se=mw_se,
            mean_interval_width_n=nw,
            n_width_missing_among_ok=a["n_width_missing_among_ok"],

            n_computational_failure=cf, computational_failure_rate=cf / n,
            computational_failure_rate_se=se(cf / n, n),

            false_negative_count_reportable=a["false_negative_count_reportable"],
            false_negative_count_including_empty_selection=(
                a["false_negative_count_reportable"] + s * emp),
            false_negative_count_including_all_nonok=(
                a["false_negative_count_reportable"] + s * (n - a["n_ok"])),
            false_negative_denominator_reportable_records=a["n_ok"],
            false_negative_denominator_all_records=n,
            false_negative_rate_per_active_coord_reportable=(
                a["false_negative_count_reportable"] / (s * a["n_ok"])) if a["n_ok"] else None,
            false_negative_rate_per_active_coord_all=(
                (a["false_negative_count_reportable"] + s * emp) / (s * n)),
            n_records_with_any_false_negative=a["n_records_with_any_false_negative"],
            false_positive_count=a["false_positive_count_reportable"],
            n_records_with_any_false_positive=a["n_records_with_any_false_positive"],
            false_positive_rate_per_record_reportable=(
                a["false_positive_count_reportable"] / a["n_ok"]) if a["n_ok"] else None,
            n_tail_boundary_hits_positive=a["n_tail_boundary_hits_positive"],
        )
        rows.append(row)

    # ---------------- proofs ----------------
    checks = {}
    grand = sum(r["n_records"] for r in rows)
    checks["total_records"] = grand
    assert grand == 360000, "total record count %d != 360000" % grand
    checks["n_cell_route_rows"] = len(rows)
    assert len(rows) == 360, len(rows)
    bad = [r["cell_id"] + "|" + r["route"] for r in rows
           if r["n_exact_recovery"] > r["n_containment"]]
    assert not bad, "exact_recovery > containment(candidate) in: %s" % bad
    bad2 = [r["cell_id"] + "|" + r["route"] for r in rows
            if r["n_exact_recovery"] > r["n_containment_selected"]]
    assert not bad2, "exact_recovery > containment(selected) in: %s" % bad2
    bad3 = [r["cell_id"] + "|" + r["route"] for r in rows
            if r["n_strict_simultaneous_coverage"] > r["n_exact_recovery"]]
    assert not bad3, "strict coverage > exact recovery in: %s" % bad3
    bad4 = [r["cell_id"] + "|" + r["route"] for r in rows if r["n_records"] != 1000]
    assert not bad4, "cells without exactly 1000 reps: %s" % bad4
    checks["cells_with_1000_records"] = 360
    checks["mismatch_field_vs_recomputed_recovery_ok_records"] = sum(
        r["n_recovery_field_vs_recomputed_mismatch"] for r in rows)
    checks["n_ok_records_total"] = sum(r["n_ok"] for r in rows)
    checks["n_numerical_failure_total"] = sum(r["n_numerical_failure"] for r in rows)
    checks["n_selection_failure_empty_total"] = sum(r["n_selection_failure_empty"] for r in rows)
    checks["n_program_or_schema_error_total"] = sum(r["n_program_or_schema_error"] for r in rows)
    checks["n_empty_selection_total"] = sum(r["n_empty_selection"] for r in rows)
    checks["n_exact_recovery_total"] = sum(r["n_exact_recovery"] for r in rows)
    checks["n_containment_candidate_total"] = sum(r["n_containment"] for r in rows)
    checks["n_containment_selected_total"] = sum(r["n_containment_selected"] for r in rows)
    checks["n_strict_coverage_total"] = sum(r["n_strict_simultaneous_coverage"] for r in rows)
    checks["naive_recovery_bug_total"] = sum(r["n_naive_recovery_bug"] for r in rows)
    checks["coverage_numerator_not_subset_of_recovery_total"] = sum(
        r["n_coverage_numerator_not_subset_of_recovery"] for r in rows)
    checks["coverage_conditional_numerator_equals_strict_total"] = (
        sum(r["n_coverage_conditional_numerator"] for r in rows) ==
        checks["n_strict_coverage_total"])
    checks["fn_reportable_total"] = sum(r["false_negative_count_reportable"] for r in rows)
    checks["fn_including_empty_total"] = sum(
        r["false_negative_count_including_empty_selection"] for r in rows)
    checks["fp_total"] = sum(r["false_positive_count"] for r in rows)
    checks["interval_reportable_equals_status_ok"] = (
        sum(r["n_interval_reportable"] for r in rows) == checks["n_ok_records_total"])
    checks["boundary_cells_in_grid"] = len({r["cell_id"] for r in rows
                                            if r["c2_scope"] == "boundary"})
    checks["interior_cells_in_grid"] = len({r["cell_id"] for r in rows
                                            if r["c2_scope"] == "interior"})

    out = {
        "schema": "canonical-cell-table-v1",
        "source_root": ROOT,
        "gate4_freeze_id": "aac4a95d20f782ac61f7cb389b4269a76001ed6af5c6732bf078adb254eee3b9",
        "n_files": len(files),
        "conventions": {
            "route_test": "exact equality on record field 'route'; endswith('known_s') is a bug "
                          "because it also matches 'unknown_s'",
            "non_ok_records": "counted as NEITHER contained NOR recovered NOR covered",
            "authority": "producer boolean 'exact_recovery'; recomputed on every ok record",
            "status_ok_iff_interval_reportable": True,
            "containment": "n_containment / containment use candidate_contains_truth (S subset of "
                           "HTP candidate) AND status=='ok'; containment_selected uses "
                           "S subset of selected AND status=='ok'",
            "strict_simultaneous_coverage": "producer field: exact recovery AND computational "
                                            "success AND active_joint_coverage; anything else is "
                                            "noncoverage; denominator = all n_records",
            "coverage_conditional_on_exact_recovery": "n_strict_simultaneous_coverage / "
                                                      "n_exact_recovery",
            "mean_interval_width": "method_intervals.mean_interval_width averaged over ok records",
            "false_negatives_a": "false_negative_count_reportable = sum len(false_negatives) over "
                                 "ok/reportable records only",
            "false_negatives_b": "false_negative_count_including_empty_selection = (a) + s * "
                                 "n_empty_selection (each empty selection misses all s actives)",
            "c2_scope": "cell-level frozen classification from work/v26_1f/C2_BOUNDARY_CELLS.json "
                        "(23 boundary cells, r_on_candidate <= 1.75). No per-record "
                        "interior/boundary field exists in the raw records, so the unknown_s "
                        "split by scope is exactly a partition of whole cells.",
            "standard_errors": "binomial MC SE sqrt(phat(1-phat)/n) at the stated denominator; "
                               "mean_interval_width_se is sd/sqrt(n) over ok records",
        },
        "checks": checks,
        "cells": rows,
    }
    with open(os.path.join(OUTDIR, "canonical_cell_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    cols = list(rows[0].keys())
    cols.remove("c2_scope_level")
    with open(os.path.join(OUTDIR, "canonical_cell_table.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(repr(round(v, 10)))
                else:
                    vals.append(str(v))
            fh.write(",".join(vals) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
