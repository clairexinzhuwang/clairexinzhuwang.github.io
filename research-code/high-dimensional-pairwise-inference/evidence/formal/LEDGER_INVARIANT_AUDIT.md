# Formal ledger invariant audit (Pro item 3)
Run 2026-09-02 against `canonical_cell_table.json` (360 cell-route rows, rebuilt from the 720
sealed jsonl files). Result: **11/11 PASS, 0 failures.**

| # | Invariant | Result |
|---|---|---|
| 1 | all 360,000 route records accounted | 360000 |
| 2 | exact recovery => candidate containment | 0 violations |
| 3 | exact recovery => selected containment | 0 violations |
| 4 | strict coverage => exact recovery | 0 violations |
| 5 | coverage numerator is a subset of the recovery numerator | 0 |
| 6 | producer `exact_recovery` == recomputed set equality (ok rows) | 0 mismatches |
| 7 | FN/FP record counts compatible with exact recovery | 0 cells out of range |
| 8 | status partition sums to 360,000 | ok 359935 + numfail 16 + empty 49 + err 0 |
| 9 | non-ok total = 65 | 65 |
| 10 | empty selections occur only on the capped unknown-s route | known_s empty = 0 |
| 11 | known-s exact recovery = 179,992 / 180,000 | 179992 |

Non-ok records by route: known_s = 8 numerical failures, 0 empty selections;
capped unknown_s = 8 numerical failures, 49 empty selections. All 65 live in the AUC (500,1000)
configuration. The manuscript states 179,992/180,000 and carries no stale "180,000/180,000".
