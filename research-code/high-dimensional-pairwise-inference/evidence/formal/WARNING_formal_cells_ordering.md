# WARNING — formal_cells.json row order is NOT the formal cell index

**Status: known hazard, one confirmed near-miss. Read before consuming `formal_cells.json`.**

## The hazard

`formal_cells.json` (produced by `analyze_formal_grid.py`) is row-ordered by the Python sort
key `(problem, N, p, s, design, law)` — auc before rank, N ascending, designs alphabetical,
laws 1–6. The sealed index source `work/v26_1f/FORMAL_GRID.json` orders its 180 `cells` in a
**different** enumeration order and is the only place `cell_index` and `formal_seed_base` are
defined. `formal_cells.json` carries **no** `cell_index` field, so positional indexing into it
silently yields the wrong cell — and therefore the wrong seeds, since every formal seed is
derived as

```
seed = 10000000 + cell_index * 10000 + rep        (rep = 0 … 999)
```

## The confirmed near-miss

One prior consumer looked up `auc_N500_p1000_s3_tridiag_law6` by its position in
`formal_cells.json` and obtained **17** instead of the correct cell_index **125**
(seed base 11 250 000). The collision is maximally misleading: FORMAL_GRID's cell_index 17 is a
*real* cell — `rank_N500_p1000_s3_tridiag_law6`, seed base 10 170 000 — so seeds computed from
the positional value re-derive a different cell's replication stream and nothing crashes. Every
downstream quantity is then computed on data from the wrong cell.

## The rule

**`work/v26_1f/FORMAL_GRID.json` is the sole valid source of `cell_index` and
`formal_seed_base`.** Look cells up by exact `cell_id`
(`{problem}_N{N}_p{p}_s{s}_{design}_law{lawint}`), never by row position in any derived file.
The law-label-to-integer mapping is the `LAW` dict in `analyze_formal_grid.py`
(1 `N(0,1)`, 2 `t4/sqrt2`, 3 `Gamma(4,1)`, 4 `t3`, 5 `Cauchy`, 6 `5%N(0,100^2)`), consistent
with the sealed `src/dgps.py` law semantics.

## The safe replacement

Use `independent_audit_v26_1f/formal_cells_indexed.json` instead of `formal_cells.json`. It is
the same 180 rows (original field values unchanged, original row order preserved), with every
row joined **by exact cell name** to FORMAL_GRID and prefixed with:

- `cell_id` — the canonical cell name;
- `cell_index` — FORMAL_GRID's index (the seed index);
- `formal_seed_base` — FORMAL_GRID's seed base (= 10000000 + cell_index·10000, verified for
  all 180 rows);
- `seed_epoch` — `formal-base10000000-stride10000-v1`;
- `law_int` — the integer law code matching the sealed sources;
- `formal_cells_positional_index` — the row's position in `formal_cells.json`, retained only
  for traceability. **It is not a seed index. Never derive seeds from it.**

Join verification (2026-09-01): 180/180 rows matched exactly one FORMAL_GRID cell, no
duplicates, all structural fields (problem, N, p, s, design, law) agree, and the seed-base
formula holds on every row. Spot checks: `auc_N500_p1000_s3_tridiag_law6` → cell_index 125 /
seed base 11250000; `rank_N500_p1000_s3_tridiag_law6` → cell_index 17 / seed base 10170000.

Cross-reference: defect entry **B6** in `TOOLING_DEFECTS.md`; the seed formula and freeze
identity are bound under Gate-4 freeze `aac4a95d…` (sealed tree unmodified by this fix).
