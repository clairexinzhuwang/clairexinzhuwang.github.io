# Support Recovery and Post-Recovery Simultaneous Inference for High-Dimensional Pairwise U-Statistic M-Estimators

This is a curated inspection snapshot of the numerical code and evidence files
distributed with the September 2, 2026 research package. The manuscript PDF,
manuscript source, internal review history, unfinished application work, and
third-party material are intentionally not included. Public visibility does not
grant a licence to copy, modify, or redistribute the code; see `NOTICE.md`.

## What is included

- `source/src/`: the pairwise models, hard-thresholding discovery, stochastic
  cleaning/refitting, and simultaneous-inference implementation.
- `source/formal_contract.py` and `audit_provenance.py`: the outcome contract
  and source-identity tooling.
- `source/FORMAL_GRID.json`: the prespecified 180-cell design.
- `source/C2_BOUNDARY_CELLS.json`: the disclosed boundary-cell classification.
- `evidence/formal/canonical_cell_table.{json,csv}`: the 360 aggregate rows,
  one for each design cell and route.
- `evidence/formal/build_canonical_cell_table.py`: the aggregate builder; the
  replication-level inputs it expects are not included.
- `evidence/formal/PAPER_NUMBERS.txt`: frozen headline summaries.
- `evidence/postfreeze/`: aggregate summaries for the separately labelled
  sensitivity studies.
- `PROVENANCE.json` and `SHA256SUMS`: the source archive identity and hashes of
  every file in this public snapshot.

## Evidence boundary

The supplied source archive passes its outer SHA-256 check, and the freeze
self-identity and shipped source/configuration hashes were checked. The formal
aggregate contains 360,000 outcomes: 359,935 `ok`, 16
`numerical_failure`, 49 `selection_failure_empty`, and no program/schema
failures. Every non-ok outcome remains in the denominator.

The approximately 13 GB replication-level record store is external to the
supplied archive. `FORMAL_EVIDENCE_SHA256SUMS.txt` addresses those files by
hash, but they are not redistributed here. Consequently, this snapshot supports
inspection of the exact selected source files, the formal design, the shipped
aggregate, and its provenance; it does not claim a new record-level rebuild.

## Important scope notes

- The capped unknown-s route assumes an externally supplied informative cap;
  the formal grid fixes K=2s.
- The formal study includes known-s and capped unknown-s routes. Empty
  selections and numerical failures count as strict noncoverage and failed
  recovery.
- The N=6,000, p=8,000 AUC configuration is a fixed stress tier, not an
  additional convergence point.
- Identity-design cells are theorem-anchored. Toeplitz and tridiagonal designs
  are stress panels, and contamination is applied to response noise rather
  than to the Gaussian covariates.

## Integrity

From this directory, verify the selected public files with:

    shasum -a 256 -c SHA256SUMS

The public snapshot is deliberately smaller than the source archive. Its file
hashes establish byte identity for the files included here, not identity with
the complete private package.
