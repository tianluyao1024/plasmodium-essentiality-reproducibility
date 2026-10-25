# Fresh Paired-Refit Repeat 3 Audit Addendum

Status: one additional paired repeat passed end-to-end reproduction checks. This is not a completed 100-repeat stability analysis.

- Repeat index: 3 (the fourth repeat when counted from 1); fraction: 0.8; seed: 20260821.
- Species: *P. falciparum*, *P. berghei* and *P. knowlesi*.
- Paired variants: full 628-feature matrix and GO/IPR/Pfam annotation-excluded matrix.
- All six OOF arrays and all six application prediction files match the corresponding archived repeat-3 outputs byte-for-byte.
- Saved-model score replay maximum absolute error: 1.11e-16. The manifests report that validation/test labels were not used for fitting or threshold selection.
- Each species uses the same selected training records and five-fold assignments in both variants.
- Repeat-3 directional summary: 338/455 groups recovered by both variants; summary SHA-256 is recorded in the machine-readable audit.
- Partial aggregation across repeats 0-3: 4/100 completed, median recovery frequency 0.75, `final_stable_groups` null. This is an exploratory partial summary and does not revise the historical archived 184/455 count.

The machine-readable report is `fresh_repeat3_reproduction_audit_20261025.json`; its verifier is `verify_fresh_repeat3_reproduction_20261025.py`. The fresh outputs are isolated under `D:\P100`; archived source outputs were not overwritten. The frozen v7 manuscript/site candidate payload and its prior public repeat-count wording have not been changed by this interim audit.

Limitations remain: the historical profile choice included prior Pk test inspection; the repeated refits are not independent prospective validation; and the full 100-repeat refit remains incomplete.

