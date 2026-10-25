# Fresh Paired-Refit Repeat 4 Audit Addendum

This addendum records one newly reproduced paired 80%-training-subset, five-fold refit repeat for the full 628-feature and GO/IPR/Pfam-masked model variants. It extends the local reproduction evidence; it is not a final stability result or independent biological validation.

## Verification

- Pf, Pb and Pk were each refit under both variants using the same selected training records and the same five-fold assignment within each paired comparison.
- All six OOF probability arrays and all six application prediction tables match the archived repeat-4 outputs byte-for-byte.
- Saved-model replay maximum absolute probability error was `1.11e-16` for every species/variant pair.
- The manifests report that validation and test labels were not used for fitting or threshold selection. Historical model-route selection had previously inspected Pk test results, so this does not restore a blind test.
- The paired directional summary recovered 326/455 reference patterns in both variants and matches the archived repeat-4 summary byte-for-byte.
- The verifier passed with zero errors. See `fresh_repeat4_reproduction_audit_20261025.json` and `verify_fresh_repeat4_reproduction_20261025.py`.

## Scope

Repeats 0-4 have now been freshly reproduced end to end. The remaining 95 of the 100 archived repeats have not been freshly refit in this revision. The integrity-audited historical 100-repeat outputs still report 184/455 groups at the prespecified 80/100 rule; that historical number is not replaced by the partial five-repeat aggregation. The partial aggregate has median recovery frequency `0.8` across 455 groups and `final_stable_groups=null`.

The five-repeat aggregate is exploratory only. Refit application calls include training records and are descriptive, not an independent performance estimate. Annotation masking removes GO/IPR/Pfam input columns but does not remove correlated sequence or other feature information. The recovered workspace does not bundle all original 628-feature matrices and metadata needed for a portable full rerun.

