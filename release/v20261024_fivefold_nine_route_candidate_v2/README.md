# Nature Microbiology release candidate v20261024_fivefold_nine_route_candidate_v2

**Internal candidate only. Not published, not submission-ready, and not an archive DOI.** Do not cite this directory as a public release until the corresponding author signs off the values/declarations and the final archive metadata is issued.

## Frozen analysis

The matched-species atlas uses the frozen five-fold nine-route LightGBM, the original 628 raw features, and the original 95%-redundancy-reduced, label-stratified random train/validation/test partitions. No PCA or manual `homology30/50` feature removal was applied. Matched-species test AUROC/AUPRC are Pf 0.821/0.873, Pb 0.874/0.896, and Pk 0.915/0.921. Historical inspection of held Pk outcomes means these are not blind estimates.

The source-only four-species LOSO table includes Pf, Pb, Pk and Tg. It is retrospective, not prospective external validation. Source-composition comparisons, Fig. 2d allocations, and the 60-run S4 label-exchange analysis are sensitivity analyses, not replacement main-test estimates.

The Pk paired-screen analysis includes 3,482 definite-call homolog pairs and 435 discordant labels (12.49%; Wilson interval in the table). The human-red-cell comparison is a retrospective homolog projection, not an A1H sequence-level or blind prediction; the source screens also differ in strain and procedure.

## Contents

- `atlas/`: standardized predictions, four-species LOSO, source-composition comparison, Pk pair-level results, S4 runs, model/data cards and upstream freeze manifest.
- `figures/main/`: the six exact PNGs embedded in the current internal manuscript, plus a provenance manifest. Fig. 1, Fig. 3 and Fig. 6 do not have separately verified vector originals.
- `source_data/`: panel-level source tables for Figs. 2-6 and the final Supplementary Fig. S1-S13 analyses. Only the named final S6 v3 and final frozen Fig. 4 source package are included; intermediate pilot/model files are excluded.
- `supplementary_workbooks/`: current internal S1-S3 workbooks, included for candidate review and still subject to author approval.
- `release_candidate_manifest.json`: status flags and SHA-256 for every payload file.

## Limitations and release gates

All gene calls and target priorities are computational hypotheses, not experimental validation. The historical v45 row-level transition data could not be verified and no transition statistic is reported. Corresponding-author approval, final declaration/reporting-summary review and archive DOI remain outstanding. The production hostname is `www.protoessatlas.com`; deploy/rollback was smoke-tested on disposable copies, while live production compatibility remains unverified. `public_release` is false.
