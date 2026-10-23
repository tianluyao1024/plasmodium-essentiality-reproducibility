# Nature Microbiology release candidate v20261024_fivefold_nine_route_candidate_v6

**Internal candidate only. Not published, not submission-ready, and not an archive DOI.** Do not cite this directory as a public release until the corresponding author signs off the values/declarations and the final archive metadata is issued.

## Frozen analysis

The matched-species atlas uses the frozen five-fold nine-route LightGBM, the original 628 raw features, and the original 95%-redundancy-reduced, label-stratified random train/validation/test partitions. No PCA or manual `homology30/50` feature removal was applied. Matched-species test AUROC/AUPRC are Pf 0.821/0.873, Pb 0.874/0.896, and Pk 0.915/0.921. Test-only row-level probabilities and labels are supplied to recompute these metrics; train and validation predictions are not included. Historical inspection of held Pk outcomes means these are not blind estimates.

The source-only four-species LOSO table includes Pf, Pb, Pk and Tg. It is retrospective, not prospective external validation. Source-composition comparisons, Fig. 2d allocations, and the 60-run S4 label-exchange analysis are sensitivity analyses, not replacement main-test estimates.

The Pk paired-screen analysis includes 3,482 definite-call homolog pairs and 435 discordant labels (12.49%; Wilson interval in the table). The human-red-cell comparison is a retrospective homolog projection, not an A1H sequence-level or blind prediction; the source screens also differ in strain and procedure.

## Contents

- `atlas/`: standardized predictions, four-species LOSO, source-composition comparison, Pk pair-level results, S4 runs, model/data cards and upstream freeze manifest.
- `source_data/Frozen_Test_Predictions/`: test-only row-level probabilities and labels used to recompute the main AUROC/AUPRC values; no train/validation predictions are included.
- `figures/main/`: the six exact PNGs embedded in the current internal manuscript, plus a provenance manifest. Fig. 1, Fig. 3 and Fig. 6 do not have separately verified vector originals.
- `code/verify_nature_microbiology_candidate_release_20261024.py`: independently checks payload hashes/portability and recomputes the headline frozen metrics, LOSO species, Fig. 2d allocation count, Pk discordance interval, refit-stable group count and S4 run count. This audits frozen outputs; it does not retrain the model from the full feature matrices.
- `code/extract_and_verify_manuscript_figures.py`: extracts the exact six embedded image objects from a supplied manuscript DOCX and verifies them against the included figure manifest. This is image extraction, not recreation of vector plots; no separate vector originals are verified for Figs. 1, 3 and 6.
- `source_data/`: panel-level source tables for Figs. 2-6 and the final Supplementary Fig. S1-S13 analyses. Only the named final S6 v3 and final frozen Fig. 4 source package are included; intermediate pilot/model files are excluded.
- `supplementary_workbooks/`: current internal S1-S3 workbooks, included for candidate review and still subject to author approval.
- `release_candidate_manifest.json`: status flags and SHA-256 for every payload file.

## Reproduce the frozen package audit

From this directory, run `python code/verify_nature_microbiology_candidate_release_20261024.py --release-dir .`. To re-extract the exact embedded figure images, supply the internal manuscript DOCX separately: `python code/extract_and_verify_manuscript_figures.py --docx PATH_TO_MANUSCRIPT.docx --figure-manifest figures/main/figure_export_manifest.json --output-dir FIGURE_EXPORTS`.

## Limitations and release gates

All gene calls and target priorities are computational hypotheses, not experimental validation. A retrospective v45 five-branch to frozen nine-route comparison is included in Supplementary Data 3: all 15,591 unique gene records match, with 7,648 essential calls retained, 1,494 removed, 470 added and 5,979 remaining nonessential. Scores use version-specific thresholds and are not interpreted as directly comparable calibrated probabilities. This is model-version context, not validation or experimental evidence. Corresponding-author approval, final declaration/reporting-summary review and archive DOI remain outstanding. The production hostname is `www.protoessatlas.com`; deploy/rollback was smoke-tested on disposable copies, while live production compatibility remains unverified. `public_release` is false.
