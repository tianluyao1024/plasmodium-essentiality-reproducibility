# Nature Microbiology release candidate v20261024_fivefold_nine_route_candidate_v6

**Public review candidate branch, not a formal archive release and not a journal submission.** The v6 Git branch is public and the candidate overlay is deployed on the ProtoEss Atlas. Do not cite it as the final version of record: author sign-off, final declarations and archive metadata are still pending.

## Frozen analysis

The matched-species atlas uses the frozen five-fold nine-route LightGBM, the original 628 raw features, and the original 95%-redundancy-reduced, label-stratified random train/validation/test partitions. No PCA or manual `homology30/50` feature removal was applied. Matched-species test AUROC/AUPRC are Pf 0.821/0.873, Pb 0.874/0.896, and Pk 0.915/0.921. Test-only row-level probabilities and labels are supplied to recompute these metrics; train and validation predictions are not included. Historical inspection of held Pk outcomes means these are not blind estimates.

Saved LightGBM meta-models and nine-route probability inputs for validation/test rows are also included. They reproduce all three frozen test probability vectors to numerical tolerance. This is a final-layer inference replay using supplied base-route probabilities; it does not refit the base routes or regenerate their training OOF predictions, and therefore is not full source-to-model retraining.

The source-only four-species LOSO table includes Pf, Pb, Pk and Tg. It is retrospective, not prospective external validation. Source-composition comparisons, Fig. 2d allocations, and the 60-run S4 label-exchange analysis are sensitivity analyses, not replacement main-test estimates.

The Pk paired-screen analysis includes 3,482 definite-call homolog pairs and 435 discordant labels (12.49%; Wilson interval in the table). The human-red-cell comparison is a retrospective homolog projection, not an A1H sequence-level or blind prediction; the source screens also differ in strain and procedure.

## Contents

- `atlas/`: standardized predictions, four-species LOSO, source-composition comparison, Pk pair-level results, S4 runs, model/data cards and upstream freeze manifest.
- `source_data/Frozen_Test_Predictions/`: test-only row-level probabilities and labels used to recompute the main AUROC/AUPRC values, plus a separate validation/test meta-model inference replay package with no validation labels.
- `figures/main/`: the six exact PNGs embedded in the current internal manuscript, plus a provenance manifest. Fig. 1, Fig. 3 and Fig. 6 do not have separately verified vector originals.
- `code/verify_nature_microbiology_candidate_release_20261024.py`: independently checks payload hashes/portability and recomputes the headline frozen metrics, LOSO species, Fig. 2d allocation count, Pk discordance interval, refit-stable group count and S4 run count. This audits frozen outputs; it does not retrain the model from the full feature matrices.
- `code/verify_frozen_meta_model_inference_20261024.py`: loads the archived species-specific LightGBM meta-models, applies them to the supplied nine-route validation/test probabilities, verifies test probabilities against the frozen row-level results, and recomputes AUROC/AUPRC. This is inference replay only; it does not retrain the base routes or the meta-model.
- `code/refresh_v6_payload_manifest.py`: refreshes byte counts and SHA-256 values for every v6 payload file except the manifest itself.
- `code/extract_and_verify_manuscript_figures.py`: extracts the exact six embedded image objects from a supplied manuscript DOCX and verifies them against the included figure manifest. This is image extraction, not recreation of vector plots; no separate vector originals are verified for Figs. 1, 3 and 6.
- `code/reproduce_main_figures_from_source.py`: redraws six main figures from the 21 packaged panel-level source tables into `reproduced_source_figures/`. The output folder includes a runtime/input/output SHA-256 manifest. This is a source-data-driven redraw, not model fitting and not a replacement for the manuscript-matched PNGs in `figures/main/`; the exact embedded images remain unchanged.
- `source_data/`: panel-level source tables for Figs. 2-6 and the final Supplementary Fig. S1-S13 analyses. Only the named final S6 v3 and final frozen Fig. 4 source package are included; intermediate pilot/model files are excluded.
- `supplementary_workbooks/`: current internal S1-S3 workbooks, included for candidate review and still subject to author approval.
- `release_candidate_manifest.json`: status flags and SHA-256 for every payload file.

## Reproduce the frozen package audit

From this directory, run `python code/verify_nature_microbiology_candidate_release_20261024.py --release-dir .` and `python code/verify_frozen_meta_model_inference_20261024.py --release-dir .`. To redraw figures from the packaged source tables, run `python code/reproduce_main_figures_from_source.py --release-dir . --output-dir reproduced_source_figures_check`; the verifier checks the committed redraws against their manifest and source-table hashes. To re-extract the exact embedded figure images, supply the internal manuscript DOCX separately: `python code/extract_and_verify_manuscript_figures.py --docx PATH_TO_MANUSCRIPT.docx --figure-manifest figures/main/figure_export_manifest.json --output-dir FIGURE_EXPORTS`.

## Limitations and release gates

All gene calls and target priorities are computational hypotheses, not experimental validation. A retrospective v45 five-branch to frozen nine-route comparison is included in Supplementary Data 3: all 15,591 unique gene records match, with 7,648 essential calls retained, 1,494 removed, 470 added and 5,979 remaining nonessential. Scores use version-specific thresholds and are not interpreted as directly comparable calibrated probabilities. This is model-version context, not validation or experimental evidence. Corresponding-author approval, final declaration/reporting-summary review and archive DOI remain outstanding. The candidate overlay is deployed at `www.protoessatlas.com`; rollback snapshots are retained. `public_release` remains false because no final archive DOI/tag has been issued.
