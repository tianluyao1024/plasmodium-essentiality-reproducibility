# ProtoEss Atlas complete reproducibility release v1.1.0

This release brings the four-species deployment data and formal manuscript analysis code together with frozen processed inputs. It accompanies the 8,735-call malaria atlas and 46 refit-stable directional groups. Pf, Pb, Pk and Tg are represented in the reusable data; Tg is a source or held-out species in the transfer analyses.

## Files

- ProtoEss_source_v1.1.0.zip: source, four-species processed deployment data, CDS and deployment models.
- ProtoEss_formal_inputs_v1.1.0.zip: frozen feature matrices, splits, mappings, annotation and interaction caches, and application inputs.
- ProtoEss_reference_results_v1.1.0.zip: manuscript source data and the three supplementary result workbooks.
- SHA256SUMS.json: archive integrity checks.

Extract the source and input archives. In formal_reproduction/config/formal_paths.yaml set formal_input_root to the extracted input tree, application_feature_root to its application_feature_cache subdirectory, and output_root to a writable results directory. Install formal_reproduction/requirements.txt. Run the reproduction launcher with --dry-run before starting any expensive analysis. Consult the scripts for stage-specific requirements. The archive is a frozen input and code release; packaging checks are not a new numerical rerun.

The biological datasets are processed derivatives of the sources recorded in DATA_NOTICE.md and the input provenance. Their original attribution and terms continue to apply. The repository software licence does not supersede upstream data terms. The input tree preserves legacy path names where required by the analysis scripts.

The live resource is https://www.protoessatlas.com/. Older v1.0.2 archives contain the earlier three-species deployment release; they do not contain this complete formal-analysis package.

## 2026-10-17 Nature Microbiology replacement release

The versioned `release/v20261017_5fold_9route_lightgbm/` directory contains the frozen five-fold nine-route LightGBM metrics, predictions, standardized genome-wide schema, four-species clade LOSO table and P. knowlesi dual-screen condition tables. The LOSO table explicitly includes P. falciparum, P. berghei, P. knowlesi and Toxoplasma gondii. The release uses the original 95%-deduplicated label-stratified random split and 628 raw features. The final freeze manifest records the historical non-blind evaluation caveat and SHA-256 hashes. The human-red-cell P. knowlesi analysis is an ortholog-mapped retrospective condition analysis, not an exact A1H sequence-level prediction.

Rebuild the package and figures with `formal_reproduction/code/build_nature_microbiology_submission_20261017.py` and `formal_reproduction/code/generate_nature_microbiology_figures_20261017.py`. The separate `plasmodium-essentiality-host-risk-atlas` project is intentionally not modified.

## 2026-10-24 replacement candidate (v7 review candidate; not a formal archive release)

**Figure 6d erratum (2026-10-28):** Floating-point rounding selected one extra non-target P. knowlesi record at the nominal 30% and 70% ranking cutoffs. Target hits and plotted recall are unchanged; exact-decile Fisher P values are 0.008836105284403978 and 2.2679519449758756e-08, respectively, instead of 0.008930721774469522 and 2.3075280294857833e-08. The archived candidate bundle remains unchanged until the corrected sources and results are synchronized.

**Figure 5d erratum (2026-10-28):** The v7 candidate's archived Fig. 5d incorrectly displays all 184 stable groups as having OrthoMCL support. The source table records 163 supported and 21 unsupported groups; Fig. 5a reports 163/184 correctly. This is a plotting error, not a change to the archived source table. Do not interpret Fig. 5d as evidence of universal support. The figure and downstream analyses require regeneration after the fresh 100-repeat audit; this remains a review candidate, not a formal release.

The current candidate is `release/v20261024_fivefold_nine_route_candidate_v7/`, tracked by its `release_candidate_manifest.json`. It carries the frozen five-fold, nine-route LightGBM results, the original 95%-deduplicated, label-stratified random split and 628 raw features. Supplementary Data 2 was advanced to audited v12; v6 remains preserved. Four-species LOSO includes `Toxoplasma gondii`, and the P. knowlesi human-red-cell comparison remains a retrospective homolog projection, not an A1H sequence-level prediction.

The v7 review candidate is deployed in the separate **Revision candidate** tab at https://www.protoessatlas.com. It remains a public review candidate, not a formal archive release or journal submission: no candidate-specific DOI or release tag has been issued. Historical Pk outcome exposure means matched-species test estimates are retrospective rather than blind. The 100-repeat directional-stability tables are hash/value-audited saved outputs, not 100 freshly rerun fits; repeats 0-4 have been freshly reproduced, while the other 95 remain archived-only.

Deployment timing and checks are recorded in [the v7 deployment addendum](docs/REVIEW_CANDIDATE_V7_DEPLOYMENT_ADDENDUM_20261025.md), which supersedes the build-time deployment wording in the v7 directory README without changing the frozen candidate bundle.

All-author approval of the manuscript and public release is recorded. The official Nature Portfolio Reporting Summary and remaining journal declarations still require completion and verification before formal submission or archival publication. Do not cite this review candidate as a DOI-backed final archive.

## Paired-refit provenance

The recovered source dependency closure and saved-output audit for the historical 100-repeat directional-stability analysis are archived at `formal_reproduction/recovered_paired_refit_source_20261024/`. The audit verified 100/100 saved repeat summaries and did not read validation or test labels. Fresh end-to-end reproductions currently cover repeats 0-4; the remaining 95 of 100 archived fits have not been freshly rerun. Repeats 1, 2, 3 and 4 each have a machine-readable audit and scope addendum in `formal_reproduction/`. The partial 5/100 aggregation has median recovery frequency 0.8 and is exploratory only, not a final stability result; the historical 184/455 groups remain an integrity-audited archived result. The public candidate ZIP and frozen test metrics remain unchanged. The source README and manifest document the original workspace/data dependencies and this distinction.
