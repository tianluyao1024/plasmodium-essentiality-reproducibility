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

## 2026-10-24 replacement candidate (internal; not released)

`release/v20261024_fivefold_nine_route_candidate/` is an offline candidate assembled from the current internal Nature Microbiology revision package. It carries the final source tables for Figs. 2-6 and Supplementary Figs. S1-S13, the 15,591-row standardized atlas, four-species source-only LOSO including Tg, two-malaria versus Tg-augmented source comparisons, the P. knowlesi paired-screen analysis, and the completed 60-run S4 sensitivity. The S1-S3 workbooks are included as internal candidates pending author sign-off.

This directory is **not a public release**. The matched-species metrics remain retrospective because Pk outcomes were inspected during historical model development; the Pk human-red-cell analysis is a retrospective homolog projection; historical v45 row-level transition data remain unavailable; and final author approval, declarations, archive DOI and production website target verification remain open. Do not cite this candidate directory as the final archive. Its file hashes and release gates are recorded in `release/v20261024_fivefold_nine_route_candidate/release_candidate_manifest.json`.

The candidate payload can be rebuilt from the internal package by running `formal_reproduction/code/build_nature_microbiology_candidate_release_20261024.py` with `--package-root` pointing to that package. The builder refuses to overwrite an existing candidate directory.
