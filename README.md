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

## 2026-10-17 Nature Microbiology replacement release (historical)

The versioned `release/v20261017_5fold_9route_lightgbm/` directory contains the frozen five-fold nine-route LightGBM metrics, predictions, standardized genome-wide schema, four-species clade LOSO table and P. knowlesi dual-screen condition tables. The LOSO table explicitly includes P. falciparum, P. berghei, P. knowlesi and Toxoplasma gondii. The release uses the original 95%-deduplicated label-stratified random split and 628 raw features. The final freeze manifest records the historical non-blind evaluation caveat and SHA-256 hashes. The human-red-cell P. knowlesi analysis is an ortholog-mapped retrospective condition analysis, not an exact A1H sequence-level prediction.

Rebuild the package and figures with `formal_reproduction/code/build_nature_microbiology_submission_20261017.py` and `formal_reproduction/code/generate_nature_microbiology_figures_20261017.py`. The separate `plasmodium-essentiality-host-risk-atlas` project is intentionally not modified.

## 2026-11-05 Infectious Diseases of Poverty release v1.2.0

The current manuscript-facing package is archived at [Zenodo, DOI 10.5281/zenodo.23983185](https://doi.org/10.5281/zenodo.23983185) (concept DOI: [10.5281/zenodo.23983184](https://doi.org/10.5281/zenodo.23983184)). It contains the bilingual manuscripts and Supplementary Information, Figs. 1-6 and Supplementary Figs. S1-S13, source data, the completed 100 paired-refit results, and the feature/model/sequence release.

- [GitHub v1.2.0 feature extraction, model weights and sequence resources](https://github.com/tianluyao1024/plasmodium-essentiality-reproducibility/tree/idp-v1.2.0/release/IDP_v1.2.0)
- [Feature extraction, model weights and sequence bundle](https://zenodo.org/records/23983185/files/ProtoEss_IDP_feature_model_sequences_v1.2.0.zip?download=1)
- [Submission package: manuscripts, figures and supplementary materials](https://zenodo.org/records/23983185/files/IDP_submission_documents_figures_v1.2.0.zip?download=1)
- [Figure-linked source data](https://zenodo.org/records/23983185/files/IDP_source_data_v1.2.0.zip?download=1)
- [Complete submission package archive](https://zenodo.org/records/23983185/files/ProtoEss_IDP_submission_package_v1.2.0.zip?download=1)

This release uses the frozen nine-route five-fold LightGBM results and the already-completed 100 paired refits; packaging did not rerun model training. The four-species LOSO analysis includes Toxoplasma gondii. The P. knowlesi human-red-cell result is a retrospective homolog projection, not an A1-H.1 sequence-level prediction or blind validation. Historical Pk test exposure is disclosed in the manuscript. The older Nature Microbiology folder is retained for provenance and is not the current target-journal package.
