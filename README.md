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
