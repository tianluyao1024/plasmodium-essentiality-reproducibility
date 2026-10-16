# ProtoEss Nature Microbiology release bundle

Release: `v20261017_5fold_9route_lightgbm`

This bundle is generated from the frozen five-fold nine-route LightGBM artifacts. The main model uses the original 628 raw features after 95% redundancy reduction and the original stratified random train/validation/test split. No PCA or manual removal of `homology30/50` is applied.

The four-species clade leave-one-species-out analysis includes **Plasmodium falciparum, P. berghei, P. knowlesi and Toxoplasma gondii**. It is a companion cross-species generalization analysis and is not the same evaluation as the main within-species five-fold test.

The P. knowlesi human-red-cell analysis is reported as an ortholog-mapped retrospective external-condition analysis. A complete A1H sequence-level feature matrix was not available in the audited input directories; therefore the bundle does not claim exact A1H sequence-level prediction.

The authoritative hash list is `manifest/final_freeze_manifest.json`.
