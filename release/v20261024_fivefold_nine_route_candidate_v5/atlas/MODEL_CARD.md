# ProtoEss Atlas v20261024 release candidate

**Status: internal release candidate only; not public, not a blind validation release, and not submission-ready.**

## Matched-species atlas

The atlas contains 15,591 frozen gene-level scores across *P. falciparum* 3D7, *P. berghei* and *P. knowlesi* H. The model is the frozen five-fold nine-route LightGBM meta-model using the original 628 raw features and the original 95%-redundancy-reduced, label-stratified random train/validation/test split. No PCA or manual removal of homology30/50 was applied. The test scores were not used for the final fit or threshold selection, but historical model-profile iteration inspected Pk outcomes; report these as retrospective final evaluations, not blind estimates.

## Leave-one-species-out results

The source-only five-fold stack includes Pf, Pb, Pk and *Toxoplasma gondii*. Outcomes had been inspected during prior project development, so this is a retrospective generalization-boundary analysis rather than prospective external validation. Thresholds are selected on source validation data.

## Pk red-cell condition comparison

The package contains 3,482 one-to-one gene pairs with definite calls in the Elsworth and Oberstaller screens; 435 calls differ (12.49%; Wilson 95% CI is provided in the summary table). The continuous scores are positively associated (Spearman rho 0.7616). This is a retrospective homolog projection using PKNH features, not an A1H sequence-level prediction. The two screens also differ in parasite strain and experimental procedure, so discordance cannot be attributed to host red-cell source alone.

## Interpretation

All scores and target priorities are computational hypotheses, not experimental essentiality, binding, efficacy, safety or clinical evidence. The release candidate has no archive DOI or public-release checksum yet.
