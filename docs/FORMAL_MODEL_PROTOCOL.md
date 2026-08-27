# Formal model and deployment model protocol

This repository is a public reproducibility and reuse package. It intentionally
separates the formal manuscript evaluation protocol from the practical release
models used for website-style prediction.

## Scope

The public files support three species-specific *Plasmodium* essentiality
models: *Plasmodium falciparum*, *Plasmodium berghei* and *Plasmodium knowlesi*.
They contain processed labelled feature matrices, class-separated CDS FASTA
records, training and prediction scripts, and frozen deployment models. They do
not contain the manuscript, genome-wide atlas, target-ranking tables, docking
outputs, figure source tables, or unpublished result workbooks.

## Formal manuscript evaluation model

The manuscript's formal predictor used a locked out-of-fold stacking design.
Every fold-local transformation was fitted on training records only before it
was applied to validation or held-out test records. The five inputs to the
formal stack were LightGBM, CatBoost, a species-aware expert, a fold-local
species-prevalence branch, and a compact Transformer feature-integration
branch. Their out-of-fold probabilities were combined by L2-regularised logistic
regression. Thresholds were selected on validation data and applied once to the
held-out test data.

The formal representation contained 582 static sequence, annotation and
protein-language-model features, 9 label-free STRING summary features, a
species indicator, 21 fold-local MMseqs2 neighbourhood features and 15
fold-local STRING/PPI neighbourhood features. Dynamic homology and interaction
features were recomputed within each formal split, so validation and test labels
could not enter feature construction.

## Transformer branch

The compact Transformer branch was the fifth probability input to the
confidential formal OOF manuscript stack. The public release distributes
practical tree-based deployment models instead; it must not be used to
recalculate held-out manuscript metrics or five-branch atlas scores.

## Public deployment models

The release models in `models/` are practical deployment artefacts trained from
the public labelled feature matrices for reproducible reuse and the online
ProtoEssAtlas resource. They are intentionally distinct from the locked
formal-evaluation models, whose metrics depend on fixed split manifests,
fold-local dynamic feature construction and OOF stacking.

## Public-use interpretation

Predictions from the supplied models are computational essentiality scores, not
experimental validation. Downstream target prioritisation, host-risk filtering
and docking are separate evidence layers and are not redistributed here.
