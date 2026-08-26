# Formal model and deployment model protocol

This repository is a public reproducibility and reuse package. It intentionally
separates the formal manuscript evaluation protocol from the practical release
models used for website-style prediction.

## Scope

The public files support three species-specific Plasmodium essentiality models:

- `Plasmodium_falciparum`
- `Plasmodium_berghei`
- `Plasmodium_knowlesi`

The repository contains processed labelled feature matrices, class-separated CDS
FASTA records, training scripts, prediction scripts and frozen release models.
It does not contain the manuscript, genome-wide prediction atlas, target-ranking
tables, docking outputs, figure source tables or unpublished result workbooks.

## Formal manuscript evaluation model

The formal model reported in the manuscript used a locked out-of-fold stacking
design. For each training fold, all fold-local transformations were fitted using
training data only and then applied to the validation or test partition.

The formal stack used four branch inputs:

- LightGBM branch
- CatBoost branch
- species-aware expert branch
- species-prior branch

The meta-learner was an L2-regularised logistic regression trained on
out-of-fold predictions. Classification thresholds were selected on the
validation partition and then applied once to the held-out test partition.

The formal 628-feature representation comprised:

- 582 static sequence, annotation and protein-language-model features
- 9 label-free STRING summary features
- 1 species indicator
- 21 fold-local MMseqs2 neighbourhood features
- 15 fold-local STRING/PPI neighbourhood features

The fold-local homology and interaction-neighbourhood features were recomputed
within each formal split so that validation and test labels could not enter
feature construction.

## Transformer branch

Transformer and other neural models were evaluated only as architecture
comparison experiments. They are not part of the frozen formal OOF stack and are
not used by the public release models in this repository.

## Public deployment models

The release models in `models/` are practical deployment artefacts trained from
the public labelled feature matrices. They are intended for reproducible reuse
and for the online ProtoEssAtlas resource.

These release models should not be used to recompute the held-out manuscript
metrics. The formal evaluation metrics depend on the locked split protocol,
fold-local feature construction, out-of-fold stacking and held-out test
evaluation described above.

## Public-use interpretation

Predictions from the supplied models are computational essentiality scores, not
experimental validation. Downstream target prioritisation, host-risk filtering
and docking analyses are separate evidence layers and are not redistributed in
this public code repository.
