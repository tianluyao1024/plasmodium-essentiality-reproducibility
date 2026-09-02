# ProtoEss Atlas reproducibility resource

Public code, processed labelled inputs and frozen deployment models for species-aware apicomplexan essentiality scoring.

**Website:** [ProtoEss Atlas](https://www.protoessatlas.com)

**Archive:** [Zenodo concept record](https://doi.org/10.5281/zenodo.21915268)
**License:** MIT for code; see [DATA_NOTICE.md](DATA_NOTICE.md) for source-data terms.

## Scope

The release contains reusable materials for four source-labelled species:

| Species | Public role | Deployment artefact |
|---|---|---|
| *Plasmodium falciparum* 3D7 | matched-species malaria scoring | `models/Plasmodium_falciparum_release_model.joblib` |
| *Plasmodium berghei* ANKA | matched-species malaria scoring | `models/Plasmodium_berghei_release_model.joblib` |
| *Plasmodium knowlesi* H | matched-species malaria scoring | `models/Plasmodium_knowlesi_release_model.joblib` |
| *Toxoplasma gondii* GT1 | distant-apicomplexan source and held-out transfer reproducibility | `models/Toxoplasma_gondii_release_model.joblib` |

Each deployment artefact is a practical LightGBM/CatBoost three-fold OOF ensemble refit on its public processed labelled matrix. It returns a computational essentiality score and a thresholded class call. It is not an experimental assertion of essentiality or druggability. The formal manuscript evaluation used a separate five-branch OOF stack; its complete source code is included in [`formal_reproduction/`](formal_reproduction/).

## Included materials

```text
data/<species>/
  all_labeled_features.npz       processed labelled matrix and split labels
  essential_features.npz         class-specific matrix
  nonessential_features.npz      class-specific matrix
  labels.tsv                     processed identifiers, label provenance and split

cds/<species>/
  essential.cds.fasta            essential-labelled CDS records
  nonessential.cds.fasta         nonessential-labelled CDS records

models/
  *_release_model.joblib         frozen practical deployment ensemble
  *_release_model.json           model manifest, threshold and feature order

formal_reproduction/
  code/                           five-branch OOF, LOSO and robustness source
  config/                         formal path template
  run_reproduction.py             formal stage launcher

scripts/
  train_release_models.py        rebuild a practical deployment model
  predict_with_release_model.py  score a prepared numeric matrix
```

This repository deliberately excludes the manuscript, genome-scale atlas calls, target-prioritisation records, docking outputs, figure source data and unpublished result tables.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/tianluyao1024/plasmodium-essentiality-reproducibility.git
cd plasmodium-essentiality-reproducibility
python -m pip install -r requirements.txt
```

## Score a prepared matrix

```bash
python scripts/predict_with_release_model.py \
  --model models/Plasmodium_falciparum_release_model.joblib \
  --features example_features.npy \
  --output predictions.tsv
```

The numeric input matrix must preserve the exact `feature_names` order in the matching JSON manifest. The current release models expect 628 columns: static sequence, ESM2, ProtT5, annotation and label-free network features plus frozen-reference homology/PPI summaries. The ProtoEss Atlas website provides CDS- and protein-FASTA workflows for supported feature-availability profiles.

## Rebuild a public deployment model

```bash
python scripts/train_release_models.py \
  --data data/Toxoplasma_gondii/all_labeled_features.npz \
  --species Toxoplasma_gondii \
  --output models/Toxoplasma_gondii_retrained.joblib
```

A retrained public deployment model is a reuse/sensitivity artefact. It will not reproduce locked manuscript metrics without the fixed formal split manifests and fold-local dynamic features. See [docs/FORMAL_MODEL_PROTOCOL.md](docs/FORMAL_MODEL_PROTOCOL.md).

## Citation and responsible use

Cite this versioned release and the original genetic-screen sources in [DATA_NOTICE.md](DATA_NOTICE.md). Essentiality depends on species, strain, life stage, assay and phenotype threshold; all scores are computational hypotheses for comparative follow-up.
