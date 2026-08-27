# Plasmodium Essentiality Reproducibility Resource

Public inputs, frozen deployment models and reuse scripts for species-matched
*Plasmodium* essentiality scoring.

**Website:** [ProtoEss Atlas](https://www.protoessatlas.com)

**Archived release:** [Zenodo record](https://doi.org/10.5281/zenodo.21915268)
**License:** MIT for the code; see [DATA_NOTICE.md](DATA_NOTICE.md) for source-data terms.

## Purpose and scope

This repository supports transparent reuse of the three practical deployment
models used by ProtoEss Atlas:

| Species | Release model | Processed training inputs |
|---|---|---|
| *Plasmodium falciparum* 3D7 | `models/Plasmodium_falciparum_release_model.joblib` | `data/Plasmodium_falciparum/` |
| *Plasmodium berghei* ANKA | `models/Plasmodium_berghei_release_model.joblib` | `data/Plasmodium_berghei/` |
| *Plasmodium knowlesi* H | `models/Plasmodium_knowlesi_release_model.joblib` | `data/Plasmodium_knowlesi/` |

Each deployment model is a practical LightGBM/CatBoost out-of-fold ensemble
refit on the corresponding public labelled matrix. It returns a computational
essentiality score and a thresholded class call. It is not experimental
validation of gene essentiality or druggability.

## What is included

```text
data/<species>/
  all_labeled_features.npz       complete processed labelled matrix
  essential_features.npz         class-specific processed matrix
  nonessential_features.npz      class-specific processed matrix
  labels.tsv                     processed record identifiers and labels

cds/<species>/
  essential.cds.fasta            labelled essential CDS sequences
  nonessential.cds.fasta         labelled non-essential CDS sequences

models/
  *_release_model.joblib         frozen deployment ensemble
  *_release_model.json           feature order, threshold and model manifest

scripts/
  train_release_models.py        rebuild a deployment ensemble from a public matrix
  predict_with_release_model.py  score a prepared feature matrix

docs/
  FORMAL_MODEL_PROTOCOL.md       formal manuscript versus deployment-model boundary
```

The repository deliberately does **not** contain the manuscript, genome-scale
prediction atlas, target-prioritisation records, docking outputs, figure source
data or unpublished result tables.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/tianluyao1024/plasmodium-essentiality-reproducibility.git
cd plasmodium-essentiality-reproducibility
python -m pip install -r requirements.txt
```

## Quick start: score a prepared matrix

The local command-line scorer accepts a numeric NumPy `.npy` matrix. Rows
correspond to genes or transcripts; columns must exactly follow the
`feature_names` array in the matching model JSON file.

```bash
python scripts/predict_with_release_model.py \
  --model models/Plasmodium_falciparum_release_model.joblib \
  --features example_features.npy \
  --output predictions.tsv
```

The output is a tab-separated table with:

| Column | Meaning |
|---|---|
| `row_index` | Input row index |
| `essentiality_score` | Deployment-ensemble probability-like score |
| `threshold` | Model-specific out-of-fold MCC threshold |
| `predicted_essential` | Score at or above the frozen threshold |

### Input representation

Each release model expects **628 columns** in its frozen order:

- 128 ESM2 protein-language-model dimensions;
- 128 ProtT5 dimensions;
- 31 sequence-composition and physicochemical descriptors;
- 128 encoded annotation dimensions;
- 128 k-mer/graph sequence dimensions;
- 15 MMseqs30 and 15 MMseqs50 static homology summaries;
- 18 label-free STRING descriptors;
- one species indicator, 21 dynamic MMseqs2-neighbourhood features and 15
  dynamic STRING/PPI-neighbourhood features.

The exact machine-readable column order is stored in
`models/<species>_release_model.json`. Do not reorder, subset or mix feature
schemas between species. For CDS FASTA-based prediction and optional feature
construction, use [ProtoEss Atlas](https://www.protoessatlas.com), which guides
the required inputs and records missing optional layers.

## Rebuild a public deployment model

The deployment ensemble can be retrained from the public processed matrix:

```bash
python scripts/train_release_models.py \
  --data data/Plasmodium_falciparum/all_labeled_features.npz \
  --species Plasmodium_falciparum \
  --output models/Plasmodium_falciparum_retrained.joblib
```

The script uses three stratified folds to obtain LightGBM and CatBoost
out-of-fold probabilities, then fits an L2-regularised logistic meta-learner.
Its threshold is selected by out-of-fold MCC maximisation. Retraining is useful
for reuse and sensitivity analyses; it will not reproduce the locked manuscript
held-out metrics unless the complete formal protocol and split manifests are used.

## Formal manuscript model versus deployment models

The manuscript's central evaluation used a separate locked **five-branch OOF
stack**: LightGBM, CatBoost, species-aware expert, fold-local prevalence,
compact Transformer and L2 logistic meta-learning. Dynamic features, threshold
selection and preprocessing were restricted to fitting partitions. The practical
models in this repository are deliberately simpler public deployment artefacts.

The submission's editor/reviewer source archive contains the canonical formal
entry point, frozen manifests and fold-local feature-construction code. It
recreates the five-column OOF matrix with random seed 42 and Transformer batch
size 64; the four-branch tree-only model is included solely as an
ablation/control. The formal archive excludes unpublished atlas, target and
docking outputs. See [docs/FORMAL_MODEL_PROTOCOL.md](docs/FORMAL_MODEL_PROTOCOL.md)
for the full boundary.

## Data provenance and responsible use

The public matrices and class-separated CDS records were derived from public
VEuPathDB/PlasmoDB genome and annotation resources plus published genetic-screen
data. Original providers retain their respective licences and attribution
requirements. See [DATA_NOTICE.md](DATA_NOTICE.md).

Essentiality depends on species, strain, life stage, assay design and threshold.
Scores should therefore be interpreted as ranked computational hypotheses for
comparative follow-up, not as clinical safety, host selectivity or direct drug
target evidence.

## Citation

Please cite the Zenodo release and the original data sources listed in
[DATA_NOTICE.md](DATA_NOTICE.md). Citation metadata are available in
[CITATION.cff](CITATION.cff).
