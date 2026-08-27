# Plasmodium species-specific essentiality reproducibility resource

This repository contains only the public reproducibility inputs for three
species-specific Plasmodium essentiality models: processed labelled feature
matrices, essential/nonessential CDS FASTA files, frozen deployment models and
usage scripts. It deliberately excludes manuscripts, figures, prediction
atlases, target-prioritisation outputs, docking outputs and result tables.

## Contents

- `data/<species>/`: `essential_features.npz`, `nonessential_features.npz`,
  `all_labeled_features.npz`, and the processed label table.
- `cds/<species>/`: CDS FASTA records separated into essential and
  nonessential labelled classes.
- `models/`: one deployment ensemble and a JSON manifest for each species.
- `scripts/`: training and prediction entry points.
- `docs/FORMAL_MODEL_PROTOCOL.md`: distinction between the formal manuscript
  evaluation protocol and the public deployment models.

## Quick use

Install Python dependencies with `pip install -r requirements.txt`. To classify
an array with rows in the exact stored feature order:

```bash
python scripts/predict_with_release_model.py \
  --model models/Plasmodium_falciparum_release_model.joblib \
  --features path/to/628_column_features.npy \
  --output predictions.tsv
```

The supplied release models are refit on all labelled records using a
three-fold out-of-fold stacked LightGBM/CatBoost deployment ensemble. They are
deployment artifacts, not the held-out models used to calculate manuscript
performance. The manuscript's locked formal OOF stack used LightGBM, CatBoost,
a species-aware expert, a fold-local prevalence branch and a compact Transformer
feature-integration branch, followed by an L2 logistic meta-learner. The
practical release models deliberately do not reproduce those held-out metrics;
see `docs/FORMAL_MODEL_PROTOCOL.md` for this boundary.
Do not interpret a predicted class as experimental validation of gene
essentiality.
