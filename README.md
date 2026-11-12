# ProtoEss — Essential Gene Prediction for Malaria Parasites: Code and Data Package

This package contains the feature-extraction, dimensionality-reduction, and modeling code, together with the sequence files, essential/non-essential label tables, and train/validation/test split ID tables used for model construction. It corresponds to the frozen analysis pipeline (five-fold nine-route LightGBM, 628-dimensional features).

## Directory structure

```
code/
  feature_extraction/        Feature extraction code
    extract_local_raw_features.py        Local raw feature extraction (basic sequence features etc.)
    extract_eukaryote_plm_global.py      ESM2 / ProtT5 protein language model embeddings
    extract_all_go_ipr_pfam.py           GO / InterPro / Pfam annotation features
    prepare_go_features.py               GO vocabulary alignment and feature building
    prepare_homology30_features.py       30% homology features
    prepare_kmer_graph_features.py       k-mer and sequence-graph features
    prepare_mmseqs_neighbor_features.py  MMseqs dynamic homology-neighbor features
    prepare_string_network_features.py   STRING/PPI network features
    combine_local_features.py            Feature merging
  dimensionality_reduction/  Dimensionality reduction code
    reduce_dimensions.py                 TruncatedSVD to 128 dims (one block each for ESM2/ProtT5)
    feature_names.json                   628-dim feature name table
  modeling/                  Modeling code
    train_release_models.py              Release model training (LightGBM + CatBoost + meta-learner)
    predict_with_release_model.py        Release model inference
    run_corrected_matrix_lgbm_training_cv_20261016.py   Five-fold LightGBM training and validation
    nine_route_refit_partitions_20261017.py             Nine-route feature-branch partitioning
    run_nine_route_refit_oof_20261017.py                Five-fold paired-refit OOF
    run_paired_nine_route_repeats_20261017.py           100-repeat stability re-runs

sequences/                   Sequence files (CDS + protein FASTA)
  Pf/     train|validation|test   Plasmodium falciparum 3D7
  Pb/     train|validation|test   Plasmodium berghei ANKA
  Pk-a/   train|validation|test   Plasmodium knowlesi (monkey-RBC model)
  Pk-h/   essential|nonessential  P. knowlesi A1H1 (human-RBC screen; no three-way split)
  Tg/     train|validation|test   Toxoplasma gondii GT1

labels/                      Essential / non-essential label tables
  Plasmodium_falciparum_labels.tsv   Pf labels (with evidence and source references)
  Plasmodium_berghei_labels.tsv      Pb labels
  Plasmodium_knowlesi_labels.tsv     Pk-a labels
  Toxoplasma_gondii_labels.tsv       Tg labels
  Pk-h_A1H1_screen_labels.tsv        Pk-h screen labels (derived from the essential/non-essential folders)

splits/                      Train / validation / test split ID tables
  partition_manifest.tsv             Per-partition gene counts, essential/non-essential counts, file paths
  split_ids.tsv                      Gene-level split ID table (24,514 rows: species / partition / gene_id / label)

optional_reducers/           Fitted per-species dimensionality reducers (joblib; optional, not required)
```

## Splits and counts (consistent with partition_manifest.tsv)

| Species | train | validation | test | Note |
|---|---:|---:|---:|---|
| Pf | 4,294 | 526 | 527 | |
| Pb | 1,687 | 204 | 205 | |
| Pk-a | 3,284 | 408 | 407 | monkey-RBC model |
| Tg | 6,515 | 814 | 815 | |
| Pk-h | essential 1,891 / non-essential 2,937 | — | — | human-RBC screen, no three-way split |

24,514 gene records in total. FASTA headers preserve source identifiers (including `gene_id=` and `label=`).

## Reproduction order

1. **Feature extraction**: `extract_eukaryote_plm_global.py` (PLM embeddings) → `extract_local_raw_features.py` / `extract_all_go_ipr_pfam.py` / `prepare_*.py` (individual feature blocks) → `combine_local_features.py` (merge).
2. **Dimensionality reduction**: `reduce_dimensions.py` fits TruncatedSVD (128 dims) on the training set and projects the ESM2 and ProtT5 blocks separately.
3. **Modeling**: `train_release_models.py` (deployment models) or `run_corrected_matrix_lgbm_training_cv_20261016.py` (five-fold training).

Feature-pipeline verification: 100 sequences per dataset (50 essential + 50 non-essential) were rebuilt from scratch into the full 628-dim matrix. Maximum absolute deviation from the original matrix: 0 for Pf/Pb/Pk-a/Tg, 4.15e-5 for Pk-h (within a 1e-4 tolerance).

## Notes

- The `run_nine_route_*` and `run_paired_nine_route_*` scripts in `code/modeling/` contain original experiment-machine paths (`D:\...`); adjust the input/output paths before reuse.
- `optional_reducers/` holds the fitted per-species reducers (esm2/prott5/go_ipr_pfam/kmer_graph transform joblib files). They are only needed for strict numerical reproduction of the 628-dim matrix and can be omitted.
- Data sources: PlasmoDB, GOA/UniProt, STRING v12, and published screens (Zhang et al. 2018; Elsworth et al. 2025; Oberstaller et al. 2025, etc.). See the `source_datasets` / `source_pmids` columns of each labels.tsv.
- Schema note: the four species labels.tsv files carry evidence and provenance columns (`final_label`, `source_datasets`, ...). The Tg table follows an earlier schema (`label`, `label_source`, plus a `split` column); both encode the same essential/non-essential call.
