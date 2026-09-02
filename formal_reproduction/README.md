# Formal five-branch manuscript analysis source

This directory contains the source code used for the locked five-branch out-of-fold (OOF) analyses reported in the accompanying manuscript: LightGBM, CatBoost, species-aware expert, fold-local prevalence and a compact Transformer branch, followed by an L2 logistic meta-learner. It includes the matched-species, four-species LOSO, Tg source-composition, family-blocked homology baseline, label-uncertainty, orthology, directionality and pharmacology-concordance entry points.

The code is public. The exact fixed formal input bundle, including confidential reviewer manifests and all intermediate matrices, is supplied to editors and reviewers during peer review and will be deposited upon publication. The public `data/`, `cds/` and `models/` directories are the reusable four-species deployment materials; their tree-based deployment models are intentionally distinct from the locked formal manuscript stack.

Run `python run_reproduction.py --help` after installing `requirements.txt`, then copy `config/formal_paths.example.yaml` to `config/formal_paths.yaml` and set the input/output roots.
