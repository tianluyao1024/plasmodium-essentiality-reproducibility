# Recovered five-fold source snapshot

This directory preserves the exact source files used by the local five-fold branch reconstruction completed on 2026-10-24. The orchestration file was recovered from the author workspace and matched the SHA-256 recorded by the frozen reconstruction manifest. All eight imported project modules are also byte-for-byte hash matches to that source snapshot.

## Source hashes

| File | SHA-256 |
|---|---|
| `code/fully_fold_matched_meta_validation_20261016.py` | `5416cfbea33a9635b25b454b124bb69af9a26518151b8c00b6af861ccedac89d` |
| `code/compare_homology_label_scopes.py` | `62f7c15e0ccd7fa79d29097a86ebe795014a4980285071d6baf62187e21d0cf2` |
| `code/train_scope_strict_oof_stack.py` | `2a8a86cee9658ec6f4fdd6819e79c6164fdd7bbebd2d23c23f4c42aa2f928ba1` |
| `code/train_scope_transformer_oof.py` | `b3cd81799846f38593e1d4d6c3350a968873254a55e0aa6dc447088330f2c5d9` |
| `code/tune_oof_probability_lightgbm_meta_20261016.py` | `9bf50176ca32b4d19392a3f9128c46911d95aeb1fff5f197c9ee4a49eff06f1f` |
| `code/build_strict_oof_dynamic_stack.py` | `cb5cbc38b4f6e801fd4f57c9111c6a2906ab5368fc890cef169882ab34aeeb78` |
| `code/protoess_safe_homology.py` | `a92227e14e58247ec710c537786f7d63e4b79cfb2e44b3c4daa0c7a9de529e76` |
| `code/train_unified_transformer_fusion.py` | `24f47ecdf24fa62b5cbbcd4231ceebf05ea625647d2b8c545e7d606be2aea6d5` |
| `code/feature_profiles.py` | `beba85aa49f110773d7e0537f08c075b4b72c83dd2820a5c7baa525696ab4907` |

## What was reproduced

The runner rebuilt the five base branches for Pf, Pb and Pk using the original 628-column raw-scale matrices and five stratified folds (shuffle enabled, seed 42). Homology30/50, MMseqs and PPI features were reconstructed fold-locally. No test arrays, IDs, metadata or labels were read. The resulting `oof_branches.npy`, `val_branches.npy` and `val_branches_by_fold.npy` matched the frozen base-branch outputs bitwise for all three species.

The run used `--rebuild-branches-only`; it did not perform the optional profile/config search in the script. The authoritative frozen all-nine LightGBM meta-model and final metrics are maintained separately in the release manifest and frozen prediction package. An optional training-only profile-selection diagnostic is not a replacement for that frozen model.

## Execution boundary

This is a provenance-preserving source snapshot, not a self-contained data release. The original runner expects the established workspace layout and local inputs, including the repaired 628-column feature matrices, source-scope hit files, train/validation metadata, target-species PPI edges, and canonical five-fold scope OOF probabilities. Those inputs are not vendored here. Historical absolute-path defaults remain in the source; configure path environment variables before running it. The included `reproduction_comparison.json` redacts machine-specific local paths and records the exact frozen-array comparisons.

To reproduce the audited run, use the original workspace root (the parent directory containing `plasmodium-essentiality-reproducibility` and `eukaryote_model_datasets`), set the input/output environment variables to the audited data roots, set `PROTOESS_RAW_BRANCH_FOLDS=5` and `PROTOESS_RAW_BRANCH_SEED=42`, then invoke this runner with `--rebuild-branches-only`. The runner derives its code and metadata locations from the working directory, so do not launch it from an arbitrary directory. Before accepting an output, compare its IDs, source hashes and arrays against the frozen manifest; do not use the optional meta-selection path to modify the frozen all-nine model.

The 100-repeat directional stability workflow is separate and is not reproduced by this runner. Archived 100-repeat outputs remain historical audited outputs until that workflow is independently rebuilt and verified.
