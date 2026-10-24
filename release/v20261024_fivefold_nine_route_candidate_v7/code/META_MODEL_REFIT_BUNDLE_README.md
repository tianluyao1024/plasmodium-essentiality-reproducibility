# Meta-model refit bundle

This bundle refits the final species-specific nine-route LightGBM meta layer from the supplied training-only base-model out-of-fold probabilities. It recalculates thresholds from repeated training-only CV and checks the resulting 1,139 test probabilities against the frozen test table.

Run from the bundle root:

```powershell
python code/reproduce_meta_stack_from_oof_inputs_20261024.py --release-dir . --output-dir reproduced_meta_fit
```

The bundle does not retrain the five base routes, reconstruct fold-local homology/MMseqs/PPI features, or refit from the 628-dimensional matrices. Test labels are used only after fitting and threshold selection, for the final metric and frozen-probability check. Historical exposure to held-out Pk outcomes remains a limitation.
