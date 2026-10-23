# Formal manuscript reproduction

This folder contains historical five-branch and other analysis workflows, plus the versioned frozen-result package described below. The general `run_reproduction.py` launcher still invokes the older five-branch workflow; it is not the training entry point for the frozen five-fold nine-route LightGBM manuscript candidate.

## Nature Microbiology candidate v5

The current internal candidate is `../release/v20261024_fivefold_nine_route_candidate_v5/`. Its headline matched-species results use the frozen five-fold nine-route LightGBM, 628 raw features and the original 95%-redundancy-reduced label-stratified random split. Test-only row-level probabilities and labels are included so the reported AUROC/AUPRC can be recalculated without exposing train or validation predictions. Run the candidate's self-contained audit from the repository root:

```powershell
python formal_reproduction/code/verify_nature_microbiology_candidate_release_20261024.py --release-dir release/v20261024_fivefold_nine_route_candidate_v5
```

The audit verifies hashes and portability, recalculates the headline test metrics from released test-only predictions, checks selected summary statistics, and checks the embedded-figure hashes. It verifies frozen outputs; it does not retrain the nine-route model from the complete 628-feature matrices. The exact figure bitmaps can be extracted from a supplied manuscript DOCX with `extract_and_verify_manuscript_figures.py`; this checks the embedded images rather than rebuilding their plotting layers. Figs. 1, 3 and 6 have no separately verified vector originals in the current package.

To run historical full analyses, the separately archived processed input tree and application-feature cache are required. Put their locations in a user-specific copy of `config/formal_paths.example.yaml`; never point an output directory into the frozen input tree. The old launcher is not evidence that the frozen nine-route model was retrained from scratch.

Public deployment models and formal manuscript models are distinct model families. The candidate is internal only and must not be presented as a public release before author approval and completion of the stated release gates.
