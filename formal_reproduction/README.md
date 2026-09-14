# Formal manuscript reproduction

This release includes the five-branch model, LOSO, source-composition, robustness, atlas, directionality and pharmacology analysis source. The separately attached ProtoEss_formal_inputs_v1.1.0.zip provides the frozen processed input tree. Extract it and point config/formal_paths.yaml to the extracted directory; set output_root to a writable directory.

Run python run_reproduction.py --config config/formal_paths.yaml --stage all --dry-run to inspect the configured commands, then omit --dry-run to execute. Full retraining has not been rerun as part of packaging. Public deployment models and the formal manuscript stack are separate model families.
