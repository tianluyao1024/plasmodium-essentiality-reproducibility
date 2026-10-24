"""Freeze/verify inputs before repeated paired-refit experiments."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
REPAIR = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
SCOPE = REPAIR / "scope_feature_sets_v3_all_splits_20261016" / "within_species"
APP = REPAIR / "application_v3_all_splits_20261016"
FROZEN = REPAIR / "nine_route_v3_5fold_fixed_params_20261016" / "full_application" / "models"
DATA = WORKSPACE / "eukaryote_model_datasets"
STRING = WORKSPACE / "downloads" / "string_v12"
APP_MAP = Path(r"D:\eukaryote_model_runs_go_ipr_pfam\genomewide_target_application\final_application_features")
OUT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_resampling_v1\source_manifest.json")
SPECIES = {"Plasmodium_falciparum": 36329, "Plasmodium_berghei": 5823, "Plasmodium_knowlesi": 5851}
CODE_FILES = ("nine_route_refit_partitions_20261017.py", "pilot_nine_route_refit_oof_20261017.py",
              "run_nine_route_refit_oof_20261017.py", "score_nine_route_refit_application_20261017.py",
              "score_all_species_nine_route_application_20261016.py", "compare_homology_label_scopes.py",
              "train_scope_strict_oof_stack.py", "train_scope_transformer_oof.py", "train_unified_transformer_fusion.py",
              "build_strict_oof_dynamic_stack.py", "build_repaired_application_matrices.py", "feature_profiles.py",
              "protoess_safe_homology.py", "tune_oof_probability_lightgbm_meta_20261016.py",
              "final_evaluate_all_species_nine_route_20261016.py")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    paths = [CODE / name for name in CODE_FILES]
    manifests = DATA / "unified_v3_experiment_manifests_pre95human" / "within_species"
    paths.extend(sorted(manifests.glob("*/train_metadata.tsv")))
    if not paths[len(CODE_FILES):]:
        raise FileNotFoundError("No training metadata found")
    paths.extend([DATA / "mmseqs30" / "cluster30_cluster.tsv", DATA / "mmseqs50" / "cluster50_cluster.tsv",
                  DATA / "features" / "genomewide_target_application" / "homology30" / "app_cluster_assignments.tsv",
                  DATA / "genomewide_target_application" / "random_species_label_split_seed42" / "app_metadata.tsv",
                  REPAIR / "wp1_dualscreen_rebuild_20261016" / "aligned_dualscreen_gene_calls.tsv"])
    for species, tax in SPECIES.items():
        paths.extend([manifests / species / "val_metadata.tsv", SCOPE / species / "X_train.npy",
                      SCOPE / species / "ids_train.npy", SCOPE / species / "feature_names.json",
                      DATA / "scope_feature_sets_v2" / "within_species" / species / "train_vs_scope_train.tsv",
                      APP / species / "X_app.npy", APP / species / "ids_app.npy", APP / species / "feature_names.json",
                      APP / "search_cache" / f"{species}_app_vs_cluster50.tsv",
                      APP / "search_cache" / f"{species}_app_vs_train.tsv",
                      STRING / f"current_map_{tax}.tsv", STRING / f"{tax}.protein.links.v12.0.txt.gz",
                      APP_MAP / f"app_map_{tax}.tsv", FROZEN / species / "model_config.json"])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs missing: {missing}")
    return {str(path): {"sha256": digest(path), "bytes": path.stat().st_size} for path in paths}


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch", "joblib"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "python_executable": str(Path(os.sys.executable).resolve()),
            "packages": packages, "thread_environment": {key: os.environ.get(key) for key in
                ("PROTOESS_LGBM_N_JOBS", "PROTOESS_CATBOOST_THREADS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


def main():
    current = {"files": inventory(), "environment": environment()}
    if OUT.exists():
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        if saved["files"] != current["files"] or saved["environment"] != current["environment"]:
            raise RuntimeError("Refit inputs or environment changed since snapshot; preserve outputs and investigate")
        print(f"Verified {len(current['files'])} input files and runtime against {OUT}")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        result = {"status": "pre-run input snapshot for future paired resampling; not a retrospective proof for prior pilots",
                  "files": current["files"], "environment": current["environment"],
                  "limitations": ["Snapshot cannot detect a change that is reversed between checks.",
                                  "Validation metadata is hashed as a whole but only record IDs are used by fitting.",
                                  "External dependencies or system-level numerical nondeterminism may not be fully captured."]}
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Snapshotted {len(current['files'])} input files and runtime to {OUT}")


if __name__ == "__main__":
    main()

