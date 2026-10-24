"""Score application proteins after a checked five-fold nine-route refit.

The current output is provisional. It does not compute directional stability.
No validation or test labels are read by this runner. Pk Oberstaller training
calls are allowed only in the frozen alternate training-label scope.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nine_route_refit_partitions_20261017 import SCOPES, build_partition, make_scope_pools
from run_nine_route_refit_oof_20261017 import BRANCHES, SPECIES, digest


WORKSPACE = Path(__file__).resolve().parents[3]
DATA = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
FEATURES = DATA / "scope_feature_sets_v3_all_splits_20261016"
APP = DATA / "application_v3_all_splits_20261016"
BASELINE = DATA / "nine_route_v3_5fold_fixed_params_20261016"
OUT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")

os.environ["PROTOESS_NINE_FEATURE_ROOT"] = str(FEATURES)
os.environ["PROTOESS_NINE_APP_ROOT"] = str(APP)
os.environ["PROTOESS_NINE_OOF_ROOT"] = str(BASELINE)
import score_all_species_nine_route_application_20261016 as app  # noqa: E402
import tune_oof_probability_lightgbm_meta_20261016 as meta  # noqa: E402
from final_evaluate_all_species_nine_route_20261016 import threshold_from_training_cv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=SPECIES, required=True)
    parser.add_argument("--fraction", type=float, default=.8)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--annotation-excluded", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != WORKSPACE.resolve():
        raise RuntimeError(f"Run from workspace directory {WORKSPACE}")
    if args.repeat < 0:
        raise ValueError("Repeat must be nonnegative")
    species = args.species
    seed = args.seed + args.repeat * 1009 + SPECIES.index(species)
    variant = "annotation_excluded" if args.annotation_excluded else "full"
    base = OUT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}" / species / variant
    oof_dir = base / "frozen_unlabelled"
    destination = base / "application"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing application artifacts: {destination}")
    oof_manifest = json.loads((oof_dir / "oof_manifest.json").read_text(encoding="utf-8"))
    if oof_manifest["oof_sha256"] != digest(oof_dir / "oof_probabilities.npy"):
        raise ValueError("OOF input hash mismatch")
    if oof_manifest["selected_ids_sha256"] != digest(oof_dir / "selected_ids.npy"):
        raise ValueError("OOF IDs hash mismatch")
    if oof_manifest["species"] != species or oof_manifest["variant"] != variant or oof_manifest["partition_seed"] != seed:
        raise ValueError("OOF manifest and requested settings differ")

    train_path = app.MANIFEST_ROOT / species / "train_metadata.tsv"
    train_all = pd.read_csv(train_path, sep="\t", dtype={"record_id": str})
    all_ids = train_all.record_id.to_numpy(dtype=str)
    all_y = train_all.label.eq("essential").to_numpy(dtype=np.int8)
    part = build_partition(all_y, args.fraction, seed)
    train = train_all.iloc[part.selected_indices].copy().reset_index(drop=True)
    ids = train.record_id.to_numpy(dtype=str)
    y = train.label.eq("essential").to_numpy(dtype=np.int8)
    if not np.array_equal(ids, np.load(oof_dir / "selected_ids.npy", allow_pickle=False)):
        raise ValueError("OOF IDs do not match sampled training data")
    if digest(train_path) != oof_manifest["training_labels_source_sha256"]:
        raise ValueError("Training metadata changed after OOF fitting")
    matrix_dir = FEATURES / "within_species" / species
    matrix_ids = np.load(matrix_dir / "ids_train.npy", allow_pickle=True).astype(str)
    if not np.array_equal(all_ids, matrix_ids):
        raise ValueError("Training matrix and manifest IDs differ")
    names = json.loads((matrix_dir / "feature_names.json").read_text(encoding="utf-8"))
    X = np.asarray(np.load(matrix_dir / "X_train.npy", mmap_mode="r")[part.selected_indices], dtype=np.float32).copy()
    if X.shape != (len(ids), 628) or len(names) != 628 or not np.isfinite(X).all():
        raise ValueError("Invalid original feature matrix")
    annotation = np.flatnonzero(np.char.startswith(np.asarray(names, dtype=str), "go_ipr_pfam_"))
    if not np.array_equal(annotation, np.arange(287, 415)):
        raise ValueError("Functional-annotation block differs from frozen schema")
    app_meta = pd.read_csv(app.APP_METADATA, sep="\t", dtype={"record_id": str, "species": str}).fillna("")
    query, app_ids, X_app, app_dir = app.load_application(species, names, app_meta)
    overrides = app.app_builder.load_app_cluster_assignments(app_ids, species, APP / "search_cache")

    # Reconstruct all label-dependent application inputs from this fit only.
    hom_ix, hom_local, hom_label_ix, mm_ix, ppi_ix = app.dynamic_columns(names)
    original_dynamic = {"homology": X_app[:, hom_label_ix].copy(), "mmseqs": X_app[:, mm_ix].copy(),
                        "ppi": X_app[:, ppi_ix].copy()}
    query_frame = pd.DataFrame({"record_id": app_ids, "species": species})
    raw_train = app.scopes.raw_homology_features(train[["record_id", "species"]], train)
    raw_app = app.scopes.raw_homology_features(query_frame, train, overrides)
    mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw_train.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
    X_app[:, hom_label_ix] = ((raw_app - mean) / std).astype(np.float32)[:, hom_local]
    labels = dict(zip(ids, y.astype(float)))
    target_species = dict.fromkeys(ids, species)
    query_species = dict.fromkeys(app_ids, species)
    hits = app.read_application_hits(species, app_ids)
    X_app[:, mm_ix] = app.scopes.aggregate_mm(hits, app_ids, labels, target_species, query_species)
    tax = app.app_builder.SPECIES_TAX[species][1]
    app_map = app.app_builder.map_rows(app.app_builder.STRING_APP_RUN / f"app_map_{tax}.tsv")
    X_app[:, ppi_ix] = app.app_builder.app_ppi_dynamic(app_ids, app_map, ids, y, species)
    reconstruction_error = {"homology": float(np.max(np.abs(X_app[:, hom_label_ix] - original_dynamic["homology"]))),
                            "mmseqs": float(np.max(np.abs(X_app[:, mm_ix] - original_dynamic["mmseqs"]))),
                            "ppi": float(np.max(np.abs(X_app[:, ppi_ix] - original_dynamic["ppi"])))}
    if args.fraction == 1.0 and max(reconstruction_error.values()) > 2e-5:
        raise ValueError(f"Full-training application features differ from the frozen matrix: {reconstruction_error}")
    if args.annotation_excluded:
        X[:, annotation] = 0
        X_app[:, annotation] = 0
    if not np.isfinite(X_app).all():
        raise ValueError("Non-finite application features after reconstruction")

    all_train = app.scopes.read_all_training_metadata()
    all_ober, ober_stats = app.scopes.build_oberstaller_training_pool(all_train)
    pools = make_scope_pools(species, train, all_train, all_ober)
    selected_set = set(ids)
    for pool in pools.values():
        if not set(pool.loc[pool.species.eq(species), "record_id"]).issubset(selected_set):
            raise ValueError("Target label pool includes non-fit records")
    frames, split_ids, arrays = {"train": train}, {"train": ids}, {"train": X}
    model_dir = destination / "models"
    print(f"Fitting {species} {variant} application scope models", flush=True)
    scope_probs = app.predict_scope_routes(species, frames, split_ids, y, arrays, X_app, app_ids, names,
                                           pools[SCOPES[2]], pools[SCOPES[3]], overrides, model_dir / "base_routes")
    print(f"Fitting {species} {variant} application branch models", flush=True)
    branch_probs, best_epoch = app.predict_five_branches(species, frames, split_ids, y, arrays, X_app, app_ids,
                                                        names, model_dir / "base_routes" / "five_branches")
    base_probs = np.column_stack([scope_probs, branch_probs]).astype(np.float32)
    oof = np.load(oof_dir / "oof_probabilities.npy", allow_pickle=False)
    if base_probs.shape != (len(app_ids), 9) or oof.shape != (len(ids), 9):
        raise ValueError("Unexpected nine-route OOF or application dimensions")
    if not np.isfinite(base_probs).all() or np.any((base_probs < 0) | (base_probs > 1)):
        raise ValueError("Invalid application branch probabilities")
    config_path = BASELINE / "full_application" / "models" / species / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["input_order"] != list(SCOPES) + list(BRANCHES):
        raise ValueError("Frozen candidate input order mismatch")
    candidate = config["candidate"]
    repeated_oof, _, _ = meta.cv_predictions(oof, y, candidate, meta.CV, 5, 3)
    threshold = threshold_from_training_cv(y, repeated_oof)
    model = meta.make_model(candidate, 42).fit(oof, y)
    score = model.predict_proba(base_probs)[:, 1]
    if not np.isfinite(score).all():
        raise ValueError("Non-finite application predictions")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "nine_route_meta_lightgbm.joblib")
    output = query[["record_id", "species", "gene_id", "protein_id", "protein_length"]].copy()
    for i, name in enumerate(list(SCOPES) + list(BRANCHES)):
        output[f"route_{name}_probability"] = base_probs[:, i]
    output["score"] = score
    output["threshold_from_training_cv"] = threshold
    output["score_margin"] = score - threshold
    output["predicted_essential"] = score >= threshold
    path = destination / "application_predictions.tsv"
    output.to_csv(path, sep="\t", index=False)
    manifest = {"status": "provisional single-species application refit; not a completed stability analysis",
                "species": species, "variant": variant, "fraction": args.fraction, "repeat": args.repeat,
                "n_training_records": len(ids), "n_application_genes": len(app_ids),
                "threshold_from_training_cv": threshold, "candidate": candidate,
                "annotation_block_masked": bool(args.annotation_excluded), "feature_count": 628,
                "dynamic_difference_vs_full_training_application_matrix": reconstruction_error,
                "input_order": list(SCOPES) + list(BRANCHES), "transformer_internal_stop_epoch": int(best_epoch),
                "oof_sha256": digest(oof_dir / "oof_probabilities.npy"),
                "frozen_candidate_sha256": digest(config_path), "application_matrix_sha256": digest(app_dir / "X_app.npy"),
                "application_predictions_sha256": digest(path), "oberstaller_training_pool": ober_stats,
                "test_or_validation_labels_used_for_fitting_or_threshold": False,
                "caveats": ["Retrospective profile choice included historical Pk test inspection.",
                            "Nine-route meta CV uses precomputed base OOF and is not fully nested.",
                            "Application contains fitted training proteins and is not an independent performance estimate.",
                            "Functional annotation is masked but other correlated annotations and static sequence features remain.",
                            "Only selected target training labels are used for dynamic features; other species training labels remain available to permitted scopes."]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

