"""Rebuild one five-fold nine-route OOF fold for refit-pipeline verification.

This is a pilot, not the annotation-excluded application or 100-refit study.
It only reads training labels and writes to an isolated provisional directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nine_route_refit_partitions_20261017 import SCOPES, build_partition, id_digest, make_scope_pools


WORKSPACE = Path(__file__).resolve().parents[3]
DATA = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
FEATURES = DATA / "scope_feature_sets_v3_all_splits_20261016"
OUT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
BRANCHES = ("lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer")

os.environ["PROTOESS_SCOPE_FEATURE_ROOT"] = str(FEATURES)
os.environ["PROTOESS_SOURCE_SCOPE_FEATURE_ROOT"] = str(WORKSPACE / "eukaryote_model_datasets" / "scope_feature_sets_v2")
import compare_homology_label_scopes as scopes  # noqa: E402
import train_scope_strict_oof_stack as strict  # noqa: E402
import train_scope_transformer_oof as transformer  # noqa: E402


def file_sha(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=SPECIES, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--fraction", type=float, default=.8)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--annotation-excluded", action="store_true")
    parser.add_argument("--edge-mapping", choices=("selected_only", "frozen_unlabelled"), default="frozen_unlabelled")
    args = parser.parse_args()
    if Path.cwd().resolve() != WORKSPACE.resolve():
        raise RuntimeError(f"Run from workspace directory {WORKSPACE}")
    if args.repeat < 0:
        raise ValueError("Repeat must be nonnegative")
    species = args.species
    seed = args.seed + args.repeat * 1009 + SPECIES.index(species)
    variant = "annotation_excluded" if args.annotation_excluded else "full"
    out = OUT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}" / species / variant / args.edge_mapping
    output = out / f"fold_{args.fold}.npz"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing pilot: {output}")

    frame = scopes.read_metadata("within_species", species, "train")
    matrix_dir = FEATURES / "within_species" / species
    ids = frame.record_id.astype(str).to_numpy()
    if not np.array_equal(ids, np.load(matrix_dir / "ids_train.npy", allow_pickle=True).astype(str)):
        raise ValueError("Training IDs do not match frozen matrix order")
    X = np.asarray(np.load(matrix_dir / "X_train.npy", mmap_mode="r"), dtype=np.float32)
    names = json.loads((matrix_dir / "feature_names.json").read_text(encoding="utf-8"))
    if X.shape != (len(ids), 628) or len(names) != 628 or not np.isfinite(X).all():
        raise ValueError("Invalid raw feature matrix")
    annotation_ix = np.flatnonzero(np.char.startswith(np.asarray(names, dtype=str), "go_ipr_pfam_"))
    if not np.array_equal(annotation_ix, np.arange(287, 415)):
        raise ValueError("Functional-annotation columns do not match the original block")
    y = frame.label.eq("essential").astype(np.int8).to_numpy()
    part = build_partition(y, args.fraction, seed)
    selected_ix = part.selected_indices
    selected = frame.iloc[selected_ix].reset_index(drop=True)
    fit_local = np.flatnonzero(part.fold_assignment != args.fold)
    held_local = np.flatnonzero(part.fold_assignment == args.fold)
    fit = selected.iloc[fit_local].copy()
    held = selected.iloc[held_local].copy()
    fit_ids, held_ids = fit.record_id.astype(str).to_numpy(), held.record_id.astype(str).to_numpy()
    if set(fit_ids) & set(held_ids):
        raise ValueError("OOF fit and held IDs overlap")
    fit_y = fit.label.eq("essential").astype(np.int8).to_numpy()
    all_train = scopes.read_all_training_metadata()
    ober, _ = scopes.build_oberstaller_training_pool(all_train)
    pools = make_scope_pools(species, fit, all_train, ober)
    for mode, pool in pools.items():
        if set(held_ids) & set(pool.record_id):
            raise ValueError(f"Held OOF label in {mode}")

    hits = scopes.read_hits(species, "train")
    if args.edge_mapping == "frozen_unlabelled":
        val_path = WORKSPACE / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species" / species / "val_metadata.tsv"
        val_ids = pd.read_csv(val_path, sep="\t", usecols=["record_id"], dtype=str).record_id.to_numpy()
        mapping_ids = set(ids) | set(val_ids)
    else:
        mapping_ids = set(selected.record_id.astype(str))
    edges = scopes.read_target_species_edges(species, mapping_ids)
    frames = {"train": selected}
    fit_raw = strict.fold_features(X[selected_ix[fit_local]].copy(), frames, hits, edges,
                                   set(fit_ids), fit_ids, 592)
    held_raw = strict.fold_features(X[selected_ix[held_local]].copy(), frames, hits, edges,
                                    set(fit_ids), held_ids, 592)
    if args.annotation_excluded:
        fit_raw[:, annotation_ix] = 0
        held_raw[:, annotation_ix] = 0
    hom_idx = np.asarray([i for i, name in enumerate(names) if name.startswith(("homology30_", "homology50_"))])
    hom_local = np.asarray([j for j, i in enumerate(hom_idx) if int(names[i].rsplit("_", 1)[1]) >= 2])
    if len(hom_idx) != 30 or len(hom_local) != 26:
        raise ValueError("Unexpected homology label columns")
    predictions = []
    for mode in SCOPES:
        hfit, hheld = scopes.scaled_homology(fit, held, pools[mode])
        xin, xout = fit_raw.copy(), held_raw.copy()
        xin[:, hom_idx[hom_local]] = hfit[:, hom_local]
        xout[:, hom_idx[hom_local]] = hheld[:, hom_local]
        model = scopes.make_model(100 + args.fold)
        model.fit(xin, fit_y)
        predictions.append(model.predict_proba(xout)[:, 1])
        print(f"{species} fold {args.fold} {variant}: {mode} complete", flush=True)

    Xfit, Xheld, _, _ = strict.fit_profile_matrix(fit_raw, held_raw, names, profile=None, seed=100 + args.fold)
    lightgbm = strict.lgbm(Xfit, fit_y, 100 + args.fold)
    catboost = strict.cat(Xfit, fit_y, 200 + args.fold)
    expert = strict.lgbm(Xfit, fit_y, 300 + args.fold)
    predictions.extend([lightgbm.predict_proba(Xheld)[:, 1], catboost.predict_proba(Xheld)[:, 1],
                        expert.predict_proba(Xheld)[:, 1], np.repeat(float(fit_y.mean()), len(held_ids))])
    fit_groups, held_groups, group_names = transformer.fit_profile_groups(
        fit_raw, held_raw, names, profile=None, seed=42 + args.fold)
    held_modalities = transformer.transformer_modalities(held_groups, group_names)
    pred, _, best_epoch = transformer.train_predict(
        transformer.transformer_modalities(fit_groups, group_names), fit_y,
        held_modalities, None, held_modalities, 42 + args.fold)
    predictions.append(pred)
    prob = np.column_stack(predictions).astype(np.float32)
    if prob.shape != (len(held_ids), 9) or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Invalid nine-route probabilities")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, held_ids=np.asarray(held_ids, dtype=str), probabilities=prob)
    metadata = {"status": "one pilot OOF fold; not a full refit or stability result", "species": species,
                "variant": variant, "fraction": args.fraction, "repeat": args.repeat,
                "partition_seed": seed, "fold": args.fold, "feature_count": 628,
                "ppi_edge_mapping": args.edge_mapping,
                "annotation_block_masked": bool(args.annotation_excluded), "fit_records": len(fit),
                "held_records": len(held), "fit_ids_sha256": id_digest(fit_ids),
                "held_ids_sha256": id_digest(held_ids), "input_order": list(SCOPES) + list(BRANCHES),
                "transformer_internal_stop_epoch": int(best_epoch), "output_sha256": file_sha(output),
                "training_matrix_sha256": file_sha(matrix_dir / "X_train.npy"),
                "training_ids_sha256": file_sha(matrix_dir / "ids_train.npy"),
                "test_or_validation_labels_used": False,
                "caveats": ["Frozen edge mapping uses training and validation record IDs, without validation labels.",
                            "Internal Transformer stop split is not fully nested with dynamic feature rebuilding.",
                            "The global unlabelled sequence clustering includes other species' proteins.",
                            "Full application scoring, threshold estimation and repeated-refit stability are not implemented by this pilot."]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
