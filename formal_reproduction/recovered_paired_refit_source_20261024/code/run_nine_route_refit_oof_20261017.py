"""Run/check five paired-refit OOF folds, without application or meta scoring."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nine_route_refit_partitions_20261017 import SCOPES, build_partition, id_digest

WORKSPACE = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")
FEATURES = Path(r"D:\ProtoEss_feature_matrix_repair_20261015\scope_feature_sets_v3_all_splits_20261016")
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
BRANCHES = ("lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", choices=SPECIES, required=True)
    p.add_argument("--fraction", type=float, default=.8)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--annotation-excluded", action="store_true")
    args = p.parse_args()
    if args.repeat < 0:
        raise ValueError("Repeat must be nonnegative")
    species = args.species
    train_path = WORKSPACE / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species" / species / "train_metadata.tsv"
    train = pd.read_csv(train_path, sep="\t", dtype={"record_id": str})
    ids = train.record_id.to_numpy(dtype=str)
    y = train.label.eq("essential").to_numpy(dtype=np.int8)
    seed = args.seed + args.repeat * 1009 + SPECIES.index(species)
    partition = build_partition(y, args.fraction, seed)
    selected = ids[partition.selected_indices]
    variant = "annotation_excluded" if args.annotation_excluded else "full"
    directory = OUT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}" / species / variant / "frozen_unlabelled"
    matrix_path = FEATURES / "within_species" / species / "X_train.npy"
    matrix_hash = digest(matrix_path)
    ids_hash = digest(FEATURES / "within_species" / species / "ids_train.npy")
    assembled = np.full((len(selected), 9), np.nan, dtype=np.float32)
    checkpoints = []
    for fold in range(5):
        file = directory / f"fold_{fold}.npz"
        if not file.exists():
            command = [sys.executable, str(CODE / "pilot_nine_route_refit_oof_20261017.py"),
                       "--species", species, "--fold", str(fold), "--fraction", str(args.fraction),
                       "--repeat", str(args.repeat), "--seed", str(args.seed), "--edge-mapping", "frozen_unlabelled"]
            if args.annotation_excluded:
                command.append("--annotation-excluded")
            subprocess.run(command, cwd=WORKSPACE, check=True)
        meta_path = file.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        block = np.load(file, allow_pickle=False)
        hold = partition.fold_assignment == fold
        fit_ids, hold_ids = selected[~hold], selected[hold]
        checks = [meta["species"] == species, meta["variant"] == variant, meta["fold"] == fold,
                  meta["partition_seed"] == seed, meta["training_matrix_sha256"] == matrix_hash,
                  meta["training_ids_sha256"] == ids_hash, meta["output_sha256"] == digest(file),
                  meta["fit_ids_sha256"] == id_digest(fit_ids), meta["held_ids_sha256"] == id_digest(hold_ids),
                  meta["input_order"] == list(SCOPES) + list(BRANCHES),
                  np.array_equal(block["held_ids"], hold_ids), block["probabilities"].shape == (len(hold_ids), 9)]
        if not all(checks):
            raise ValueError(f"Checkpoint does not match fixed inputs: {file}")
        assembled[hold] = block["probabilities"]
        checkpoints.append({"fold": fold, "npz_sha256": digest(file), "metadata_sha256": digest(meta_path)})
        print(f"Checked {species} {variant}: fold {fold + 1}/5", flush=True)
    if not np.isfinite(assembled).all() or np.any((assembled < 0) | (assembled > 1)):
        raise ValueError("Incomplete or invalid OOF matrix")
    np.save(directory / "oof_probabilities.npy", assembled)
    np.save(directory / "selected_ids.npy", np.asarray(selected, dtype=str))
    np.save(directory / "fold_assignments.npy", partition.fold_assignment)
    manifest = {"status": "five-fold base OOF inputs only; application and meta model not yet computed",
                "species": species, "variant": variant, "fraction": args.fraction, "repeat": args.repeat,
                "partition_seed": seed, "records": len(selected), "folds": 5,
                "input_order": list(SCOPES) + list(BRANCHES), "checkpoints": checkpoints,
                "oof_sha256": digest(directory / "oof_probabilities.npy"),
                "selected_ids_sha256": digest(directory / "selected_ids.npy"),
                "training_labels_source": str(train_path), "training_labels_source_sha256": digest(train_path),
                "historical_test_caveat": "Model route was selected after historical test inspection.",
                "limitations": ["Meta scoring, application features and complete repeated refits are not computed here.",
                                "Unlabelled validation IDs are used for frozen PPI node mapping.",
                                "Transformer internal stop feature reconstruction is not fully nested."]}
    (directory / "oof_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

