"""Run resumable paired refits; incomplete application artifacts require inspection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audit_nine_route_application_refit_20261017 import sha256
from nine_route_refit_partitions_20261017 import build_partition
from run_nine_route_refit_oof_20261017 import SPECIES
import snapshot_nine_route_refit_sources_20261017 as snapshot

WORKSPACE = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
PILOT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")


def call(script, *args):
    subprocess.run([sys.executable, str(CODE / script), *map(str, args)], cwd=WORKSPACE, check=True)


def check_snapshot():
    frozen = json.loads(snapshot.OUT.read_text(encoding="utf-8"))
    if frozen["files"] != snapshot.inventory() or frozen["environment"] != snapshot.environment():
        raise RuntimeError("Source files or runtime differ from the pre-run snapshot")


def verify_oof(directory, species, seed, repeat, fraction, variant):
    manifest = json.loads((directory / "oof_manifest.json").read_text(encoding="utf-8"))
    train = WORKSPACE / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species" / species / "train_metadata.tsv"
    if (manifest["species"] != species or manifest["variant"] != variant or manifest["repeat"] != repeat
            or manifest["fraction"] != fraction or manifest["partition_seed"] != seed + repeat * 1009 + SPECIES.index(species)
            or sha256(train) != manifest["training_labels_source_sha256"]
            or sha256(directory / "oof_probabilities.npy") != manifest["oof_sha256"]
            or sha256(directory / "selected_ids.npy") != manifest["selected_ids_sha256"]):
        raise ValueError(f"OOF manifest failed verification: {directory}")
    metadata = pd.read_csv(train, sep="\t", dtype={"record_id": str})
    partition = build_partition(metadata.label.eq("essential").to_numpy(dtype=np.int8), fraction, manifest["partition_seed"])
    ids = metadata.record_id.to_numpy(dtype=str)[partition.selected_indices]
    if (not np.array_equal(ids, np.load(directory / "selected_ids.npy", allow_pickle=False))
            or not np.array_equal(partition.fold_assignment, np.load(directory / "fold_assignments.npy", allow_pickle=False))):
        raise ValueError(f"OOF subset/folds do not match requested partition: {directory}")


def ensure_species(species, repeat, seed, fraction):
    root = PILOT / f"fraction_{fraction:.3f}_repeat_{repeat:03d}_seed_{seed}" / species
    for variant in ("full", "annotation_excluded"):
        check_snapshot()
        common = ["--species", species, "--fraction", fraction, "--repeat", repeat, "--seed", seed]
        switches = ["--annotation-excluded"] if variant == "annotation_excluded" else []
        oof = root / variant / "frozen_unlabelled"
        if not (oof / "oof_manifest.json").exists():
            call("run_nine_route_refit_oof_20261017.py", *common, *switches)
        verify_oof(oof, species, seed, repeat, fraction, variant)
        app = root / variant / "application"
        if not app.exists():
            call("score_nine_route_refit_application_20261017.py", *common, *switches)
        if not (app / "manifest.json").exists():
            raise RuntimeError(f"Partial application artifacts preserved for manual inspection: {app}")
        manifest = json.loads((app / "manifest.json").read_text(encoding="utf-8"))
        predictions = app / "application_predictions.tsv"
        if (manifest["species"] != species or manifest["variant"] != variant
                or manifest["fraction"] != fraction or manifest["repeat"] != repeat
                or sha256(predictions) != manifest["application_predictions_sha256"]
                or manifest["oof_sha256"] != sha256(oof / "oof_probabilities.npy")):
            raise ValueError(f"Application manifest failed verification: {app}")
        audit = app / "independent_score_audit.json"
        if not audit.exists():
            call("audit_nine_route_application_refit_20261017.py", *common, "--variant", variant)
        if json.loads(audit.read_text(encoding="utf-8"))["prediction_sha256"] != sha256(predictions):
            raise ValueError(f"Application audit failed verification: {audit}")
        frame = pd.read_csv(predictions, sep="\t")
        model = joblib.load(app / "models" / "nine_route_meta_lightgbm.joblib")
        inputs = frame[[f"route_{name}_probability" for name in manifest["input_order"]]].to_numpy(dtype=np.float32)
        if not np.allclose(model.predict_proba(inputs)[:, 1], frame.score.to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"Stored model no longer reproduces application scores: {app}")
        check_snapshot()
    pair = root / "paired_refit_audit.json"
    if not pair.exists():
        call("audit_nine_route_paired_refit_20261017.py", "--species", species, "--fraction", fraction,
             "--repeat", repeat, "--seed", seed)
    pair_data = json.loads(pair.read_text(encoding="utf-8"))
    if (pair_data["species"] != species or pair_data["seed"] != seed or pair_data["repeat"] != repeat
            or pair_data["fraction"] != fraction or not pair_data["same_selected_training_ids"]
            or not pair_data["same_fivefold_assignment"]):
        raise ValueError(f"Paired audit parameters do not match: {pair}")
    for variant in ("full", "annotation_excluded"):
        if (pair_data["artifacts"][variant]["prediction_sha256"] != sha256(root / variant / "application" / "application_predictions.tsv")
                or pair_data["artifacts"][variant]["oof_sha256"] != sha256(root / variant / "frozen_unlabelled" / "oof_probabilities.npy")):
            raise ValueError(f"Paired audit failed verification: {pair}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True, help="exclusive; max 100")
    args = p.parse_args()
    if not 0 <= args.start < args.end <= 100:
        raise ValueError("Require 0 <= start < end <= 100")
    check_snapshot()
    for repeat in range(args.start, args.end):
        for species in SPECIES:
            print(f"Repeat {repeat + 1}/100: {species}", flush=True)
            ensure_species(species, repeat, 20260821, .8)
        check_snapshot()
        repeat_dir = snapshot.OUT.parent / "repeats" / f"repeat_{repeat:03d}_seed_20260821"
        if not repeat_dir.exists():
            call("summarise_paired_nine_route_repeat_20261017.py", "--repeat", repeat)
        else:
            manifest = json.loads((repeat_dir / "manifest.json").read_text(encoding="utf-8"))
            if sha256(repeat_dir / "paired_directional_recovery.tsv") != manifest["output_sha256"]:
                raise ValueError(f"Repeat output hash mismatch: {repeat_dir}")
        print(f"Repeat {repeat + 1}/100 complete", flush=True)


if __name__ == "__main__":
    main()

