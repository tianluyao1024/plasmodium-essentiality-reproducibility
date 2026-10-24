"""Summarise one complete three-species paired resampling repeat."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_nine_route_application_refit_20261017 import sha256
import snapshot_nine_route_refit_sources_20261017 as source_snapshot

ROOT = Path(r"D:\ProtoEss_original_structure_revision_20261017")
INPUT = ROOT / "provisional_figure4_annotation_exclusion" / "single_copy_groups_full_vs_annotation_excluded.tsv"
PILOT = ROOT / "nine_route_refit_pilot"
OUT = ROOT / "nine_route_resampling_v1" / "repeats"
SPECIES = {"Pf": "Plasmodium_falciparum", "Pb": "Plasmodium_berghei", "Pk": "Plasmodium_knowlesi"}


def expected_recovery(reference, full, excluded):
    return (full == reference) & (excluded == reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--fraction", type=float, default=.8)
    args = parser.parse_args()
    frozen_sources = json.loads(source_snapshot.OUT.read_text(encoding="utf-8"))
    if (frozen_sources["files"] != source_snapshot.inventory()
            or frozen_sources["environment"] != source_snapshot.environment()):
        raise ValueError("Refit source snapshot changed before repeat aggregation")
    triplets = pd.read_csv(INPUT, sep="\t", dtype={"homology_group_id": str})
    triplets = triplets.loc[triplets.retained_same_strong_direction.astype(bool)].copy()
    if len(triplets) != 455 or triplets.homology_group_id.duplicated().any():
        raise ValueError("Unexpected frozen annotation-retained groups")
    sources = {str(INPUT): sha256(INPUT)}
    for short, species in SPECIES.items():
        base = PILOT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}" / species
        pair_path = base / "paired_refit_audit.json"
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if pair["species"] != species or pair["repeat"] != args.repeat or pair["fraction"] != args.fraction:
            raise ValueError("Paired-refit audit does not match repeat")
        if not pair["same_selected_training_ids"] or not pair["same_fivefold_assignment"]:
            raise ValueError("Paired-refit training partitions differ")
        sources[str(pair_path)] = sha256(pair_path)
        for variant in ("full", "annotation_excluded"):
            path = base / variant / "application" / "application_predictions.tsv"
            if sha256(path) != pair["artifacts"][variant]["prediction_sha256"]:
                raise ValueError("Application file hash differs from paired audit")
            frame = pd.read_csv(path, sep="\t", dtype={"gene_id": str}, usecols=["gene_id", "predicted_essential"])
            if frame.gene_id.isna().any() or frame.gene_id.duplicated().any():
                raise ValueError(f"Nonunique gene IDs in {species} {variant}")
            frame = frame.rename(columns={"gene_id": f"{short}_gene", "predicted_essential": f"{short}_{variant}"})
            triplets = triplets.merge(frame, on=f"{short}_gene", how="left", validate="many_to_one")
            if triplets[f"{short}_{variant}"].isna().any():
                raise ValueError(f"Missing candidate scores for {species} {variant}")
            sources[str(path)] = sha256(path)
    for variant in ("full", "annotation_excluded"):
        a = [np.where(triplets[f"{s}_{variant}"].astype(bool), "E", "NE") for s in SPECIES]
        triplets[f"{variant}_pattern"] = np.char.add(np.char.add(np.char.add(np.char.add(a[0], "/"), a[1]), "/"), a[2])
    triplets["matches_reference_both"] = expected_recovery(triplets.new_pattern, triplets.full_pattern, triplets.annotation_excluded_pattern)
    result = triplets[["homology_group_id", "Pf_gene", "Pb_gene", "Pk_gene", "new_pattern", "full_pattern",
                       "annotation_excluded_pattern", "matches_reference_both"]].copy()
    destination = OUT / f"repeat_{args.repeat:03d}_seed_{args.seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    path = destination / "paired_directional_recovery.tsv"
    result.to_csv(path, sep="\t", index=False)
    manifest = {"status": "one complete paired repeat; not the 100-repeat stability result", "repeat": args.repeat,
                "seed": args.seed, "fraction": args.fraction, "candidate_groups": len(result),
                "reference_pattern_recovered_by_both": int(result.matches_reference_both.sum()),
                "recovery_rule": "Both models recover the original E/NE pattern; no repeat-specific margin filter.",
                "input_sha256": sources, "source_snapshot_sha256": sha256(source_snapshot.OUT),
                "summariser_sha256": sha256(Path(__file__)), "output_sha256": sha256(path),
                "caveat": "Repeated training performance is not an independent screen or prospective validation."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

