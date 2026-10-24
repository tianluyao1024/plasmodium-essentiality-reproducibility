"""Aggregate completed paired repeats without treating partial runs as final."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from audit_nine_route_application_refit_20261017 import sha256

ROOT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_resampling_v1")


def aggregate_tables(tables, expected_repeats=100):
    if not tables or len(tables) > expected_repeats:
        raise ValueError("Invalid number of repeats")
    grouped = pd.concat(tables, ignore_index=True)
    if grouped.duplicated(["repeat", "homology_group_id"]).any():
        raise ValueError("Duplicate group/repeat")
    counts = grouped.groupby("homology_group_id", as_index=False).agg(
        completed_repeats=("repeat", "nunique"), recovered=("matches_reference_both", "sum"))
    if not counts.completed_repeats.eq(len(tables)).all():
        raise ValueError("Different candidate groups across repeats")
    counts["recovery_frequency"] = counts.recovered / counts.completed_repeats
    complete = len(tables) == expected_repeats
    counts["final_refit_stable"] = ((counts.recovered >= math.ceil(.8 * expected_repeats)).astype("boolean")
                                    if complete else pd.Series(pd.NA, index=counts.index, dtype="boolean"))
    return counts, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.expected_repeats < 1:
        raise ValueError("Expected repeats must be positive")
    tables, seen, sources, common_snapshot, common_candidates, common_summariser = [], set(), {}, None, None, None
    for directory in sorted((ROOT / "repeats").glob(f"repeat_*_seed_{args.seed}")):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        repeat = int(manifest["repeat"])
        if (repeat in seen or repeat < 0 or repeat >= args.expected_repeats
                or manifest["seed"] != args.seed or manifest["fraction"] != .8):
            raise ValueError("Duplicate or unexpected repeat")
        candidate_hash = manifest["input_sha256"][str(ROOT.parent / "provisional_figure4_annotation_exclusion" / "single_copy_groups_full_vs_annotation_excluded.tsv")]
        snapshot_hash = manifest["source_snapshot_sha256"]
        if common_snapshot is None:
            common_snapshot, common_candidates, common_summariser = snapshot_hash, candidate_hash, manifest["summariser_sha256"]
        if (common_snapshot != snapshot_hash or common_candidates != candidate_hash
                or common_summariser != manifest["summariser_sha256"]):
            raise ValueError("Repeat uses a different source snapshot, candidate set, or summariser")
        path = directory / "paired_directional_recovery.tsv"
        if sha256(path) != manifest["output_sha256"]:
            raise ValueError(f"Repeat output hash mismatch: {path}")
        table = pd.read_csv(path, sep="\t", dtype={"homology_group_id": str})
        if len(table) != manifest["candidate_groups"] or len(table) != 455:
            raise ValueError("Unexpected repeat candidate count")
        table["repeat"] = repeat
        tables.append(table[["homology_group_id", "matches_reference_both", "repeat"]])
        seen.add(repeat)
        sources[str(manifest_path)], sources[str(path)] = sha256(manifest_path), sha256(path)
    counts, complete = aggregate_tables(tables, args.expected_repeats)
    if complete and seen != set(range(args.expected_repeats)):
        raise ValueError("Complete repeat sequence has missing indices")
    suffix = "" if complete else "_nullable"
    destination = ROOT / f"aggregate_{len(tables):03d}_of_{args.expected_repeats:03d}_seed_{args.seed}{suffix}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    path = destination / "directional_refit_recovery.tsv"
    counts.to_csv(path, sep="\t", index=False)
    manifest = {"status": "complete" if complete else "partial exploratory summary, not a final stability result",
                "completed_repeats": len(tables), "expected_repeats": args.expected_repeats,
                "candidate_groups": len(counts), "final_stable_groups": int(counts.final_refit_stable.sum()) if complete else None,
                "median_partial_recovery_frequency": float(counts.recovery_frequency.median()),
                "input_sha256": sources, "output_sha256": sha256(path),
                "aggregator_sha256": sha256(Path(__file__).resolve())}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
