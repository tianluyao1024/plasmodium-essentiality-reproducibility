"""Package the audited 100-repeat Fig. 4 stability outputs without local paths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing repeats/ and aggregate_100_of_100_seed_20260821/")
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    release = args.release_dir.resolve()
    output = release / "source_data" / "Fig4_refit_stability_100_repeats_20261024"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    aggregate_dir = source / "aggregate_100_of_100_seed_20260821"
    aggregate_manifest = json.loads((aggregate_dir / "manifest.json").read_text(encoding="utf-8"))
    if aggregate_manifest["status"] != "complete" or aggregate_manifest["completed_repeats"] != 100:
        raise ValueError("Source aggregate is not the complete 100-repeat run")
    aggregate_table = aggregate_dir / "directional_refit_recovery.tsv"
    shutil.copy2(aggregate_table, output / "aggregate_expected_refit_recovery.tsv")

    repeat_manifests: list[dict[str, object]] = []
    candidate_ids: list[str] | None = None
    files: dict[str, str] = {"aggregate_expected_refit_recovery.tsv": sha256(output / "aggregate_expected_refit_recovery.tsv")}
    snapshots: set[str] = set()
    for repeat in range(100):
        source_dir = source / "repeats" / f"repeat_{repeat:03d}_seed_20260821"
        original_manifest_path = source_dir / "manifest.json"
        result_path = source_dir / "paired_directional_recovery.tsv"
        original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        rows = read_tsv(result_path)
        ids = sorted(row["homology_group_id"] for row in rows)
        if len(rows) != 455 or len(set(ids)) != 455:
            raise ValueError(f"Unexpected candidate table size/IDs for repeat {repeat}")
        if candidate_ids is None:
            candidate_ids = ids
        elif ids != candidate_ids:
            raise ValueError(f"Candidate group universe changed in repeat {repeat}")
        snapshots.add(str(original["source_snapshot_sha256"]))

        relative_dir = f"repeat_{repeat:03d}_seed_20260821"
        target_dir = output / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_result = target_dir / "paired_directional_recovery.tsv"
        shutil.copy2(result_path, target_result)
        files[f"{relative_dir}/paired_directional_recovery.tsv"] = sha256(target_result)
        per_repeat = {
            "repeat": repeat,
            "seed": int(original["seed"]),
            "training_fraction": float(original["fraction"]),
            "candidate_groups": int(original["candidate_groups"]),
            "reference_pattern_recovered_by_both": int(original["reference_pattern_recovered_by_both"]),
            "recovery_rule": original["recovery_rule"],
            "source_snapshot_sha256": original["source_snapshot_sha256"],
            "paired_result_sha256": files[f"{relative_dir}/paired_directional_recovery.tsv"],
            "original_repeat_manifest_sha256": sha256(original_manifest_path),
        }
        (target_dir / "repeat_manifest.json").write_text(json.dumps(per_repeat, indent=2) + "\n", encoding="utf-8")
        repeat_manifests.append(per_repeat)

    if snapshots != {"64d862d063114160d3fc03e2bbd38fba8c811e5d5c5698df7711e46da23d92b3"}:
        raise ValueError(f"Unexpected source snapshot hashes: {sorted(snapshots)}")
    if candidate_ids is None:
        raise ValueError("No candidate IDs found")
    manifest = {
        "status": "historical_100_of_100_paired_refit_outputs_packaged_for_audit",
        "repeats": 100,
        "seed": 20260821,
        "training_fraction": 0.8,
        "candidate_groups": 455,
        "candidate_group_ids": candidate_ids,
        "stability_threshold": 0.8,
        "stable_group_count_expected": 184,
        "source_snapshot_sha256": next(iter(snapshots)),
        "aggregate_reference_sha256": sha256(aggregate_table),
        "source_scope": "Saved paired refit outcomes and aggregate table; no labels for formal test rows are included in this evidence subpackage.",
        "reproducibility_boundary": "This package verifies saved repeat outputs and their aggregate against frozen Fig. 4. It does not refit models. The exact orchestration command for all 100 fitting repeats is not pinned in the archived run manifests; do not claim a full refit reproduction.",
        "repeat_manifests": repeat_manifests,
        "files_sha256": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "repeats": 100, "candidate_groups": len(candidate_ids),
                      "source_snapshot_sha256": next(iter(snapshots)), "result_file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
