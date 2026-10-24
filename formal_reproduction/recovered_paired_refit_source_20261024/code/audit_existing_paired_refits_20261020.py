"""Verify completed paired-refit repeats against the frozen source snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REVISION = Path(r"D:\ProtoEss_original_structure_revision_20261017")
REPEATS = REVISION / "nine_route_resampling_v1" / "repeats"
SNAPSHOT = REVISION / "nine_route_resampling_v1" / "source_manifest.json"
SUMMARISER = Path(__file__).resolve().with_name("summarise_paired_nine_route_repeat_20261017.py")
OUTPUT = REVISION / "paired_refit_interim_audit_20261020.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    snapshot_hash = sha256(SNAPSHOT)
    summariser_hash = sha256(SUMMARISER)
    records = []
    for repeat in range(100):
        folder = REPEATS / f"repeat_{repeat:03d}_seed_20260821"
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures = []
        if manifest.get("repeat") != repeat:
            failures.append("repeat index mismatch")
        if manifest.get("candidate_groups") != 455:
            failures.append("candidate group count mismatch")
        if manifest.get("source_snapshot_sha256") != snapshot_hash:
            failures.append("source snapshot mismatch")
        if manifest.get("summariser_sha256") != summariser_hash:
            failures.append("summariser hash mismatch")
        for raw_path, expected in manifest.get("input_sha256", {}).items():
            path = Path(raw_path)
            if not path.is_file():
                failures.append(f"missing input: {path}")
            elif sha256(path) != expected:
                failures.append(f"input hash mismatch: {path}")
        result_path = folder / "paired_directional_recovery.tsv"
        if not result_path.is_file() or sha256(result_path) != manifest.get("output_sha256"):
            failures.append("paired recovery output missing or hash mismatch")
        records.append({
            "repeat": repeat,
            "status": manifest.get("status"),
            "candidate_groups": manifest.get("candidate_groups"),
            "recovered_by_both": manifest.get("reference_pattern_recovered_by_both"),
            "verified": not failures,
            "failures": failures,
        })

    observed = [item["repeat"] for item in records]
    if len(observed) != len(set(observed)):
        raise ValueError("Completed repeat manifests contain duplicate repeat indices")
    if any(not item["verified"] for item in records):
        raise ValueError("At least one interim repeat failed integrity checks")
    missing = sorted(set(range(100)) - set(observed))
    result = {
        "status": ("100 completed repeats integrity-verified" if not missing else
                   f"{len(records)} completed repeats integrity-verified; interim only, not final 100-repeat stability analysis"),
        "verified_repeat_count": len(records),
        "required_repeat_count": 100,
        "unverified_or_incomplete_repeat_count": len(missing),
        "missing_repeat_indices": missing,
        "source_snapshot_sha256": snapshot_hash,
        "summariser_sha256": summariser_hash,
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "test_or_validation_labels_read_by_this_audit": False,
        "repeats": records,
        "note": "Integrity verification does not make an interim subset a final stability estimate. Missing or incomplete repeats are not included.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

