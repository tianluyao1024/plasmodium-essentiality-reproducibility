"""Audit the archived 100-repeat paired-refit evidence against frozen Fig. 4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    evidence = release / "source_data" / "Fig4_refit_stability_100_repeats_20261024"
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repeats = int(manifest["repeats"])
    expected_groups = int(manifest["candidate_groups"])
    failures: list[str] = []

    for relative, expected_hash in manifest["files_sha256"].items():
        path = evidence / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected_hash:
            failures.append(f"sha256:{relative}")

    recovery_count = {group: 0 for group in manifest["candidate_group_ids"]}
    per_group_repeat_count = {group: 0 for group in manifest["candidate_group_ids"]}
    source_snapshots: set[str] = set()
    for repeat in range(repeats):
        folder = evidence / f"repeat_{repeat:03d}_seed_{manifest['seed']}"
        repeat_manifest = json.loads((folder / "repeat_manifest.json").read_text(encoding="utf-8"))
        if int(repeat_manifest["repeat"]) != repeat:
            failures.append(f"repeat_index:{repeat}")
        source_snapshots.add(repeat_manifest["source_snapshot_sha256"])
        result_path = folder / "paired_directional_recovery.tsv"
        if sha256(result_path) != repeat_manifest["paired_result_sha256"]:
            failures.append(f"repeat_result_sha256:{repeat}")
        rows = read_tsv(result_path)
        observed = [row["homology_group_id"] for row in rows]
        if len(rows) != expected_groups or len(set(observed)) != expected_groups:
            failures.append(f"candidate_rows:{repeat}")
        if set(observed) != set(recovery_count):
            failures.append(f"candidate_universe:{repeat}")
        for row in rows:
            group = row["homology_group_id"]
            if group not in recovery_count:
                continue
            per_group_repeat_count[group] += 1
            if row["matches_reference_both"].strip().lower() == "true":
                recovery_count[group] += 1

    aggregate = {row["homology_group_id"]: row for row in read_tsv(evidence / "aggregate_expected_refit_recovery.tsv")}
    frozen_path = release / "source_data" / "Fig4_final_frozen_20261024" / "Fig4d_refit_recovery.tsv"
    frozen = {row["homology_group_id"]: row for row in read_tsv(frozen_path)}
    if set(aggregate) != set(recovery_count):
        failures.append("aggregate_group_universe")
    if set(frozen) != set(recovery_count):
        failures.append("frozen_Fig4_group_universe")

    for group, recovered in recovery_count.items():
        expected = aggregate.get(group)
        final = frozen.get(group)
        if expected is None or final is None:
            continue
        if int(expected["completed_repeats"]) != per_group_repeat_count[group] or int(expected["recovered"]) != recovered:
            failures.append(f"aggregate_count:{group}")
        if abs(float(expected["recovery_frequency"]) - recovered / repeats) > 1e-12:
            failures.append(f"aggregate_frequency:{group}")
        stable = recovered / repeats >= float(manifest["stability_threshold"])
        if (expected["final_refit_stable"].lower() == "true") != stable:
            failures.append(f"aggregate_stable_call:{group}")
        if int(final["completed_repeats"]) != per_group_repeat_count[group] or int(final["recovered"]) != recovered:
            failures.append(f"frozen_Fig4_count:{group}")
        if (final["final_refit_stable"].lower() == "true") != stable:
            failures.append(f"frozen_Fig4_stable_call:{group}")

    stable_groups = sum(count / repeats >= float(manifest["stability_threshold"]) for count in recovery_count.values())
    report = {
        "status": "passed" if not failures else "failed",
        "scope": "hash and value audit of saved paired-refit outputs; does not rerun model fitting",
        "repeats": repeats,
        "paired_result_files_checked": repeats,
        "candidate_groups_per_repeat": expected_groups,
        "unique_candidate_groups": len(recovery_count),
        "stable_groups_at_or_above_threshold": stable_groups,
        "stability_threshold": manifest["stability_threshold"],
        "unique_source_snapshot_hashes": len(source_snapshots),
        "source_snapshot_sha256": sorted(source_snapshots),
        "aggregate_and_frozen_fig4_rows_match": not any("aggregate_" in item or "frozen_Fig4_" in item for item in failures),
        "failures": failures,
        "reproducibility_boundary": manifest["reproducibility_boundary"],
    }
    (evidence / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
