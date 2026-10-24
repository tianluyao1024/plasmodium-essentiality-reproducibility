"""Build frozen nine-route group tables from the audited 100-repeat refit."""
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\ProtoEss_original_structure_revision_20261017")
OUT = ROOT / "final_refit_results_20261023"
AGG = ROOT / "nine_route_resampling_v1" / "aggregate_100_of_100_seed_20260821"
TRIPLETS = ROOT / "evidence" / "new_atlas_original_exact_triplets.tsv"
CANDIDATES = ROOT / "provisional_figure4_annotation_exclusion" / "single_copy_groups_full_vs_annotation_excluded.tsv"
ORTHO = ROOT / "original_orthomcl_context_fivefold_20261019" / "original_orthomcl_context_by_group.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    agg_manifest = json.loads((AGG / "manifest.json").read_text(encoding="utf-8"))
    assert agg_manifest["status"] == "complete"
    assert agg_manifest["completed_repeats"] == agg_manifest["expected_repeats"] == 100
    assert agg_manifest["candidate_groups"] == 455
    assert agg_manifest["final_stable_groups"] == 184

    triplets = pd.read_csv(TRIPLETS, sep="\t", dtype={"homology_group_id": str})
    candidate_universe = pd.read_csv(CANDIDATES, sep="\t", dtype={"homology_group_id": str})
    candidates = candidate_universe.loc[candidate_universe.retained_same_strong_direction].copy()
    recovery = pd.read_csv(AGG / "directional_refit_recovery.tsv", sep="\t", dtype={"homology_group_id": str})
    orthology = pd.read_csv(ORTHO, sep="\t", dtype={"homology_group_id": str})

    assert len(triplets) == 4183 and triplets.homology_group_id.is_unique
    assert len(candidate_universe) == 4183
    assert len(candidates) == len(recovery) == 455
    assert candidates.homology_group_id.is_unique and recovery.homology_group_id.is_unique
    assert set(candidates.homology_group_id) == set(recovery.homology_group_id)
    assert set(candidates.homology_group_id) <= set(triplets.homology_group_id)
    assert int(recovery.final_refit_stable.sum()) == 184
    assert recovery.completed_repeats.eq(100).all()

    joined = candidates[["homology_group_id", "retained_same_strong_direction", "new_pattern"]].merge(
        recovery, on="homology_group_id", validate="one_to_one"
    ).merge(
        triplets.rename(columns={"new_pattern": "new_pattern_atlas"}),
        on="homology_group_id", validate="one_to_one", suffixes=("_candidate", "_atlas")
    ).merge(
        orthology[["homology_group_id", "shared_orthomcl_group", "shared_orthomcl_group_support"]],
        on="homology_group_id", validate="one_to_one"
    )
    assert len(joined) == 455
    assert joined.retained_same_strong_direction.all()
    assert joined.new_strong_directional.all()
    assert joined.new_pattern.eq(joined.new_pattern_atlas).all()
    assert joined.final_refit_stable.sum() == 184

    stable = joined.loc[joined.final_refit_stable].copy()
    stable = stable.sort_values(["new_pattern_atlas", "recovery_frequency", "homology_group_id"], ascending=[True, False, True])
    patterns = stable.groupby("new_pattern_atlas", as_index=False).agg(
        stable_groups=("homology_group_id", "size"),
        median_recovery_frequency=("recovery_frequency", "median"),
        orthomcl_supported=("shared_orthomcl_group_support", "sum"),
    ).rename(columns={"new_pattern_atlas": "pattern"})
    all_patterns = joined.groupby("new_pattern_atlas", as_index=False).agg(candidate_groups=("homology_group_id", "size"))
    summary = all_patterns.merge(patterns, left_on="new_pattern_atlas", right_on="pattern", how="left").drop(columns="pattern")
    summary["stable_groups"] = summary.stable_groups.fillna(0).astype(int)
    summary["orthomcl_supported"] = summary.orthomcl_supported.fillna(0).astype(int)
    summary["recovery_threshold"] = 0.8
    summary["expected_repeats"] = 100
    summary = summary.rename(columns={"new_pattern_atlas": "pattern"}).sort_values("pattern")

    OUT.mkdir(parents=True)
    joined.to_csv(OUT / "F4_455_directional_candidates_100_refit_recovery.tsv", sep="\t", index=False)
    stable.to_csv(OUT / "F4_184_refit_stable_directional_groups.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "F4_directional_pattern_stability_summary.tsv", sep="\t", index=False)
    ortho_summary = pd.DataFrame([
        {"cohort": "all_exact_triplets", "groups": len(triplets),
         "supported": int(orthology.shared_orthomcl_group_support.sum())},
        {"cohort": "refit_stable_directional", "groups": len(stable),
         "supported": int(stable.shared_orthomcl_group_support.sum())},
    ])
    ortho_summary.to_csv(OUT / "F4_orthomcl_support_summary.tsv", sep="\t", index=False)

    inputs = [AGG / "manifest.json", AGG / "directional_refit_recovery.tsv", TRIPLETS, CANDIDATES, ORTHO]
    manifest = {
        "status": "complete_final_stability_tables_from_audited_100_of_100_refits",
        "aggregate_status": agg_manifest["status"],
        "completed_repeats": 100,
        "candidate_groups": len(joined),
        "stable_groups_at_recovery_frequency_ge_0.80": len(stable),
        "test_or_validation_labels_read_by_this_builder": False,
        "inputs_sha256": {str(p): sha256(p) for p in inputs},
        "outputs_sha256": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "notes": [
            "Group membership uses the prespecified threshold: both full and annotation-excluded models recover the reference pattern in at least 80 of 100 paired refits.",
            "Calls and margins are from the frozen nine-route matched-species atlas.",
            "Shared OrthoMCL membership is external orthology context, not experimental validation or proof of lineage causation.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "candidate_groups": len(joined), "stable_groups": len(stable),
                      "pattern_summary": summary.to_dict(orient="records"),
                      "stable_orthomcl_supported": int(stable.shared_orthomcl_group_support.sum())}, indent=2))


if __name__ == "__main__":
    main()
