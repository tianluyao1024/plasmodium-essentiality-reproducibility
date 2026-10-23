from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path


RELEASE_ID = "v20261024_fivefold_nine_route_candidate"
WEBSITE_FILES = [
    "predictions.tsv",
    "loso_metrics.tsv",
    "source_composition_metrics.tsv",
    "pk_dual_screen_gene_pairs.tsv",
    "pk_dual_screen_summary.tsv",
    "s4_label_exchange_runs.tsv",
    "s4_label_exchange_summary.tsv",
    "MODEL_CARD.md",
    "DATA_DICTIONARY.md",
    "final_freeze_manifest.json",
]
SOURCE_GROUPS = {
    "Fig2": ["F2a_performance.tsv", "F2b_branch_benchmarks.tsv", "F2c_calibration.tsv", "manifest.json"],
    "Fig2d_sensitivity": ["F2d_five_allocation_summary.tsv", "F2d_per_allocation.tsv", "manifest.json"],
    "Fig3": ["F3a_LOSO.tsv", "F3bc_full_stack_bootstrap.tsv", "F3bc_shared_bootstrap.tsv",
             "F3d_full_stack_metrics.tsv", "F3d_shared_metrics.tsv", "manifest.json"],
    "Fig4_final_frozen_20261024": ["Fig4a_species_calls.tsv", "Fig4b_triplet_funnel.tsv",
                                    "Fig4c_directional_pattern_counts.tsv", "Fig4d_recovery_histogram.tsv",
                                    "Fig4d_refit_recovery.tsv", "Fig4e_orthomcl_support_by_triplet.tsv",
                                    "Fig4e_orthomcl_support_summary.tsv", "manifest.json", "README.tsv"],
    "Figure5_context_recomputed_20261023_final": ["Fig5_GO_coverage.tsv", "Fig5_GO_enrichment_all_terms.tsv",
                                                   "Fig5_GO_enrichment_FDR05.tsv", "Fig5_length_matched_group_pairs.tsv",
                                                   "Fig5_pairwise_alignment_identity.tsv",
                                                   "Fig5_sequence_identity_matched_groups.tsv",
                                                   "Fig5_sequence_identity_statistics.tsv",
                                                   "Fig5_stable_group_evidence_matrix.tsv",
                                                   "Fig5_stable_group_GO_annotations.tsv", "S12_length_match_pairs.tsv"],
    "Fig6": ["F6a_loso_performance.tsv", "F6bc_target_enrichment.tsv", "F6d_target_recovery.tsv",
             "target_gene_audit.tsv", "target_registry_coverage.tsv", "manifest.json"],
    "Pk_dual_screen": ["source_gene_pairs.tsv", "manifest.json", "FIGURE_QA_AND_LEGEND.md"],
    "Supplementary_Fig_S1_meta_route_gain_20261024": ["S1_meta_route_gain_and_split_counts.tsv", "manifest.json"],
    "Supplementary_Fig_S2_S5_frozen_20261024": ["S2_essentiality_matched_summary.tsv", "S2_matched_member_audit.tsv",
                                                  "S5_frozen_atlas_target_recovery.tsv", "S5_registry_member_scores.tsv", "manifest.json"],
    "Supplementary_Fig_S3": ["S3_paired_source_composition_fivefold.tsv", "S3_completed_fivefold_loso_audit.json",
                              "S3_figure_working_manifest.json"],
    "Supplementary_Fig_S4_label_uncertainty_20261024": ["S4_fivefold_all_runs.tsv",
                                                         "S4_fivefold_summary_with_baseline.tsv", "manifest.json"],
    "Supplementary_Fig_S6_final_v3_20261023": ["S6_definition_sensitivity_with_184_stable_groups.tsv",
                                                 "S6_permutation_null_10000.tsv", "S6_permutation_summary.tsv", "manifest.json"],
    "Supplementary_Fig_S7": ["S7_no_reused_original_label_predictions.tsv", "S7_species_summary.tsv", "manifest.json"],
    "Supplementary_Fig_S8_S9_final_20261023": ["S8_OrthoMCL_support.tsv", "S9_pattern_by_refit_stability.tsv",
                                                  "S9_Pk_call_state_composition.tsv"],
    "Supplementary_Fig_S10_final_20261023": ["s10_registry_null_summary.tsv", "s10_triplet_eligibility_universe.tsv",
                                               "stable_directional_registry_group_members.tsv",
                                               "stable_directional_registry_manifest.json",
                                               "stable_directional_registry_null_summary.tsv", "manifest.json"],
    "Supplementary_Data_S3_frozen_nine_route_20261024": ["S3_All_15591.tsv", "S3_Call_transition.tsv",
                                                           "S3_Novel_candidates.tsv", "S3_Predicted_essential.tsv",
                                                           "S3_Priority_summary.tsv", "S3_README.tsv",
                                                           "S3_Shared_EEE_members.tsv", "S3_Structure_reuse.tsv",
                                                           "S3_Target_validation.tsv", "manifest.json"],
    ".": ["Source_Data_Supplementary_Fig_S13.tsv"],
}
WORKBOOKS = {
    "05_Supplementary_Data_S1_INTERNAL_WORKING_v10.xlsx": "Supplementary_Data_1_internal_v10.xlsx",
    "05_Supplementary_Data_S2_essentiality_atlas_INTERNAL_v10.xlsx": "Supplementary_Data_2_internal_v10.xlsx",
    "05_Supplementary_Data_S3_post_score_evidence_context_INTERNAL_v11.xlsx": "Supplementary_Data_3_internal_v11.xlsx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the versioned Nature Microbiology release candidate.")
    parser.add_argument("--package-root", required=True, type=Path, help="Internal manuscript package root")
    parser.add_argument("--release-dir", type=Path, help="Candidate release output directory")
    args = parser.parse_args()
    package = args.package_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    output = args.release_dir.resolve() if args.release_dir else repo / "release" / RELEASE_ID
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release path: {output}")

    source_root = package / "Source_Data"
    web = package / "Website_release_candidate_v20261024"
    required = [web / name for name in WEBSITE_FILES] + [web / "release_candidate_manifest.json"]
    for group, names in SOURCE_GROUPS.items():
        folder = package if group == "." else source_root / group
        required.extend(folder / name for name in names)
    s10 = source_root / "Supplementary_Fig_S10_final_20261023"
    required.extend([s10 / "stable_directional_registry_null_draws.tsv", s10 / "independent_audit.json"])
    required.extend(package / name for name in WORKBOOKS)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Candidate preflight found missing inputs:\n" + "\n".join(missing))

    staged: list[Path] = []

    def copy_file(source: Path, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged.append(destination)

    for name in WEBSITE_FILES:
        copy_file(web / name, output / "atlas" / name)
    copy_file(web / "release_candidate_manifest.json", output / "atlas" / "upstream_candidate_manifest.json")

    for group, names in SOURCE_GROUPS.items():
        for name in names:
            folder = package if group == "." else source_root / group
            copy_file(folder / name, output / "source_data" / group.replace(".", "supplementary") / name)
    s10 = source_root / "Supplementary_Fig_S10_final_20261023"
    copy_file(s10 / "stable_directional_registry_null_draws.tsv",
              output / "source_data" / "Supplementary_Fig_S10_final_20261023" / "stable_directional_registry_null_draws.tsv")
    copy_file(s10 / "independent_audit.json",
              output / "source_data" / "Supplementary_Fig_S10_final_20261023" / "independent_audit.json")

    for source_name, target_name in WORKBOOKS.items():
        copy_file(package / source_name, output / "supplementary_workbooks" / target_name)

    readme = """# Nature Microbiology release candidate v20261024

**Internal candidate only. Not published, not submission-ready, and not an archive DOI.** Do not cite this directory as a public release until the corresponding author signs off the values/declarations and the final archive metadata is issued.

## Frozen analysis

The matched-species atlas uses the frozen five-fold nine-route LightGBM, the original 628 raw features, and the original 95%-redundancy-reduced, label-stratified random train/validation/test partitions. No PCA or manual `homology30/50` feature removal was applied. Matched-species test AUROC/AUPRC are Pf 0.821/0.873, Pb 0.874/0.896, and Pk 0.915/0.921. Historical inspection of held Pk outcomes means these are not blind estimates.

The source-only four-species LOSO table includes Pf, Pb, Pk and Tg. It is retrospective, not prospective external validation. Source-composition comparisons, Fig. 2d allocations, and the 60-run S4 label-exchange analysis are sensitivity analyses, not replacement main-test estimates.

The Pk paired-screen analysis includes 3,482 definite-call homolog pairs and 435 discordant labels (12.49%; Wilson interval in the table). The human-red-cell comparison is a retrospective homolog projection, not an A1H sequence-level or blind prediction; the source screens also differ in strain and procedure.

## Contents

- `atlas/`: standardized predictions, four-species LOSO, source-composition comparison, Pk pair-level results, S4 runs, model/data cards and upstream freeze manifest.
- `source_data/`: panel-level source tables for Figs. 2-6 and the final Supplementary Fig. S1-S13 analyses. Only the named final S6 v3 and final frozen Fig. 4 source package are included; intermediate pilot/model files are excluded.
- `supplementary_workbooks/`: current internal S1-S3 workbooks, included for candidate review and still subject to author approval.
- `release_candidate_manifest.json`: status flags and SHA-256 for every payload file.

## Limitations and release gates

All gene calls and target priorities are computational hypotheses, not experimental validation. The historical v45 row-level transition data could not be verified and no transition statistic is reported. Corresponding-author approval, final declaration/reporting-summary review, archive DOI, and production website target verification remain outstanding. `public_release` is false.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(readme, encoding="utf-8")
    staged.append(output / "README.md")

    manifest = {
        "release_id": RELEASE_ID,
        "created_local_date": date.today().isoformat(),
        "status": "offline_internal_candidate_not_published",
        "public_release": False,
        "submission_ready": False,
        "archive_doi": None,
        "production_website_target_verified": False,
        "upstream_model_freeze_manifest_sha256": sha256(web / "final_freeze_manifest.json"),
        "payload_files": {
            p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(staged)
        },
        "limitations": [
            "Author-level sign-off of all values and declarations is outstanding.",
            "Historical v45 row-level transition source data could not be verified.",
            "No final archive DOI has been minted.",
            "The production website domain and deployment target require confirmation.",
            "Historical Pk test-outcome exposure makes reported evaluations retrospective, not blind.",
            "Pk human-red-cell results are homolog-projected and not A1H sequence-level predictions.",
        ],
    }
    (output / "release_candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release_dir": str(output), "payload_files": len(staged),
                      "payload_bytes": sum(p.stat().st_size for p in staged), "public_release": False}, indent=2))


if __name__ == "__main__":
    main()
