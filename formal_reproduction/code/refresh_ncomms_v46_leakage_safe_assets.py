"""Overlay the submission-facing source data with leakage-safe v46 outputs.

This script is intentionally run only after the complete matched-species
application, annotation-exclusion comparison and repeated directional refits
have finished.  It does not train or rescore a model; it copies the immutable
result lineage into the submission package and records the values used by the
manuscript builder.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get(
    "PROTOESS_SUBMISSION_OUT",
    ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v46_reproducible",
))
FORMAL = Path(r"D:\ProtoEss_formal_inner_validation_v46")
APPLICATION = Path(r"D:\ProtoEss_full_application_inner_validation_v46")
DIRECTIONAL = Path(r"D:\ProtoEss_directional_audits_inner_validation_v46")
REFITS = Path(r"D:\ProtoEss_directional_refit_inner_validation_v46")
LOSO = Path(r"D:\ProtoEss_loso_inner_validation_v46")
TG_SHARED = Path(r"D:\ProtoEss_tg_shared_control_inner_validation_v46")
SHORT = {
    "Plasmodium_falciparum_3D7": "Pf",
    "Plasmodium_berghei_ANKA": "Pb",
    "Plasmodium_knowlesi_H": "Pk",
}
METRIC_DIRS = {
    "Pf": "Plasmodium_falciparum",
    "Pb": "Plasmodium_berghei",
    "Pk": "Plasmodium_knowlesi",
}


def required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required leakage-safe result is missing: {path}")
    return path


def main() -> None:
    source = required(OUT / "Source_Data")
    atlas_dir = required(APPLICATION / "essentiality_atlas")
    atlas_summary = json.loads(required(atlas_dir / "summary.json").read_text(encoding="utf-8"))
    retention = json.loads(required(DIRECTIONAL / "strong_directional_retention.json").read_text(encoding="utf-8"))
    refit_summary = json.loads(required(REFITS / "summary.json").read_text(encoding="utf-8"))

    scores = pd.read_csv(required(APPLICATION / "full_formal_stack_genomewide_all_three.tsv"), sep="\t", low_memory=False)
    scores["species_short"] = scores["species"].map(SHORT)
    scores["score_provenance"] = "leakage-safe five-branch application score; Transformer epoch selection used training-internal validation only"
    scores["formal_model_architecture"] = "LightGBM + CatBoost + source-specific expert + prevalence + Transformer -> L2 logistic meta-learner"
    scores.to_csv(source / "Fig4a_atlas_scores_with_provenance.tsv", sep="\t", index=False)

    stable = int(refit_summary["groups_recovered_at_least_80pct"])
    flow = pd.DataFrame([
        ("Exact 1:1:1 orthogroups", int(atlas_summary["exact_1to1to1_orthogroups"])),
        ("Concordant predicted-essential E/E/E triplets", int(atlas_summary["concordant_E_E_E_orthogroups"])),
        ("Strong directional, full model", int(retention["full_strong_directional_groups"])),
        ("Retained after annotation exclusion with same direction", int(retention["full_strong_direction_retained"])),
        ("Recovered in at least 80% of paired refits", stable),
    ], columns=["stage", "orthogroups"])
    flow.to_csv(source / "Fig4b_directional_filtering_flow.tsv", sep="\t", index=False)

    patterns = pd.read_csv(required(DIRECTIONAL / "single_copy_1to1to1_directional_patterns.tsv"), sep="\t", low_memory=False)
    refit = pd.read_csv(required(REFITS / "directional_refit_stability.tsv"), sep="\t", low_memory=False)
    patterns.merge(refit, on="homology_group_id", how="left", validate="one_to_one").to_csv(
        source / "Fig4c_directional_patterns.tsv", sep="\t", index=False
    )
    refit.to_csv(source / "Fig4d_directional_refit_stability.tsv", sep="\t", index=False)

    recovery = pd.read_csv(required(atlas_dir / "leakage_safe_target_registry_recovery.tsv"), sep="\t")
    recovery = recovery.rename(columns={"recovered_predicted_essential": "recovered_by_canonical"})
    recovery["recovery_fraction"] = recovery["recovered_by_canonical"] / recovery["target_supported_genes"]
    recovery.to_csv(source / "Supplementary_FigS5a_target_recovery.tsv", sep="\t", index=False)

    # Fig. 3 is rebuilt entirely from the corrected, target-label-naive lineage.
    shutil.copy2(
        required(LOSO / "four_species_loso_bootstrap.tsv"),
        source / "Fig3a_four_species_LOSO.tsv",
    )
    shutil.copy2(
        required(TG_SHARED / "shared_molecular_tg_augmentation_metrics.tsv"),
        source / "Fig3b_shared_molecular_Tg_source_metrics.tsv",
    )
    shutil.copy2(
        required(TG_SHARED / "shared_molecular_tg_augmentation_paired_bootstrap.tsv"),
        source / "Fig3b_shared_molecular_Tg_source_bootstrap.tsv",
    )

    metrics: dict[str, dict[str, float]] = {}
    for short, directory in METRIC_DIRS.items():
        payload = json.loads(required(FORMAL / "within_species" / directory / "metrics.json").read_text(encoding="utf-8"))
        metrics[short] = payload["test"]
    values = {
        "lineage": "inner-validation leakage-safe five-branch rerun",
        "matched_test_metrics": metrics,
        "atlas": atlas_summary,
        "directional_retention": retention,
        "directional_refits": refit_summary,
        "target_registry_recovery": recovery.to_dict(orient="records"),
    }
    (OUT / "leakage_safe_result_summary.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    (source / "Fig1_study_design_summary.tsv").write_text(
        "stage\tvalue\n"
        f"Malaria-atlas genes\t{atlas_summary['genes_scored']}\n"
        f"Predicted-essential calls\t{atlas_summary['predicted_essential_total']}\n"
        f"Exact Pf-Pb-Pk triplets\t{atlas_summary['exact_1to1to1_orthogroups']}\n"
        f"Concordant predicted-essential E/E/E triplets\t{atlas_summary['concordant_E_E_E_orthogroups']}\n"
        f"Refit-stable directional triplets\t{stable}\n",
        encoding="utf-8",
    )
    shutil.copy2(
        required(atlas_dir / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx"),
        OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx",
    )
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
