"""Rebuild post-score target-prioritisation tables from the leakage-safe atlas.

This is deliberately an annotation-only downstream step.  Essentiality calls,
scores and species-specific thresholds are read from the frozen v46 atlas and
are never changed by human-sequence context, pharmacology or structural data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ATLAS_DIR = Path(r"D:\ProtoEss_full_application_inner_validation_v46\essentiality_atlas")
CONTEXT = Path(r"D:\ProtoEss_Reviewer_Inputs_v46\eukaryote_model_results\reviewer_context")
TARGETS = CONTEXT / "target_registry_members.tsv"
OUT = ATLAS_DIR.parent / "target_prioritisation_leakage_safe_v46"
SPECIES = ("Pf", "Pb", "Pk")


def as_bool(series: pd.Series, *, field: str) -> pd.Series:
    """Parse explicit Boolean values safely, including CSV text values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    result = series.map({
        True: True, False: False, "True": True, "False": False,
        "true": True, "false": False, 1: True, 0: False, "1": True, "0": False,
    })
    if result.isna().any():
        bad = series.loc[result.isna()].drop_duplicates().tolist()[:10]
        raise ValueError(f"Unexpected Boolean values in {field}: {bad}")
    return result.astype(bool)


def priority_class(frame: pd.DataFrame) -> pd.Series:
    """Assign navigation tiers from post-score evidence without redefining calls."""
    essential = frame["full_formal_stack_predicted_essential"]
    strong_margin = frame["full_formal_stack_margin"].abs().ge(0.10)
    lower_host_context = frame["lower_human_risk_v2"]
    intermediate_host_context = frame["intermediate_human_risk_v2"]
    tractable = frame["druggability_evidence"].eq("supported")
    high = essential & strong_margin & lower_host_context & tractable
    medium = essential & ~high & (
        (lower_host_context & strong_margin) |
        (intermediate_host_context & strong_margin & tractable)
    )
    return pd.Series(np.select(
        [high, medium, essential], ["High", "Medium", "Low"], default="Not_predicted_essential"
    ), index=frame.index)


def strict_triplet_members(atlas: pd.DataFrame) -> pd.DataFrame:
    """Return members of exact three-species E/E/E orthogroups for broad follow-up."""
    selected: list[pd.DataFrame] = []
    for group_id, group in atlas.dropna(subset=["homology_group_id"]).groupby("homology_group_id", sort=False):
        if len(group) != 3 or set(group["species_short"]) != set(SPECIES):
            continue
        if group["full_formal_stack_predicted_essential"].all():
            member = group.copy()
            member["three_species_shared_predicted_essential"] = True
            selected.append(member)
    return pd.concat(selected, ignore_index=True) if selected else atlas.iloc[0:0].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(ATLAS_DIR / "leakage_safe_all_15591_gene_atlas.tsv", sep="\t", low_memory=False)
    targets = pd.read_csv(TARGETS, sep="\t", low_memory=False)
    if atlas["record_id"].duplicated().any():
        raise ValueError("Atlas record identifiers must be unique.")

    for field in (
        "full_formal_stack_predicted_essential", "lower_human_risk_v2",
        "intermediate_human_risk_v2", "structure_screened", "legacy_docking_available",
    ):
        atlas[field] = as_bool(atlas[field], field=field)
    atlas["pocket_screen_pass"] = atlas["pocket_screen_pass"].fillna(False).map(
        {True: True, False: False, "True": True, "False": False}
    ).fillna(False).astype(bool)

    target_keys = set(zip(targets["species"].astype(str), targets["gene_id"].astype(str)))
    atlas["known_target_registry"] = [
        (species, gene) in target_keys for species, gene in zip(atlas["species"].astype(str), atlas["gene_id"].astype(str))
    ]
    atlas["target_annotation_used_for_scoring"] = False
    atlas["target_priority"] = priority_class(atlas)
    atlas["essentiality_margin_class"] = np.select(
        [
            atlas.full_formal_stack_predicted_essential & atlas.full_formal_stack_margin.abs().ge(0.10),
            atlas.full_formal_stack_predicted_essential,
        ],
        ["strong_margin", "near_threshold_margin"],
        default="not_predicted_essential",
    )
    atlas["post_score_structure_status"] = np.select(
        [
            atlas.legacy_docking_available,
            atlas.pocket_screen_pass,
            atlas.structure_screened,
        ],
        ["existing_docking_reusable", "pocket_pass_without_saved_docking", "structure_screened_no_pocket_pass"],
        default="not_structure_screened",
    )
    atlas["requires_new_structure_or_docking"] = (
        atlas.full_formal_stack_predicted_essential
        & atlas.target_priority.isin(["High", "Medium"])
        & atlas.post_score_structure_status.eq("not_structure_screened")
    )
    atlas["priority_rule"] = (
        "High: predicted essential + |margin|>=0.10 + lower host-sequence context + tractability annotation; "
        "Medium: remaining strong-margin context-supported predicted essential; "
        "Low: other predicted-essential records."
    )

    predicted = atlas.loc[atlas.full_formal_stack_predicted_essential].copy()
    shared = strict_triplet_members(atlas)
    novel = predicted.loc[~predicted.known_target_registry].copy()

    priority_summary = (
        predicted.groupby(["species_short", "target_priority"], sort=False).size()
        .rename("genes").reset_index()
    )
    validation = (
        atlas.groupby("species_short", sort=False)
        .agg(
            proteome_genes=("gene_id", "size"),
            predicted_essential=("full_formal_stack_predicted_essential", "sum"),
            registry_targets=("known_target_registry", "sum"),
        ).reset_index()
    )
    recovery = (
        predicted.groupby("species_short", sort=False)["known_target_registry"].sum()
        .rename("recovered_registry_targets").reset_index()
    )
    validation = validation.merge(recovery, on="species_short", how="left")
    validation["recovered_registry_targets"] = validation["recovered_registry_targets"].fillna(0).astype(int)
    validation["registry_target_recall"] = validation["recovered_registry_targets"] / validation["registry_targets"]
    transition = pd.read_csv(ATLAS_DIR / "v45_to_leakage_safe_prediction_transitions.tsv", sep="\t")
    structural_reuse = (
        predicted.groupby(["species_short", "target_priority", "post_score_structure_status"], sort=False)
        .size().rename("genes").reset_index()
    )

    atlas.to_csv(OUT / "all_15591_leakage_safe_target_context.tsv", sep="\t", index=False)
    predicted.to_csv(OUT / "all_8735_predicted_essential_target_context.tsv", sep="\t", index=False)
    novel.to_csv(OUT / "novel_predicted_essential_target_context.tsv", sep="\t", index=False)
    for short in SPECIES:
        predicted.loc[predicted.species_short.eq(short)].to_csv(
            OUT / f"{short}_predicted_essential_target_context.tsv", sep="\t", index=False
        )
    for tier in ("High", "Medium", "Low"):
        predicted.loc[predicted.target_priority.eq(tier)].to_csv(
            OUT / f"{tier.lower()}_priority_predicted_essential_targets.tsv", sep="\t", index=False
        )
    shared.to_csv(OUT / "three_species_shared_predicted_essential_members.tsv", sep="\t", index=False)
    priority_summary.to_csv(OUT / "priority_summary.tsv", sep="\t", index=False)
    validation.to_csv(OUT / "known_target_validation_by_species.tsv", sep="\t", index=False)
    transition.to_csv(OUT / "legacy_v45_to_leakage_safe_call_transition.tsv", sep="\t", index=False)
    structural_reuse.to_csv(OUT / "structure_and_docking_reuse_audit.tsv", sep="\t", index=False)

    with pd.ExcelWriter(OUT / "Supplementary_Data_S3_Leakage_Safe_Target_Prioritisation.xlsx", engine="openpyxl") as writer:
        pd.DataFrame({"field": [
            "input_atlas", "genes_scored", "predicted_essential", "known_target_handling", "priority_definition", "structure_and_docking",
        ], "value": [
            "Leakage-safe v46 matched-species atlas", len(atlas), len(predicted),
            "Registry membership is joined only after scoring and never enters model fitting, thresholds or priority class.",
            predicted.priority_rule.iloc[0],
            "Existing structural and docking records are retained as reusable post-score context; newly eligible candidates are flagged rather than assigned an invented docking result.",
        ]}).to_excel(writer, sheet_name="README", index=False)
        atlas.to_excel(writer, sheet_name="All_15591", index=False)
        predicted.to_excel(writer, sheet_name="Predicted_8735", index=False)
        novel.to_excel(writer, sheet_name="Novel_candidates", index=False)
        shared.to_excel(writer, sheet_name="Shared_EEE_members", index=False)
        priority_summary.to_excel(writer, sheet_name="Priority_summary", index=False)
        validation.to_excel(writer, sheet_name="Target_validation", index=False)
        structural_reuse.to_excel(writer, sheet_name="Structure_reuse", index=False)
        transition.to_excel(writer, sheet_name="Call_transition", index=False)

    manifest = {
        "lineage": "leakage-safe v46 matched-species scoring; post-score target-context rebuild",
        "genes_scored": int(len(atlas)),
        "predicted_essential": int(len(predicted)),
        "per_species_predicted_essential": predicted.groupby("species_short").size().to_dict(),
        "priority_counts": predicted.target_priority.value_counts().to_dict(),
        "novel_predicted_essential": int(len(novel)),
        "exact_three_species_shared_members": int(len(shared)),
        "exact_three_species_shared_groups": int(shared.homology_group_id.nunique()),
        "known_target_registry_total": int(len(targets)),
        "known_target_recovered": int(predicted.known_target_registry.sum()),
        "known_target_used_for_scoring_or_priority": False,
        "docking_policy": "Reuse prior structural/docking context only for unchanged record-level annotations; flag newly eligible high/medium candidates for future structure/docking rather than fabricating results.",
    }
    (OUT / "run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
