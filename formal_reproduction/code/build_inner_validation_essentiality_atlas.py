"""Build a leakage-safe genome-scale malaria atlas from the v46 application scores.

The script deliberately creates a new lineage instead of overwriting the
historical 9,142-call atlas.  Static biological context is re-linked by stable
record identifiers; it is never used to alter the newly generated calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REV = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision"
APPLICATION = Path(os.environ.get("NCOMMS_FULL_APPLICATION_OUT", r"D:\ProtoEss_full_application_inner_validation_v46"))
CONTEXT = Path(os.environ.get(
    "PROTOESS_REVIEW_CONTEXT_ROOT",
    str(REV / "reviewer_context"),
))
HOST = Path(os.environ.get(
    "PROTOESS_HOST_CONTEXT_TABLE",
    str(CONTEXT / "all_three_plasmodium_genes_host_risk_revised.tsv"),
))
TARGETS = Path(os.environ.get(
    "PROTOESS_TARGET_REGISTRY_TABLE",
    str(CONTEXT / "target_registry_members.tsv"),
))
LEGACY_SCORES = Path(os.environ.get(
    "PROTOESS_LEGACY_CANONICAL_SCORES",
    str(REV / "ncomms_submission_v44" / "Source_Data" / "Fig5a_canonical_genomewide_scores_with_provenance.tsv"),
))
OUT = APPLICATION / "essentiality_atlas"
ORDER = ("Plasmodium_falciparum_3D7", "Plasmodium_berghei_ANKA", "Plasmodium_knowlesi_H")
SHORT = dict(zip(ORDER, ("Pf", "Pb", "Pk")))


def as_bool(series: pd.Series, *, name: str) -> pd.Series:
    """Parse explicit Boolean values without treating the text ``False`` as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.map({
        True: True, False: False,
        "True": True, "False": False,
        "true": True, "false": False,
        "1": True, "0": False,
        1: True, 0: False,
    })
    if mapped.isna().any():
        values = series.loc[mapped.isna()].drop_duplicates().tolist()[:10]
        raise ValueError(f"Unrecognised Boolean value(s) in {name}: {values}")
    return mapped.astype(bool)


def exact_triplets(atlas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, frame in atlas.dropna(subset=["homology_group_id"]).groupby("homology_group_id", sort=False):
        if len(frame) != 3 or set(frame.species) != set(ORDER) or frame.species.duplicated().any():
            continue
        frame = frame.set_index("species").loc[list(ORDER)].reset_index()
        call = ["E" if value else "NE" for value in frame.full_formal_stack_predicted_essential]
        margin = frame.full_formal_stack_margin.astype(float).tolist()
        rows.append({
            "homology_group_id": group_id,
            "Pf_gene": frame.loc[0, "gene_id"],
            "Pb_gene": frame.loc[1, "gene_id"],
            "Pk_gene": frame.loc[2, "gene_id"],
            "pattern": "/".join(call),
            "predicted_essential_members": int(sum(value == "E" for value in call)),
            "min_abs_margin": min(abs(value) for value in margin),
            "mean_abs_margin": sum(abs(value) for value in margin) / 3,
            "directional": len(set(call)) > 1,
            "strong_directional": len(set(call)) > 1 and min(abs(value) for value in margin) >= 0.10,
            "Pf_margin": margin[0], "Pb_margin": margin[1], "Pk_margin": margin[2],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(APPLICATION / "full_formal_stack_genomewide_all_three.tsv", sep="\t", low_memory=False)
    if scores.record_id.duplicated().any():
        raise ValueError("Genome-scale application scores must have unique record identifiers.")
    atlas = scores.copy()
    atlas["species_short"] = atlas.species.map(SHORT)
    atlas["score_provenance"] = "leakage-safe five-branch application score; Transformer epoch selection used training-internal validation only"
    atlas["formal_model_architecture"] = "LightGBM + CatBoost + source-specific expert + prevalence + Transformer -> L2 logistic meta-learner"

    host = pd.read_csv(HOST, sep="\t", low_memory=False)
    host_columns = [
        "record_id", "homology_group_id", "product", "lower_human_risk_v2", "intermediate_human_risk_v2",
        "druggability_evidence", "structure_screened", "pocket_screen_pass", "legacy_docking_available",
        "human_best_gene_id", "human_pident", "human_qcov", "human_tcov", "LOEUF",
    ]
    host = host[[column for column in host_columns if column in host.columns]].drop_duplicates("record_id")
    atlas = atlas.merge(host, on="record_id", how="left", validate="one_to_one", suffixes=("", "_host"))
    if "product_host" in atlas:
        atlas["product"] = atlas["product"].fillna(atlas["product_host"])
        atlas = atlas.drop(columns=["product_host"])

    targets = pd.read_csv(TARGETS, sep="\t", low_memory=False)
    target_index = set(zip(targets.species.astype(str), targets.gene_id.astype(str)))
    atlas["target_supported_registry"] = [
        (species, gene) in target_index for species, gene in zip(atlas.species.astype(str), atlas.gene_id.astype(str))
    ]
    atlas["target_annotation_used_for_scoring"] = False

    atlas["full_formal_stack_predicted_essential"] = as_bool(
        atlas["full_formal_stack_predicted_essential"], name="full_formal_stack_predicted_essential"
    )
    candidates = atlas.loc[atlas.full_formal_stack_predicted_essential].copy()
    triplets = exact_triplets(atlas)
    recovery = (
        atlas.loc[atlas.target_supported_registry]
        .groupby("species_short", sort=False)
        .agg(target_supported_genes=("gene_id", "size"), recovered_predicted_essential=("full_formal_stack_predicted_essential", "sum"))
        .reset_index()
    )

    transition: pd.DataFrame | None = None
    if LEGACY_SCORES.exists():
        legacy = pd.read_csv(LEGACY_SCORES, sep="\t", low_memory=False)
        legacy = legacy[["record_id", "species_short", "full_formal_stack_predicted_essential"]].copy()
        legacy["legacy_predicted_essential"] = as_bool(
            legacy.pop("full_formal_stack_predicted_essential"),
            name="legacy full_formal_stack_predicted_essential",
        )
        transition = legacy.merge(
            atlas[["record_id", "species_short", "full_formal_stack_predicted_essential"]],
            on=["record_id", "species_short"], how="inner", validate="one_to_one",
        )
        if len(transition) != len(atlas):
            raise ValueError("Legacy-to-leakage-safe transition audit lost records.")
        transition["prediction_transition_from_v45"] = "unchanged_nonessential"
        transition.loc[
            transition.legacy_predicted_essential & transition.full_formal_stack_predicted_essential,
            "prediction_transition_from_v45",
        ] = "retained_predicted_essential"
        transition.loc[
            transition.legacy_predicted_essential & ~transition.full_formal_stack_predicted_essential,
            "prediction_transition_from_v45",
        ] = "no_longer_predicted_essential"
        transition.loc[
            ~transition.legacy_predicted_essential & transition.full_formal_stack_predicted_essential,
            "prediction_transition_from_v45",
        ] = "newly_predicted_essential"
        transition = (
            transition.groupby(["species_short", "prediction_transition_from_v45"], sort=False)
            .size().rename("genes").reset_index()
        )
    summary = {
        "genes_scored": int(len(atlas)),
        "predicted_essential_total": int(len(candidates)),
        "per_species_predicted_essential": candidates.groupby("species_short").size().to_dict(),
        "exact_1to1to1_orthogroups": int(len(triplets)),
        "concordant_E_E_E_orthogroups": int(triplets.pattern.eq("E/E/E").sum()),
        "strong_directional_before_annotation_exclusion": int(triplets.strong_directional.sum()),
        "targets_recovered": int(candidates.target_supported_registry.sum()),
        "targets_in_registry": int(len(targets)),
    }
    atlas.to_csv(OUT / "leakage_safe_all_15591_gene_atlas.tsv", sep="\t", index=False)
    candidates.to_csv(OUT / "leakage_safe_predicted_essential_atlas.tsv", sep="\t", index=False)
    for short in ("Pf", "Pb", "Pk"):
        candidates.loc[candidates.species_short.eq(short)].to_csv(OUT / f"leakage_safe_{short}_predicted_essential_atlas.tsv", sep="\t", index=False)
    triplets.to_csv(OUT / "leakage_safe_exact_triplet_patterns.tsv", sep="\t", index=False)
    recovery.to_csv(OUT / "leakage_safe_target_registry_recovery.tsv", sep="\t", index=False)
    if transition is not None:
        transition.to_csv(OUT / "v45_to_leakage_safe_prediction_transitions.tsv", sep="\t", index=False)
    with pd.ExcelWriter(OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx", engine="openpyxl") as writer:
        pd.DataFrame({"field": list(summary), "value": list(summary.values())}).to_excel(writer, sheet_name="README", index=False)
        atlas.to_excel(writer, sheet_name="All_15591", index=False)
        candidates.to_excel(writer, sheet_name="Predicted_essential", index=False)
        triplets.to_excel(writer, sheet_name="Exact_triplets", index=False)
        recovery.to_excel(writer, sheet_name="Target_recovery", index=False)
        if transition is not None:
            transition.to_excel(writer, sheet_name="v45_to_v46_transitions", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
