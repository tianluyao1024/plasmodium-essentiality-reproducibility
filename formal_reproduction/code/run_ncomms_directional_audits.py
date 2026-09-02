"""Reconcile full and functional-annotation-excluded malaria dependency maps.

Primary directional inference is restricted to 1:1:1 orthogroups with one
member from each malaria species.  Complex orthogroups are retained as a
secondary sensitivity dataset instead of being silently converted into an
arbitrary representative-gene comparison.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from run_plasmodium_priority_go_enrichment import bh_adjust, read_obo
from run_orthogroup_functional_enrichment import GAF, GO_ROOT, read_gaf


ROOT = Path(__file__).resolve().parent
CONTEXT = Path(os.environ.get(
    "PROTOESS_REVIEW_CONTEXT_ROOT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "reviewer_context"),
))
FREEZE_ROOT = Path(os.environ.get(
    "NCOMMS_DOWNSTREAM_FREEZE_ROOT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_annotation_excluded"),
))
FULL = Path(os.environ.get("NCOMMS_FULL_GENOMEWIDE_TABLE", str(FREEZE_ROOT / "full_stack_application_baseline" / "full_formal_stack_genomewide_all_three.tsv")))
EXCLUDED = Path(os.environ.get("NCOMMS_EXCLUDED_GENOMEWIDE_TABLE", str(FREEZE_ROOT / "functional_annotation_excluded_genomewide_all_three.tsv")))
ATLAS = Path(os.environ.get(
    "PROTOESS_ORTHOGROUP_CONTEXT_TABLE",
    str(CONTEXT / "orthogroup_membership.tsv"),
))
PRODUCTS = Path(os.environ.get(
    "PROTOESS_HOST_CONTEXT_TABLE",
    str(CONTEXT / "all_three_plasmodium_genes_host_risk_revised.tsv"),
))
OUT = Path(os.environ.get("NCOMMS_DIRECTIONAL_OUT", str(FREEZE_ROOT / "directional_orthogroup_audits")))
STATE = OUT / "state.json"
ORDER = ["Plasmodium_falciparum_3D7", "Plasmodium_berghei_ANKA", "Plasmodium_knowlesi_H"]
SHORT = {"Plasmodium_falciparum_3D7": "Pf", "Plasmodium_berghei_ANKA": "Pb", "Plasmodium_knowlesi_H": "Pk"}


def write_state(**updates: object) -> None:
    """Persist resumable audit state for the generic watchdog."""
    OUT.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    if STATE.exists():
        try:
            current = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    if updates.get("status") == "running":
        current.pop("error", None)
        current.pop("traceback", None)
    current.update(updates)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def call_pattern(frame: pd.DataFrame, prefix: str) -> str:
    calls = []
    for species in ORDER:
        row = frame.loc[frame.species.eq(species)].iloc[0]
        calls.append("E" if bool(row[f"{prefix}_predicted_essential"]) else "NE")
    return "/".join(calls)


def margins(frame: pd.DataFrame, prefix: str) -> list[float]:
    return [float(frame.loc[frame.species.eq(species), f"{prefix}_margin"].iloc[0]) for species in ORDER]


def group_table(table: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, frame in table.groupby("homology_group_id", dropna=False):
        counts = frame.species.value_counts()
        represented = set(frame.species)
        complete = represented == set(ORDER)
        single_copy = complete and len(frame) == 3 and bool((counts == 1).all())
        if not complete:
            continue
        if mode == "single_copy" and not single_copy:
            continue
        if mode == "complex" and single_copy:
            continue
        # Complex groups are emitted separately, with no directional claim.
        if not single_copy:
            rows.append({
                "homology_group_id": group_id, "orthology_class": "complex_three_species", "member_count": int(len(frame)),
                "full_pattern": "complex", "excluded_pattern": "complex", "full_min_abs_margin": np.nan,
                "excluded_min_abs_margin": np.nan, "strong_full": False, "strong_excluded": False,
            })
            continue
        full_pattern = call_pattern(frame, "full_formal_stack")
        excluded_pattern = call_pattern(frame, "functional_annotation_excluded")
        full_margin = margins(frame, "full_formal_stack")
        excluded_margin = margins(frame, "functional_annotation_excluded")
        divergent_full = len(set(full_pattern.split("/"))) > 1
        divergent_excluded = len(set(excluded_pattern.split("/"))) > 1
        row = {
            "homology_group_id": group_id,
            "orthology_class": "single_copy_1to1to1",
            "member_count": 3,
            "full_pattern": full_pattern,
            "excluded_pattern": excluded_pattern,
            "pattern_retained": full_pattern == excluded_pattern,
            "full_min_abs_margin": float(np.min(np.abs(full_margin))),
            "excluded_min_abs_margin": float(np.min(np.abs(excluded_margin))),
            "full_mean_abs_margin": float(np.mean(np.abs(full_margin))),
            "excluded_mean_abs_margin": float(np.mean(np.abs(excluded_margin))),
            "strong_full": bool(divergent_full and np.min(np.abs(full_margin)) >= 0.10),
            "strong_excluded": bool(divergent_excluded and np.min(np.abs(excluded_margin)) >= 0.10),
            "full_divergent": divergent_full,
            "excluded_divergent": divergent_excluded,
        }
        for species, full_value, excluded_value in zip(ORDER, full_margin, excluded_margin):
            row[f"{SHORT[species]}_full_margin"] = full_value
            row[f"{SHORT[species]}_excluded_margin"] = excluded_value
        rows.append(row)
    return pd.DataFrame(rows)


def add_go_terms(table: pd.DataFrame) -> pd.DataFrame:
    mapping = {species: read_gaf(path) for species, path in GAF.items()}
    collected: dict[str, set[str]] = defaultdict(set)
    for row in table.itertuples(index=False):
        hits = mapping.get(row.species, {})
        collected[str(row.homology_group_id)].update(hits.get(str(row.gene_id), hits.get(str(row.gene_id).upper(), set())))
    output = table.drop_duplicates("homology_group_id")[["homology_group_id"]].copy()
    output["GO_terms"] = output.homology_group_id.astype(str).map(lambda key: ";".join(sorted(collected[key])))
    output["GO_term_count"] = output.GO_terms.map(lambda value: 0 if not value else len(value.split(";")))
    return output


def enrich(groups: pd.DataFrame, annotation: pd.DataFrame, label: str) -> pd.DataFrame:
    ontology = read_obo(GO_ROOT / "go-basic.obo")
    merged = groups.merge(annotation, on="homology_group_id", how="left")
    background = set(merged.homology_group_id.astype(str))
    selected = set(merged.loc[merged[label].astype(bool), "homology_group_id"].astype(str))
    term_map: dict[str, set[str]] = defaultdict(set)
    for row in merged.itertuples(index=False):
        for term in str(row.GO_terms).split(";"):
            if term.startswith("GO:"):
                term_map[term].add(str(row.homology_group_id))
    rows: list[dict[str, object]] = []
    for term, term_groups in term_map.items():
        selected_hits = len(selected & term_groups)
        background_hits = len(background & term_groups)
        if selected_hits < 2 or background_hits < 5:
            continue
        a = selected_hits
        b = len(selected) - selected_hits
        c = background_hits - selected_hits
        d = len(background) - len(selected) - c
        odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        name, namespace = ontology.get(term, (term, "unknown"))
        rows.append({
            "analysis": label, "GO_ID": term, "GO_name": name, "namespace": namespace,
            "selected_hits": a, "selected_total": len(selected), "background_hits": background_hits,
            "background_total": len(background),
            "fold_enrichment": (a / len(selected)) / (background_hits / len(background)),
            "odds_ratio": float(odds), "fisher_p_greater": float(pvalue),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["FDR_BH"] = bh_adjust(result.fisher_p_greater.tolist())
        result["GO_term_rank"] = result.fisher_p_greater.rank(method="min").astype(int)
        result = result.sort_values(["FDR_BH", "fold_enrichment"], ascending=[True, False])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_state(status="running", stage="load_inputs", pid=os.getpid())
    try:
        if not FULL.exists() or not EXCLUDED.exists():
            raise FileNotFoundError("Both full-stack and functional-annotation-excluded whole-genome tables are required")
        full = pd.read_csv(FULL, sep="\t", low_memory=False)
        excluded = pd.read_csv(EXCLUDED, sep="\t", low_memory=False)
        atlas = pd.read_csv(ATLAS, sep="\t", low_memory=False)
        products = pd.read_csv(PRODUCTS, sep="\t", low_memory=False, usecols=["species", "gene_id", "product"]).drop_duplicates(["species", "gene_id"])
        merged = full.merge(excluded, on=["record_id", "species", "gene_id", "protein_id", "protein_length"], validate="one_to_one")
        merged = merged.merge(atlas[["species", "gene_id", "homology_group_id"]], on=["species", "gene_id"], how="left", validate="one_to_one")
        merged = merged.merge(products, on=["species", "gene_id"], how="left", validate="one_to_one")
        merged.to_csv(OUT / "gene_level_full_vs_annotation_excluded.tsv", sep="\t", index=False)
        write_state(status="running", stage="genomewide_stability", pid=os.getpid())
        stability_rows = []
        for species, frame in merged.groupby("species"):
            full_call = frame.full_formal_stack_predicted_essential.astype(bool)
            excluded_call = frame.functional_annotation_excluded_predicted_essential.astype(bool)
            stability_rows.append({
                "species": species,
                "genes": int(len(frame)),
                "spearman_score": float(frame.full_formal_stack_score.corr(frame.functional_annotation_excluded_score, method="spearman")),
                "classification_agreement": float(np.mean(full_call == excluded_call)),
                "threshold_crossing_rate": float(np.mean(full_call != excluded_call)),
                "predicted_essential_jaccard": float((full_call & excluded_call).sum() / max(1, (full_call | excluded_call).sum())),
            })
        pd.DataFrame(stability_rows).to_csv(OUT / "genomewide_score_and_call_stability.tsv", sep="\t", index=False)
        write_state(status="running", stage="orthogroup_patterns", pid=os.getpid())
        strict = group_table(merged, "single_copy")
        complex_groups = group_table(merged, "complex")
        strict.to_csv(OUT / "single_copy_1to1to1_directional_patterns.tsv", sep="\t", index=False)
        complex_groups.to_csv(OUT / "complex_three_species_orthogroup_inventory.tsv", sep="\t", index=False)
        summary = strict.groupby(["full_pattern", "excluded_pattern", "strong_full", "strong_excluded", "pattern_retained"], dropna=False).size().rename("orthogroups").reset_index()
        summary.to_csv(OUT / "directional_pattern_transition_summary.tsv", sep="\t", index=False)
        retention = {
            "single_copy_1to1to1_groups": int(len(strict)),
            "full_strong_directional_groups": int(strict.strong_full.sum()),
            "excluded_strong_directional_groups": int(strict.strong_excluded.sum()),
            "full_strong_retained_as_excluded_strong": int((strict.strong_full & strict.strong_excluded).sum()),
            "full_strong_direction_retained": int((strict.strong_full & strict.strong_excluded & strict.pattern_retained).sum()),
            "complex_three_species_groups": int(len(complex_groups)),
        }
        (OUT / "strong_directional_retention.json").write_text(json.dumps(retention, indent=2), encoding="utf-8")
        write_state(status="running", stage="posthoc_go", pid=os.getpid())
        annotations = add_go_terms(merged)
        annotations.to_csv(OUT / "single_copy_orthogroup_GO_annotation.tsv", sep="\t", index=False)
        strict["full_strong_pattern_retained"] = strict.strong_full & strict.strong_excluded & strict.pattern_retained
        enrichment_full = enrich(strict, annotations, "strong_full")
        enrichment_stable = enrich(strict, annotations, "full_strong_pattern_retained")
        pd.concat([enrichment_full, enrichment_stable], ignore_index=True).to_csv(OUT / "directional_GO_enrichment_full_and_retained.tsv", sep="\t", index=False)
        representatives = merged.merge(strict.loc[strict.full_strong_pattern_retained, ["homology_group_id", "full_pattern", "full_min_abs_margin", "excluded_min_abs_margin"]], on="homology_group_id", how="inner")
        representatives = representatives.loc[
            ~representatives["product"].fillna("").str.contains(
                r"hypothetical|uncharacter|unknown", case=False, regex=True
            )
        ].copy()
        representatives["minimum_robust_margin"] = representatives[["full_min_abs_margin", "excluded_min_abs_margin"]].min(axis=1)
        selected_groups = representatives.groupby("full_pattern", group_keys=False).apply(lambda frame: frame.sort_values(["minimum_robust_margin", "full_formal_stack_score"], ascending=False).groupby("homology_group_id").head(1).head(6), include_groups=False)
        selected_groups.to_csv(OUT / "predefined_representative_stable_directional_orthogroups.tsv", sep="\t", index=False)
        (OUT / "README.md").write_text(
            "Primary directional analysis is restricted to 1:1:1 orthogroups with exactly one member per malaria species. "
            "A strong directional pattern requires divergent calls and an absolute threshold margin of at least 0.10 for every member. "
            "GO/Pfam/InterPro were excluded from the sensitivity model but GO is used post hoc here for descriptive enrichment. "
            "Persistence argues against direct functional-annotation circularity and is not experimental validation.\n",
            encoding="utf-8",
        )
        (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        write_state(status="complete", stage="complete", pid=os.getpid())
        print(json.dumps(retention, indent=2))
    except Exception as error:
        trace = traceback.format_exc()
        (OUT / "last_error.log").write_text(trace, encoding="utf-8")
        write_state(status="failed", stage="exception", error=str(error), traceback=trace, pid=os.getpid())
        raise


if __name__ == "__main__":
    main()
