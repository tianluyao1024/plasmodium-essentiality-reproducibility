"""Orthogroup-level functional interpretation of predicted dependencies.

GO annotations were included among model inputs, so this analysis is
descriptive interpretation rather than independent validation.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from run_plasmodium_priority_go_enrichment import bh_adjust, read_obo


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "protozoa_human_sci_analysis" / "manuscript_v6_protozoan_clade" / "source_data"
ATLAS = SRC / "Complete_atlas_15591_genes.csv"
PRODUCTS = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "host_risk_revised_atlas" / "all_three_plasmodium_genes_host_risk_revised.tsv"
GO_ROOT = ROOT / "downloads" / "go_current"
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "orthogroup_functional_enrichment"

GAF = {
    "Plasmodium_berghei_ANKA": GO_ROOT / "Plasmodium_berghei" / "2828414.P_berghei_ANKA.goa",
    "Plasmodium_falciparum_3D7": GO_ROOT / "Plasmodium_falciparum" / "493.P_falciparum.goa",
    "Plasmodium_knowlesi_H": GO_ROOT / "Plasmodium_knowlesi" / "31342.P_knowlesi.goa",
}
CLASSES = [
    "three_species_conserved_essential",
    "species_restricted_essentiality",
    "two_species_shared_essential",
    "partial_cross_species_support",
]


def read_gaf(path: Path) -> dict[str, set[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    mapping: dict[str, set[str]] = defaultdict(set)
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11 or not parts[4].startswith("GO:"):
                continue
            aliases = {parts[1], parts[2]}
            aliases.update(x for x in re.split(r"[|,;\s]+", parts[10]) if x)
            for alias in aliases:
                mapping[alias].add(parts[4])
                mapping[alias.upper()].add(parts[4])
    return mapping


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(ATLAS, low_memory=False)
    products = pd.read_csv(PRODUCTS, sep="\t", usecols=["gene_id", "product"], low_memory=False).drop_duplicates("gene_id")
    atlas = atlas.merge(products, on="gene_id", how="left")
    atlas["predicted_essential_bool"] = atlas.predicted_essential.astype(str).str.lower().eq("true")
    ontology = read_obo(GO_ROOT / "go-basic.obo")
    mappings = {species: read_gaf(path) for species, path in GAF.items()}
    atlas["go_terms"] = [
        mappings.get(species, {}).get(str(gene), mappings.get(species, {}).get(str(gene).upper(), set()))
        for species, gene in zip(atlas.species, atlas.gene_id)
    ]

    positive = atlas.loc[atlas.predicted_essential_bool].copy()
    group_rows: list[dict[str, object]] = []
    for group, frame in positive.groupby("homology_group_id", dropna=False):
        classes = frame.homology_essentiality_class.value_counts()
        group_class = str(classes.index[0])
        terms = set().union(*frame.go_terms.tolist()) if len(frame) else set()
        descriptions = [
            str(value) for value in frame["product"].dropna()
            if not re.search(r"hypothetical|unknown|uncharacteri[sz]ed", str(value), flags=re.I)
        ]
        group_rows.append({
            "homology_group_id": group,
            "homology_essentiality_class": group_class,
            "predicted_essential_records": int(len(frame)),
            "species_with_predicted_essentiality": int(frame.species.nunique()),
            "median_within_score": float(frame.within_score.median()),
            "median_absolute_margin": float(frame.within_margin.abs().median()),
            "GO_terms": ";".join(sorted(terms)),
            "GO_term_count": int(len(terms)),
            "representative_product": descriptions[0] if descriptions else "uncharacterized protein",
        })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUT / "orthogroup_GO_annotation_table.tsv", sep="\t", index=False)

    term_to_groups: dict[str, set[str]] = defaultdict(set)
    for row in groups.itertuples(index=False):
        for term in str(row.GO_terms).split(";"):
            if term.startswith("GO:"):
                term_to_groups[term].add(str(row.homology_group_id))
    background = set(groups.homology_group_id.astype(str))
    enrichment_rows: list[dict[str, object]] = []
    for group_class in CLASSES:
        selected = set(groups.loc[groups.homology_essentiality_class.eq(group_class), "homology_group_id"].astype(str))
        tested: list[dict[str, object]] = []
        for term, annotated_groups in term_to_groups.items():
            bg_hits = len(background & annotated_groups)
            selected_hits = len(selected & annotated_groups)
            if bg_hits < 10 or selected_hits < 3:
                continue
            a = selected_hits
            b = len(selected) - selected_hits
            c = bg_hits - selected_hits
            d = len(background) - len(selected) - c
            odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            name, namespace = ontology.get(term, (term, "unknown"))
            tested.append({
                "homology_essentiality_class": group_class,
                "GO_ID": term,
                "GO_name": name,
                "namespace": namespace,
                "selected_hits": a,
                "selected_total": len(selected),
                "background_hits": bg_hits,
                "background_total": len(background),
                "fold_enrichment": (a / len(selected)) / (bg_hits / len(background)),
                "odds_ratio": float(odds),
                "fisher_p_greater": float(pvalue),
            })
        if tested:
            adjusted = bh_adjust([row["fisher_p_greater"] for row in tested])
            for row, fdr in zip(tested, adjusted):
                row["FDR_BH"] = float(fdr)
            enrichment_rows.extend(tested)
    enrichment = pd.DataFrame(enrichment_rows).sort_values(
        ["homology_essentiality_class", "FDR_BH", "fold_enrichment"],
        ascending=[True, True, False],
    )
    enrichment.to_csv(OUT / "orthogroup_GO_enrichment_all_terms.tsv", sep="\t", index=False)
    significant = enrichment.loc[enrichment.FDR_BH.le(0.05)].copy()
    significant.to_csv(OUT / "orthogroup_GO_enrichment_FDR05.tsv", sep="\t", index=False)

    coverage = (
        groups.assign(annotated=groups.GO_term_count.gt(0))
        .groupby("homology_essentiality_class")
        .agg(orthogroups=("homology_group_id", "size"), GO_annotated=("annotated", "sum"))
        .reset_index()
    )
    coverage["GO_coverage"] = coverage.GO_annotated / coverage.orthogroups
    coverage.to_csv(OUT / "orthogroup_GO_annotation_coverage.tsv", sep="\t", index=False)

    # Continuous-score audit separates robust discordance from threshold-near calls.
    multi = atlas.loc[atlas.homology_group_id.notna()].copy()
    continuous_rows: list[dict[str, object]] = []
    for group, frame in multi.groupby("homology_group_id"):
        if frame.species.nunique() < 2:
            continue
        call_count = int(frame.predicted_essential_bool.sum())
        discordant = 0 < call_count < frame.species.nunique()
        min_abs_margin = float(frame.within_margin.abs().min())
        continuous_rows.append({
            "homology_group_id": group,
            "species_records": int(frame.species.nunique()),
            "predicted_essential_species": call_count,
            "score_min": float(frame.within_score.min()),
            "score_max": float(frame.within_score.max()),
            "score_range": float(frame.within_score.max() - frame.within_score.min()),
            "minimum_absolute_margin": min_abs_margin,
            "discordant_hard_calls": discordant,
            "discordance_strength": (
                "strong_discordance" if discordant and min_abs_margin >= 0.10
                else "boundary_sensitive" if discordant
                else "concordant"
            ),
        })
    continuous = pd.DataFrame(continuous_rows)
    continuous.to_csv(OUT / "orthogroup_continuous_score_audit.tsv", sep="\t", index=False)
    continuous.groupby("discordance_strength").size().rename("orthogroups").reset_index().to_csv(
        OUT / "orthogroup_continuous_score_summary.tsv", sep="\t", index=False
    )

    # Representative groups require interpretable products and robust score margins.
    sig_terms = significant.groupby("homology_essentiality_class").GO_ID.apply(set).to_dict()
    representatives = groups.loc[
        groups.homology_essentiality_class.isin(["species_restricted_essentiality", "partial_cross_species_support"])
        & groups.representative_product.ne("uncharacterized protein")
        & groups.median_absolute_margin.ge(0.10)
    ].copy()
    representatives["significant_GO_terms"] = representatives.apply(
        lambda row: ";".join(sorted(set(str(row.GO_terms).split(";")) & sig_terms.get(row.homology_essentiality_class, set()))),
        axis=1,
    )
    representatives = representatives.loc[representatives.significant_GO_terms.ne("")].sort_values(
        ["homology_essentiality_class", "median_absolute_margin", "median_within_score"], ascending=[True, False, False]
    )
    representatives.groupby("homology_essentiality_class", group_keys=False).head(5).to_csv(
        OUT / "representative_divergent_orthogroups.tsv", sep="\t", index=False
    )
    (OUT / "README.md").write_text(
        "GO enrichment uses predicted-essential orthogroups as the background and one-sided Fisher tests with "
        "Benjamini-Hochberg correction within each orthogroup class. GO annotations were model inputs; results "
        "are descriptive functional interpretation, not independent validation. Discordant hard calls are "
        "classified as strong only when every member lies at least 0.10 from its species-specific threshold.\n",
        encoding="utf-8",
    )
    print(coverage.to_string(index=False))
    print("significant_terms", len(significant))
    print(continuous.groupby("discordance_strength").size().to_string())


if __name__ == "__main__":
    main()
