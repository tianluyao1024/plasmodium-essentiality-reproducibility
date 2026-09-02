"""GO enrichment for revised Plasmodium target-priority classes.

The analysis is deliberately descriptive: GO annotations are part of the model
feature set, so enrichment must not be presented as independent validation.
"""
from __future__ import annotations

import gzip
import math
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent
ATLAS = (
    ROOT
    / "protozoa_human_sci_analysis"
    / "reviewer_revision"
    / "host_risk_revised_atlas"
    / "all_predicted_essential_genes_host_risk_revised.tsv"
)
GO_ROOT = ROOT / "downloads" / "go_current"
OUT = (
    ROOT
    / "protozoa_human_sci_analysis"
    / "reviewer_revision"
    / "functional_enrichment"
)
OBO = GO_ROOT / "go-basic.obo"

SPECIES = {
    "Pberg": ("Plasmodium_berghei", "Plasmodium berghei ANKA"),
    "Pfal": ("Plasmodium_falciparum", "Plasmodium falciparum 3D7"),
    "Pknow": ("Plasmodium_knowlesi", "Plasmodium knowlesi H"),
}


def bh_adjust(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def read_obo(path: Path) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    current: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                if current.get("id") and current.get("name"):
                    records[current["id"]] = (current["name"], current.get("namespace", ""))
                current = {}
            elif line.startswith("id: GO:"):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
        if current.get("id") and current.get("name"):
            records[current["id"]] = (current["name"], current.get("namespace", ""))
    return records


def gaf_gene_terms(path: Path) -> dict[str, set[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    mapping: dict[str, set[str]] = defaultdict(set)
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11 or not parts[4].startswith("GO:"):
                continue
            go_id = parts[4]
            aliases = {parts[1], parts[2]}
            aliases.update(x for x in re.split(r"[|,;\s]+", parts[10]) if x)
            if len(parts) > 16 and parts[16]:
                aliases.add(parts[16])
            for alias in aliases:
                clean = alias.strip()
                if clean:
                    mapping[clean].add(go_id)
                    mapping[clean.upper()].add(go_id)
    return mapping


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not OBO.exists():
        request = urllib.request.Request(
            "https://raw.githubusercontent.com/geneontology/go-ontology/master/src/ontology/go-edit.obo",
            headers={"User-Agent": "Plasmodium-target-atlas/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            OBO.write_bytes(response.read())
    ontology = read_obo(OBO)
    atlas = pd.read_csv(ATLAS, sep="\t", low_memory=False)
    all_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for short, (folder, label) in SPECIES.items():
        gaf = next(p for p in (GO_ROOT / folder).iterdir() if p.is_file())
        mapping = gaf_gene_terms(gaf)
        frame = atlas.loc[atlas["species_short"].eq(short)].copy()
        frame["go_terms"] = frame["gene_id"].astype(str).map(
            lambda gene: mapping.get(gene, mapping.get(gene.upper(), set()))
        )
        annotated = int(frame["go_terms"].map(bool).sum())
        coverage_rows.append(
            {
                "species": label,
                "background_predicted_essential": len(frame),
                "genes_with_GO": annotated,
                "GO_coverage": annotated / len(frame) if len(frame) else math.nan,
                "GAF_file": gaf.name,
            }
        )
        background = set(frame["gene_id"].astype(str))
        term_to_genes: dict[str, set[str]] = defaultdict(set)
        for gene, terms in zip(frame["gene_id"].astype(str), frame["go_terms"]):
            for term in terms:
                term_to_genes[term].add(gene)

        for priority in ("High", "Medium"):
            selected = set(frame.loc[frame["target_priority_v2"].eq(priority), "gene_id"].astype(str))
            tested: list[dict[str, object]] = []
            for term, genes in term_to_genes.items():
                bg_hits = len(genes & background)
                selected_hits = len(genes & selected)
                if bg_hits < 5 or selected_hits < 3:
                    continue
                a = selected_hits
                b = len(selected) - selected_hits
                c = bg_hits - selected_hits
                d = len(background) - len(selected) - c
                odds, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
                name, namespace = ontology.get(term, (term, "unknown"))
                tested.append(
                    {
                        "species": label,
                        "priority": priority,
                        "GO_ID": term,
                        "GO_name": name,
                        "namespace": namespace,
                        "selected_hits": a,
                        "selected_total": len(selected),
                        "background_hits": bg_hits,
                        "background_total": len(background),
                        "fold_enrichment": (a / len(selected)) / (bg_hits / len(background)),
                        "odds_ratio": odds,
                        "fisher_p_greater": p_value,
                    }
                )
            if tested:
                q = bh_adjust([float(row["fisher_p_greater"]) for row in tested])
                for row, fdr in zip(tested, q):
                    row["FDR_BH"] = fdr
                all_rows.extend(tested)

    enrichment = pd.DataFrame(all_rows).sort_values(
        ["species", "priority", "FDR_BH", "fold_enrichment"],
        ascending=[True, True, True, False],
    )
    enrichment.to_csv(OUT / "priority_GO_enrichment_all_terms.tsv", sep="\t", index=False)
    enrichment.loc[enrichment["FDR_BH"].le(0.05)].to_csv(
        OUT / "priority_GO_enrichment_FDR05.tsv", sep="\t", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(OUT / "GO_annotation_coverage.tsv", sep="\t", index=False)
    (OUT / "README.md").write_text(
        "# Priority-class GO enrichment\n\n"
        "High and Medium classes were compared separately with all predicted-essential genes "
        "from the same species using one-sided Fisher exact tests. Terms with fewer than five "
        "background genes or fewer than three selected genes were not tested. Benjamini-Hochberg "
        "FDR correction was applied within each species-priority analysis. GO annotations were "
        "also used as model features; these enrichments are therefore descriptive interpretation, "
        "not independent validation.\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print("tested_terms", len(enrichment), "FDR05", int(enrichment["FDR_BH"].le(0.05).sum()))


if __name__ == "__main__":
    main()
