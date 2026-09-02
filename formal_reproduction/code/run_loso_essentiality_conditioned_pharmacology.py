"""Test pharmacological evidence ranks against matched LOSO-score backgrounds.

This audit does not use pharmacology membership during model fitting, threshold
selection, or score generation.  It answers a narrower question than the
unconditional analysis: within the same binary LOSO predicted-essentiality
state and among proteins of similar length, do evidence genes remain unusually
highly ranked by the continuous frozen LOSO score?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision"
EVIDENCE = REVIEW / "loso_pharmacology_evidence_layer_audit"
GENOME = REVIEW / "loso_full_genome_closure" / "genomewide_predictions"
OUT = REVIEW / "loso_essentiality_conditioned_pharmacology"
SEED = 20260821
N_PERMUTATIONS = 10_000
MATCHES_PER_GENE = 30


def percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def load_genome(species: str) -> pd.DataFrame:
    species_prefix = {
        "Plasmodium_falciparum_3D7": "Plasmodium_falciparum",
        "Plasmodium_berghei_ANKA": "Plasmodium_berghei",
        "Plasmodium_knowlesi_H": "Plasmodium_knowlesi",
    }[species]
    path = GENOME / f"{species_prefix}_loso_full_genome.tsv"
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["score_percentile"] = percentile(frame["loso_full_genome_score"])
    frame["log_length"] = np.log1p(frame["protein_length"].clip(lower=1))
    return frame


def matched_control_pool(target: pd.Series, background: pd.DataFrame) -> pd.DataFrame:
    same_state = background.loc[
        background.loso_full_genome_predicted_essential.eq(target.loso_full_genome_predicted_essential)
    ].copy()
    if same_state.empty:
        return same_state
    same_state["length_distance"] = (same_state.log_length - target.log_length).abs()
    # The 30 closest length-matched candidates make the null conditional on both
    # binary essentiality state and protein scale without conditioning on score.
    return same_state.nsmallest(min(MATCHES_PER_GENE, len(same_state)), "length_distance")


def analyse_set(evidence: pd.DataFrame, genome: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    ids = set(evidence.gene_id.astype(str))
    targets = genome.loc[genome.gene_id.astype(str).isin(ids)].copy()
    if len(targets) != len(ids):
        missing = sorted(ids - set(targets.gene_id.astype(str)))
        raise ValueError(f"Evidence identifiers absent from genome score table: {missing}")
    background = genome.loc[~genome.gene_id.astype(str).isin(ids)].copy()
    pools = []
    audit_rows = []
    for target in targets.itertuples(index=False):
        pool = matched_control_pool(pd.Series(target._asdict()), background)
        if pool.empty:
            raise ValueError(f"No matched controls for {target.gene_id}")
        pools.append(pool.score_percentile.to_numpy())
        audit_rows.append({
            "species": target.species,
            "evidence_set": evidence.evidence_set.iloc[0],
            "gene_id": target.gene_id,
            "observed_score_percentile": target.score_percentile,
            "predicted_essential_state": bool(target.loso_full_genome_predicted_essential),
            "protein_length": int(target.protein_length),
            "matched_control_n": len(pool),
            "matched_control_median_percentile": float(pool.score_percentile.median()),
        })
    observed = float(targets.score_percentile.median())
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        null[i] = np.median([pool[rng.integers(len(pool))] for pool in pools])
    empirical_p = (1 + int((null >= observed).sum())) / (N_PERMUTATIONS + 1)
    result = {
        "species": evidence.species_label.iloc[0],
        "evidence_set": evidence.evidence_set.iloc[0],
        "evidence_type": evidence.evidence_type.iloc[0],
        "evidence_gene_n": len(targets),
        "observed_median_percentile": observed,
        "matched_null_median_percentile": float(np.median(null)),
        "median_difference": float(observed - np.median(null)),
        "empirical_p_greater": empirical_p,
        "null_95ci_low": float(np.quantile(null, 0.025)),
        "null_95ci_high": float(np.quantile(null, 0.975)),
        "permutations": N_PERMUTATIONS,
        "matching_rule": "Same binary frozen LOSO predicted-essentiality state; 30 nearest proteins in log protein length; evidence genes excluded from control pools.",
        "interpretation_boundary": "Conditions on binary predicted-essentiality state and protein length, not on continuous score. It does not establish binding, efficacy, safety, or a direct target relationship.",
    }
    return result, pd.DataFrame(audit_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(EVIDENCE / "pharmacology_evidence_sets.tsv", sep="\t", low_memory=False)
    genome_cache: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(SEED)
    summaries, audits = [], []
    for _, group in evidence.groupby(["species", "evidence_set"], sort=True):
        species = group.species.iloc[0]
        if species not in genome_cache:
            genome_cache[species] = load_genome(species)
        summary, audit = analyse_set(group, genome_cache[species], rng)
        summaries.append(summary)
        audits.append(audit)
    pd.DataFrame(summaries).to_csv(OUT / "essentiality_conditioned_pharmacology_summary.tsv", sep="\t", index=False)
    pd.concat(audits, ignore_index=True).to_csv(OUT / "essentiality_conditioned_pharmacology_gene_audit.tsv", sep="\t", index=False)
    manifest = {
        "purpose": "Conditional ranking audit for pharmacological evidence under frozen held-species LOSO scores.",
        "score_used_for_matching": False,
        "continuous_score_is_the_outcome": True,
        "matching_variables": ["binary frozen LOSO predicted-essentiality state", "nearest log protein length"],
        "annotation_density_matching": "Not applied because no species-comparable whole-proteome annotation-density field exists in the frozen LOSO application tables.",
        "permutations": N_PERMUTATIONS,
        "seed": SEED,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
