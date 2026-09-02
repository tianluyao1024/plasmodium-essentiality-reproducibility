"""Eligibility-matched sensitivity analysis for the conserved triplet screen.

The strict conserved screen selects exact Pf/Pb/Pk 1:1:1 orthogroups whose
three members are canonical predicted-essential calls.  The original
species-stratified null sampled canonical calls and therefore did not preserve
the exact-triplet eligibility universe.  This script samples whole exact
Pf/Pb/Pk orthogroups instead.  Target-registry annotation is read only after
candidate membership has been defined.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REV = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision"
OUT = REV / "exact_triplet_eligibility_matched_null"

# The historical complete atlas is used solely as the immutable orthogroup map.
# Current canonical scores and calls always come from the v40 source table.
GROUP_MAP = REV / "ncomms_annotation_excluded" / "priority_rule_sensitivity" / "Complete_atlas_15591_genes_ncomms_formal_scores.tsv"
CURRENT_SCORES = REV / "ncomms_submission_v40" / "Source_Data" / "Fig5a_canonical_genomewide_scores_with_provenance.tsv"
REGISTRY = REV / "ncomms_submission_v34" / "Source_Data" / "Supplementary_FigS5a_v34_target_registry_members.tsv"
STRICT_GROUPS = REV / "canonical_three_species_conserved_target_screen" / "canonical_exact_triplet_conserved_essential_group_evidence.tsv"

SPECIES = ("Pf", "Pb", "Pk")
N_PERMUTATIONS = 100_000
SEED = 20260831


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score = read_tsv(CURRENT_SCORES)
    group_map = read_tsv(GROUP_MAP)[["species", "gene_id", "homology_group_id"]].drop_duplicates()
    registry = read_tsv(REGISTRY)[["species", "gene_id", "target_supported"]].drop_duplicates()
    strict = read_tsv(STRICT_GROUPS)

    data = score.merge(group_map, on=["species", "gene_id"], how="left", validate="one_to_one")
    data = data.merge(registry, on=["species", "gene_id"], how="left", validate="one_to_one")
    data["target_supported"] = data["target_supported"].fillna(False).astype(bool)

    group_counts = (
        data.dropna(subset=["homology_group_id"])
        .groupby("homology_group_id")
        .agg(
            n_members=("gene_id", "size"),
            n_species=("species_short", "nunique"),
            species_set=("species_short", lambda x: ";".join(sorted(x))),
        )
        .reset_index()
    )
    eligible = group_counts.loc[
        group_counts.n_members.eq(3)
        & group_counts.n_species.eq(3)
        & group_counts.species_set.eq("Pb;Pf;Pk"),
        "homology_group_id",
    ].tolist()

    strict_ids = strict["homology_group_id"].dropna().astype(str).unique().tolist()
    eligible_set = set(map(str, eligible))
    strict_ids = [x for x in strict_ids if x in eligible_set]
    if len(strict_ids) != len(set(strict_ids)):
        raise ValueError("Strict group identifiers are not unique.")
    if not set(strict_ids).issubset(eligible_set):
        raise ValueError("Strict candidate groups are not all exact-triplet eligible.")

    group_hits = (
        data.loc[data.homology_group_id.astype(str).isin(eligible_set)]
        .assign(homology_group_id=lambda x: x.homology_group_id.astype(str))
        .groupby("homology_group_id", sort=False)["target_supported"]
        .sum()
        .reindex(sorted(eligible_set), fill_value=0)
        .astype(int)
    )
    observed = int(group_hits.reindex(strict_ids).sum())
    n_candidates = len(strict_ids)
    rng = np.random.default_rng(SEED)
    values = group_hits.to_numpy()
    null = np.empty(N_PERMUTATIONS, dtype=np.int16)
    for i in range(N_PERMUTATIONS):
        null[i] = rng.choice(values, size=n_candidates, replace=False).sum()

    p_greater_equal = (1 + int((null >= observed).sum())) / (N_PERMUTATIONS + 1)
    summary = pd.DataFrame(
        [
            {
                "candidate_definition": "exact Pf/Pb/Pk 1:1:1 orthogroups with three canonical predicted-essential members",
                "eligibility_universe": "all exact Pf/Pb/Pk 1:1:1 orthogroups",
                "eligible_exact_triplet_orthogroups": int(len(eligible_set)),
                "strict_candidate_orthogroups": int(n_candidates),
                "candidate_members": int(n_candidates * 3),
                "registry_supported_members_observed": observed,
                "null_mean_registry_supported_members": float(null.mean()),
                "null_sd_registry_supported_members": float(null.std(ddof=1)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "one_sided_empirical_p_greater_equal": float(p_greater_equal),
                "permutations": N_PERMUTATIONS,
                "seed": SEED,
            }
        ]
    )
    summary.to_csv(OUT / "exact_triplet_eligibility_matched_null_summary.tsv", sep="\t", index=False)
    pd.DataFrame({"registry_supported_member_count": null}).to_csv(
        OUT / "exact_triplet_eligibility_matched_null_distribution.tsv", sep="\t", index=False
    )
    group_detail = group_hits.rename("registry_supported_members").reset_index()
    group_detail["strict_candidate"] = group_detail.homology_group_id.isin(strict_ids)
    group_detail.to_csv(OUT / "exact_triplet_eligibility_universe.tsv", sep="\t", index=False)
    (OUT / "run_summary.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
