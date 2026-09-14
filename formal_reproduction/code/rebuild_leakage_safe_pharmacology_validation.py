"""Recalculate post-score pharmacology concordance for the leakage-safe atlas.

Known target records are never used to fit a model, choose a threshold or set a
priority tier.  This script therefore evaluates their overlap only after all
essentiality calls have been frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent
TARGET_DIR = Path(r"D:\ProtoEss_full_application_inner_validation_v46\target_prioritisation_leakage_safe_v46")
TARGETS = Path(r"D:\ProtoEss_Reviewer_Inputs_v46\eukaryote_model_results\reviewer_context\target_registry_members.tsv")
LEGACY_EVIDENCE = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "loso_pharmacology_evidence_layer_audit" / "pharmacology_evidence_sets.tsv"
LOSO_DIR = Path(r"D:\ProtoEss_loso_inner_validation_v46")
LOSO_RESULTS = ROOT / "eukaryote_model_results" / "leakage_safe_clade_loso_v46" / "protozoa"
MANIFEST = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "clade" / "protozoa"
OUT = TARGET_DIR / "pharmacology_validation"
SPECIES = ("Pf", "Pb", "Pk")
FULL_NAMES = {
    "Pf": "Plasmodium_falciparum",
    "Pb": "Plasmodium_berghei",
    "Pk": "Plasmodium_knowlesi",
}
REGISTRY_SPECIES = {
    "Plasmodium_falciparum": "Plasmodium_falciparum_3D7",
    "Plasmodium_berghei": "Plasmodium_berghei_ANKA",
    "Plasmodium_knowlesi": "Plasmodium_knowlesi_H",
}
COLORS = {"Pf": "#B94A48", "Pb": "#4F7C4A", "Pk": "#476F9E"}


def target_flag(frame: pd.DataFrame, target_keys: set[tuple[str, str]]) -> pd.Series:
    return pd.Series(
        [(species, gene) in target_keys for species, gene in zip(frame.species.astype(str), frame.gene_id.astype(str))],
        index=frame.index, dtype=bool,
    )


def held_metadata(species: str) -> pd.DataFrame:
    frames = [pd.read_csv(MANIFEST / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str}) for split in ("train", "val", "test")]
    return pd.concat([f.loc[f.species.astype(str).eq(species)] for f in frames], ignore_index=True)


def label_and_length_matched_null(held: pd.DataFrame, short: str, rng: np.random.Generator) -> dict[str, object]:
    """Generate a transparent non-target null matched before inspecting scores.

    Each registry-supported held record is compared with a non-target record
    carrying the same experimental label and one of its 20 closest proteins by
    log protein length. Sampling is with replacement because the Pb registry is
    small and a strict one-to-one control would otherwise discard eligible
    targets. The score itself is never used to construct a match.
    """
    targets = held.loc[held.known_target_registry].copy()
    background = held.loc[~held.known_target_registry].copy()
    if targets.empty:
        raise RuntimeError(f"No mapped registry targets among held {short} records.")
    candidate_sets: list[np.ndarray] = []
    for row in targets.itertuples(index=False):
        pool = background.loc[background.label.eq(row.label)].copy()
        if pool.empty:
            pool = background.copy()
        distance = np.abs(np.log1p(pool.protein_length.to_numpy(float)) - np.log1p(float(row.protein_length)))
        nearest = pool.iloc[np.argsort(distance)[: min(20, len(pool))]]
        candidate_sets.append(nearest.score_percentile.to_numpy(float))
    observed = float(targets.score_percentile.median())
    null_medians = np.empty(10_000, dtype=float)
    for iteration in range(len(null_medians)):
        sampled = [choices[rng.integers(0, len(choices))] for choices in candidate_sets]
        null_medians[iteration] = float(np.median(sampled))
    null_low, null_high = np.quantile(null_medians, [0.025, 0.975])
    return {
        "species_short": short,
        "target_genes": int(len(targets)),
        "observed_median_percentile": observed,
        "null_median_percentile": float(np.median(null_medians)),
        "null_ci95_low": float(null_low),
        "null_ci95_high": float(null_high),
        "observed_minus_null": float(observed - np.median(null_medians)),
        "difference_ci95_low": float(observed - null_high),
        "difference_ci95_high": float(observed - null_low),
        "empirical_p_greater": float((1 + np.count_nonzero(null_medians >= observed)) / (1 + len(null_medians))),
        "matching_definition": "same experimental label; one of 20 nearest non-target proteins by log protein length; 10,000 draws with replacement",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(TARGET_DIR / "all_15591_leakage_safe_target_context.tsv", sep="\t", low_memory=False)
    predicted = pd.read_csv(TARGET_DIR / "all_8735_predicted_essential_target_context.tsv", sep="\t", low_memory=False)
    shared = pd.read_csv(TARGET_DIR / "three_species_shared_predicted_essential_members.tsv", sep="\t", low_memory=False)
    targets = pd.read_csv(TARGETS, sep="\t", low_memory=False)
    target_keys = set(zip(targets.species.astype(str), targets.gene_id.astype(str)))

    layers = {
        "All proteome genes": all_rows,
        "All predicted-essential genes": predicted,
        "High-priority predicted-essential genes": predicted.loc[predicted.target_priority.eq("High")],
        "Medium-priority predicted-essential genes": predicted.loc[predicted.target_priority.eq("Medium")],
        "Low-priority predicted-essential genes": predicted.loc[predicted.target_priority.eq("Low")],
        "Three-species shared predicted-essential members": shared,
    }
    rows: list[dict[str, object]] = []
    for layer_name, layer in layers.items():
        layer = layer.drop_duplicates(["species", "gene_id"]).copy()
        layer["known_target_registry"] = target_flag(layer, target_keys)
        for short in (*SPECIES, "All"):
            current = layer if short == "All" else layer.loc[layer.species_short.eq(short)]
            background = all_rows if short == "All" else all_rows.loc[all_rows.species_short.eq(short)]
            target_total = int(targets.shape[0] if short == "All" else targets.species_short.eq(short).sum())
            hit = int(current.known_target_registry.sum())
            bg_hit = int(target_flag(background, target_keys).sum())
            n = int(len(current)); bg_n = int(len(background))
            if n and bg_n:
                odds, p = fisher_exact([[hit, n - hit], [bg_hit, bg_n - bg_hit]], alternative="greater")
            else:
                odds, p = np.nan, np.nan
            rows.append({
                "candidate_layer": layer_name, "species_short": short, "candidate_genes": n,
                "known_target_hits": hit, "registry_targets": target_total,
                "target_recall": hit / target_total if target_total else np.nan,
                "target_fraction": hit / n if n else np.nan,
                "background_genes": bg_n, "background_target_fraction": bg_hit / bg_n if bg_n else np.nan,
                "fisher_odds_ratio_vs_proteome": odds, "fisher_p_greater_vs_proteome": p,
            })
    layer_summary = pd.DataFrame(rows)
    layer_summary.to_csv(OUT / "post_score_target_concordance_by_layer.tsv", sep="\t", index=False)

    # Target-label-naive LOSO ranking uses the frozen held-species vectors.
    loso_rows: list[dict[str, object]] = []
    matched_null_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(46)
    for short, species in FULL_NAMES.items():
        metadata = held_metadata(species)
        probability = np.load(LOSO_RESULTS / species / "test_prob.npy")
        if len(probability) != len(metadata):
            raise RuntimeError(f"LOSO record order mismatch for {species}.")
        held = metadata[["record_id", "species", "gene_id", "label", "protein_length"]].copy()
        held["score"] = probability
        held["score_percentile"] = held.score.rank(pct=True, method="average")
        held["registry_species"] = held.species.map(REGISTRY_SPECIES)
        held["known_target_registry"] = [
            (registry_species, gene) in target_keys
            for registry_species, gene in zip(held.registry_species.astype(str), held.gene_id.astype(str))
        ]
        target_scores = held.loc[held.known_target_registry, "score"]
        # Report the complete cumulative-recovery curve rather than a single
        # arbitrary cut-off. This is a ranking audit, not a thresholded call.
        for top_fraction in np.arange(0.10, 1.01, 0.10):
            cutoff = held.score.quantile(1 - top_fraction)
            selected = held.score.ge(cutoff)
            hits = int((selected & held.known_target_registry).sum())
            total = int(held.known_target_registry.sum())
            odds, p = fisher_exact(
                [[hits, int(selected.sum()) - hits], [total - hits, len(held) - int(selected.sum()) - (total - hits)]],
                alternative="greater",
            )
            loso_rows.append({
                "species_short": short, "species": species, "top_fraction": top_fraction,
                "held_genes": int(len(held)), "registry_targets_observed": total,
                "target_hits_in_top_fraction": hits, "target_recall": hits / total if total else np.nan,
                "median_target_score_percentile_among_proteome": float(held.loc[held.known_target_registry, "score_percentile"].median()),
                "fisher_odds_ratio": odds, "fisher_p_greater": p,
            })
        matched_null_rows.append(label_and_length_matched_null(held, short, rng))
        held.to_csv(OUT / f"{short}_LOSO_target_label_naive_scores.tsv", sep="\t", index=False)
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(OUT / "LOSO_target_label_naive_target_ranking.tsv", sep="\t", index=False)
    pd.DataFrame(matched_null_rows).to_csv(OUT / "LOSO_label_length_matched_target_control.tsv", sep="\t", index=False)

    # A fair old-versus-new comparison must preserve the narrow evidence sets
    # used in the historical LOSO report rather than comparing against the
    # expanded 111-member registry.
    legacy = pd.read_csv(LEGACY_EVIDENCE, sep="\t", low_memory=False)
    legacy = legacy.loc[legacy.evidence_type.isin(["direct_target", "species_supported_target"])].copy()
    legacy_rows: list[dict[str, object]] = []
    for short, species in FULL_NAMES.items():
        held = pd.read_csv(OUT / f"{short}_LOSO_target_label_naive_scores.tsv", sep="\t", low_memory=False)
        registry_species = REGISTRY_SPECIES[species]
        evidence = legacy.loc[legacy.species.eq(registry_species), "gene_id"].astype(str)
        evidence_ids = set(evidence)
        observed = held.gene_id.astype(str).isin(evidence_ids)
        total = int(observed.sum())
        cutoff = held.score.quantile(.80)
        hits = int((observed & held.score.ge(cutoff)).sum())
        legacy_rows.append({
            "species_short": short,
            "historical_evidence_definition": "Pf direct molecular targets; Pb/Pk species-supported pharmacological targets",
            "historical_evidence_members_mapped": total,
            "new_leakage_safe_hits_in_top20pct": hits,
            "new_leakage_safe_top20pct_recall": hits / total if total else np.nan,
            "new_leakage_safe_median_percentile": float(held.loc[observed, "score_percentile"].median()) if total else np.nan,
        })
    pd.DataFrame(legacy_rows).to_csv(OUT / "historical_evidence_set_retested_on_leakage_safe_LOSO.tsv", sep="\t", index=False)

    mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none"})
    figure = plt.figure(figsize=(7.15, 3.0), facecolor="white")
    ax1, ax2 = figure.subplots(1, 2)
    aggregate = layer_summary.loc[layer_summary.species_short.eq("All")].copy()
    display = aggregate.loc[aggregate.candidate_layer.isin([
        "All predicted-essential genes", "High-priority predicted-essential genes",
        "Medium-priority predicted-essential genes", "Low-priority predicted-essential genes",
        "Three-species shared predicted-essential members",
    ])]
    labels = ["All\nessential", "High", "Medium", "Low", "Shared\nE/E/E"]
    ax1.bar(range(len(display)), display.target_fraction * 100, color=["#4776A8", "#B06D2C", "#8B6BB5", "#9AA5B1", "#4F7C4A"])
    ax1.set_xticks(range(len(display)), labels); ax1.set_ylabel("Known-target fraction (%)")
    ax1.set_title("a  Post-score target overlap", loc="left", fontweight="bold", fontsize=8.5)
    ax1.spines[["top", "right"]].set_visible(False)
    top = loso.loc[loso.top_fraction.eq(.20)].set_index("species_short").loc[list(SPECIES)]
    nulls = pd.DataFrame(matched_null_rows).set_index("species_short").reindex(list(SPECIES))
    delta = nulls.observed_minus_null.to_numpy(float)
    delta_error = np.vstack([
        delta - nulls.difference_ci95_low.to_numpy(float),
        nulls.difference_ci95_high.to_numpy(float) - delta,
    ])
    bars = ax2.bar(range(3), delta, yerr=delta_error, capsize=2.5, color=[COLORS[s] for s in SPECIES])
    ax2.axhline(0, color="#667085", linestyle="--", linewidth=.8)
    ax2.set_xticks(range(3), list(SPECIES)); ax2.set_ylabel("Target − matched-null\nmedian percentile")
    ax2.set_title("b  Label- and length-matched null", loc="left", fontweight="bold", fontsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, nulls.itertuples()):
        offset = .012 if row.observed_minus_null >= 0 else -.012
        ax2.text(bar.get_x() + bar.get_width() / 2, row.observed_minus_null + offset,
                 f"P={row.empirical_p_greater:.3g}", ha="center",
                 va="bottom" if row.observed_minus_null >= 0 else "top", fontsize=6.3)
    figure.tight_layout(w_pad=3)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}: kwargs["dpi"] = 600
        figure.savefig(OUT / f"Supplementary_FigS13_leakage_safe_pharmacology_concordance.{ext}", **kwargs)
    plt.close(figure)

    summary = {
        "analysis": "post-score pharmacology concordance from leakage-safe v46 essentiality calls",
        "predicted_essential_total": int(len(predicted)),
        "known_target_registry_total": int(len(targets)),
        "known_target_recovered": int(predicted.known_target_registry.sum()),
        "known_target_used_for_model_or_priority": False,
        "LOSO_scope": "held-species rankings only; no LOSO prediction changes matched-species atlas membership",
        "matched_null": "Registry targets were compared with non-target held records matched on experimental label and protein length; score was not used for matching.",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
