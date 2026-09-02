"""Build the canonical Pf-Pb-Pk conserved-essentiality target-context screen.

The screen is deliberately derived from the frozen v37 full-refit application
scores.  A candidate is an exact 1:1:1 Pf-Pb-Pk orthogroup for which every
member is called essential by its species-matched, validation-fixed threshold.
Target registry, host-sequence, tractability and structural fields are read
only after candidate membership has been fixed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent
REV = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision"
V37 = REV / "ncomms_submission_v37"
ATLAS = V37 / "canonical_9142_complete_essentiality_evidence_atlas.tsv"
PATTERNS = V37 / "Source_Data" / "Fig5c_directional_orthogroup_patterns.tsv"
OUT = REV / "canonical_three_species_conserved_target_screen"
SOURCE = OUT / "source_data"
SPECIES = ["Pf", "Pb", "Pk"]
RNG = np.random.default_rng(20260831)


def write_tsv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, sep="\t", index=False)


def p_format(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def stratified_permutation(
    candidates: pd.DataFrame, all_calls: pd.DataFrame, iterations: int = 10_000
) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomise group-member selection within each species' canonical calls."""
    observed = int(candidates.target_supported_registry.astype(bool).sum())
    counts = candidates.groupby("species_short").size().reindex(SPECIES).astype(int)
    null = np.zeros(iterations, dtype=int)
    pools = {
        species: all_calls.loc[all_calls.species_short.eq(species), "target_supported_registry"].astype(bool).to_numpy()
        for species in SPECIES
    }
    for i in range(iterations):
        null[i] = sum(int(RNG.choice(pools[species], size=counts[species], replace=False).sum()) for species in SPECIES)
    empirical_p = (1 + int((null >= observed).sum())) / (iterations + 1)
    summary = pd.DataFrame(
        [
            {
                "comparison": "conserved exact-triplet members versus species-stratified random canonical calls",
                "iterations": iterations,
                "observed_target_supported_members": observed,
                "null_mean": float(null.mean()),
                "null_ci95_low": float(np.quantile(null, 0.025)),
                "null_ci95_high": float(np.quantile(null, 0.975)),
                "empirical_p_greater": empirical_p,
            }
        ]
    )
    return summary, null


def build_figure(flow: pd.DataFrame, members: pd.DataFrame, species_stats: pd.DataFrame,
                 permutation: pd.DataFrame, groups: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    navy, blue, teal, orange, grey = "#1f4e79", "#3a78ad", "#2f8f83", "#d77b32", "#76818c"

    # a: hierarchical candidate definition.
    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    stages = flow.to_dict("records")
    ys = np.linspace(0.82, 0.16, len(stages))
    colors = ["#e8eef4", "#d7e5f2", "#c7e7e1", "#f7dfc9"]
    for i, (stage, y, color) in enumerate(zip(stages, ys, colors)):
        width = 0.78 - 0.10 * i
        x = 0.5 - width / 2
        ax.add_patch(plt.Rectangle((x, y - 0.075), width, 0.15, facecolor=color, edgecolor=navy, linewidth=1.0))
        ax.text(0.5, y + 0.018, stage["stage"], ha="center", va="center", weight="bold", color=navy, fontsize=9)
        ax.text(0.5, y - 0.031, stage["display_count"], ha="center", va="center", color="#24313d", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(0.5, ys[i + 1] + 0.09), xytext=(0.5, y - 0.085),
                        arrowprops={"arrowstyle": "-|>", "color": grey, "lw": 1.2})
    ax.text(0.5, 0.02, "Candidate membership was fixed before any target, host or structure annotation.",
            ha="center", va="bottom", fontsize=7.2, color="#4d5964")
    ax.set_title("a  Strict conserved-essentiality definition", loc="left", fontweight="bold", pad=4)

    # b: threshold-centred margins are comparable within species.
    ax = fig.add_subplot(grid[0, 1])
    values = [members.loc[members.species_short.eq(s), "canonical_margin"].to_numpy() for s in SPECIES]
    violin = ax.violinplot(values, positions=np.arange(1, 4), showmeans=False, showmedians=True, widths=0.78)
    for body, color in zip(violin["bodies"], [blue, teal, orange]):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.85)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color(navy)
        violin[key].set_linewidth(0.9)
    ax.axhline(0, color="#4d5964", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks([1, 2, 3], ["Pf", "Pb", "Pk"])
    ax.set_ylabel("Threshold-centred essentiality margin")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b  Confidence across conserved triplet members", loc="left", fontweight="bold", pad=4)

    # c: registry records were not used in candidate construction.
    ax = fig.add_subplot(grid[1, 0])
    order = ["Pf", "Pb", "Pk", "All"]
    table = species_stats.set_index("species_short").loc[order]
    x = np.arange(len(order))
    observed = table["candidate_target_rate"].to_numpy() * 100
    background = table["other_canonical_target_rate"].to_numpy() * 100
    ax.bar(x - 0.18, observed, width=0.34, color=navy, label="Exact-triplet members")
    ax.bar(x + 0.18, background, width=0.34, color="#cbd5df", label="Other canonical calls")
    for xi, y, hits, total in zip(x - 0.18, observed, table["candidate_target_hits"], table["candidate_members"]):
        ax.text(xi, y + 0.08, f"{int(hits)}/{int(total)}", ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Target-registry membership (%)")
    ax.set_ylim(0, max(observed.max(), background.max()) * 1.42)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    observed_total = int(permutation.iloc[0].observed_target_supported_members)
    exp = permutation.iloc[0].null_mean
    pval = permutation.iloc[0].empirical_p_greater
    ax.text(0.98, 0.97, f"Stratified null: {exp:.1f} expected\nObserved: {observed_total}; P={p_format(pval)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#4d5964")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c  Post-score pharmacology concordance", loc="left", fontweight="bold", pad=4)

    # d: evidence navigation is explicitly post-score and group level.
    ax = fig.add_subplot(grid[1, 1])
    flags = pd.DataFrame(
        {
            "All three strong-margin": groups.all_members_strong_margin,
            "Any registry-supported member": groups.any_target_supported,
            "Any lower human-similarity context": groups.any_lower_host_context,
            "Any tractability annotation": groups.any_tractability_annotation,
            "Any prior pocket screen": groups.any_pocket_screen_pass,
        }
    )
    counts = flags.sum().sort_values()
    colors = ["#7ca6c9", navy, teal, orange, "#9a6bb0"]
    ax.barh(np.arange(len(counts)), counts.values, color=colors)
    ax.set_yticks(np.arange(len(counts)), counts.index, fontsize=7.4)
    ax.set_xlabel("Exact conserved-essentiality orthogroups")
    ax.set_xlim(0, len(groups) * 1.12)
    for yi, count in enumerate(counts.values):
        ax.text(count + len(groups) * 0.018, yi, f"{int(count):,}", va="center", fontsize=7.3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e8edf1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("d  Post-score evidence navigation", loc="left", fontweight="bold", pad=4)

    fig.suptitle("Conserved predicted-essentiality identifies a strict Pf-Pb-Pk follow-up layer", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", color="#1c2b39")
    for suffix, kwargs in {
        "svg": {"format": "svg"}, "pdf": {"format": "pdf"}, "png": {"format": "png", "dpi": 300},
        "tiff": {"format": "tiff", "dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(OUT / f"Fig6_conserved_three_species_target_context.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(ATLAS, sep="\t", low_memory=False)
    patterns = pd.read_csv(PATTERNS, sep="\t", low_memory=False)
    calls = atlas.loc[atlas.canonical_predicted_essential.astype(bool)].copy()
    strict_groups = patterns.loc[
        patterns.orthology_class.eq("single_copy_1to1to1") & patterns.full_pattern.eq("E/E/E"),
        ["homology_group_id", "full_pattern", "full_min_abs_margin", "full_mean_abs_margin"],
    ].copy()
    members = calls.merge(strict_groups, on="homology_group_id", how="inner", validate="many_to_one")
    membership = members.groupby("homology_group_id").species_short.agg(lambda x: set(x)).rename("species_set")
    valid_ids = membership.loc[membership.eq(set(SPECIES))].index
    members = members.loc[members.homology_group_id.isin(valid_ids)].copy()
    assert len(members) == len(strict_groups) * 3, "Every strict group must have exactly three canonical members"
    assert members.groupby("homology_group_id").size().eq(3).all()

    flow = pd.DataFrame(
        [
            {"stage": "All Pf-Pb-Pk genome records", "count": int(len(atlas)), "display_count": f"{len(atlas):,} genes"},
            {"stage": "Canonical predicted-essential calls", "count": int(len(calls)), "display_count": f"{len(calls):,} genes"},
            {"stage": "Exact Pf-Pb-Pk orthogroups", "count": int((patterns.orthology_class == 'single_copy_1to1to1').sum()), "display_count": f"{int((patterns.orthology_class == 'single_copy_1to1to1').sum()):,} groups"},
            {"stage": "All-three predicted essential", "count": int(len(strict_groups)), "display_count": f"{len(strict_groups):,} groups / {len(members):,} members"},
        ]
    )
    write_tsv(flow, "candidate_definition_flow.tsv")

    members = members.sort_values(["homology_group_id", "species_short"]).copy()
    members["strict_conserved_candidate"] = True
    write_tsv(members, "canonical_exact_triplet_conserved_essential_members.tsv")

    grouped = members.groupby("homology_group_id", sort=True)
    groups = grouped.agg(
        Pf_gene=("gene_id", lambda x: x[members.loc[x.index, "species_short"].eq("Pf")].iloc[0]),
        Pb_gene=("gene_id", lambda x: x[members.loc[x.index, "species_short"].eq("Pb")].iloc[0]),
        Pk_gene=("gene_id", lambda x: x[members.loc[x.index, "species_short"].eq("Pk")].iloc[0]),
        Pf_score=("canonical_essentiality_score", lambda x: float(x[members.loc[x.index, "species_short"].eq("Pf")].iloc[0])),
        Pb_score=("canonical_essentiality_score", lambda x: float(x[members.loc[x.index, "species_short"].eq("Pb")].iloc[0])),
        Pk_score=("canonical_essentiality_score", lambda x: float(x[members.loc[x.index, "species_short"].eq("Pk")].iloc[0])),
        min_margin=("canonical_margin", "min"),
        mean_margin=("canonical_margin", "mean"),
        all_members_strong_margin=("canonical_essentiality_confidence", lambda x: bool((x == "strong_margin").all())),
        any_target_supported=("target_supported_registry", lambda x: bool(x.astype(bool).any())),
        target_supported_member_count=("target_supported_registry", lambda x: int(x.astype(bool).sum())),
        any_lower_host_context=("lower_host_sequence_similarity_context", lambda x: bool(x.astype(bool).any())),
        any_intermediate_host_context=("intermediate_host_sequence_similarity_context", lambda x: bool(x.astype(bool).any())),
        any_tractability_annotation=("product_based_tractability_annotation", lambda x: bool((x == "supported").any())),
        any_structure_screened=("structure_screened", lambda x: bool(x.astype(bool).any())),
        any_pocket_screen_pass=("pocket_screen_pass", lambda x: bool(x.fillna(False).astype(bool).any())),
        any_legacy_docking=("legacy_docking_available", lambda x: bool(x.astype(bool).any())),
    ).reset_index()
    groups["follow_up_navigation"] = np.select(
        [groups.any_target_supported, groups.any_pocket_screen_pass | groups.any_tractability_annotation,
         groups.any_lower_host_context | groups.any_intermediate_host_context],
        ["registry-supported context", "tractability or structural context", "host-sequence context"],
        default="essentiality evidence only",
    )
    write_tsv(groups, "canonical_exact_triplet_conserved_essential_group_evidence.tsv")

    stat_rows = []
    for species in SPECIES + ["All"]:
        candidate = members if species == "All" else members.loc[members.species_short.eq(species)]
        all_call = calls if species == "All" else calls.loc[calls.species_short.eq(species)]
        other = all_call.loc[~all_call.gene_id.isin(candidate.gene_id)]
        a = int(candidate.target_supported_registry.astype(bool).sum())
        b = int(len(candidate) - a)
        c = int(other.target_supported_registry.astype(bool).sum())
        d = int(len(other) - c)
        odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        stat_rows.append({
            "species_short": species,
            "candidate_members": len(candidate),
            "candidate_target_hits": a,
            "candidate_target_rate": a / len(candidate),
            "other_canonical_members": len(other),
            "other_canonical_target_hits": c,
            "other_canonical_target_rate": c / len(other),
            "fisher_odds_ratio_greater": float(odds),
            "fisher_p_greater": float(pvalue),
        })
    species_stats = pd.DataFrame(stat_rows)
    write_tsv(species_stats, "canonical_exact_triplet_target_registry_comparison.tsv")
    permutation, null = stratified_permutation(members, calls)
    write_tsv(permutation, "canonical_exact_triplet_target_registry_permutation.tsv")
    pd.DataFrame({"stratified_random_target_hits": null}).to_csv(SOURCE / "Fig6c_stratified_permutation_null.tsv", sep="\t", index=False)

    group_summary = {
        "analysis_boundary": "Candidate membership was defined from canonical species-matched scores and exact 1:1:1 Pf-Pb-Pk orthogroups before target, host, tractability or structure fields were inspected.",
        "canonical_predicted_essential_genes": int(len(calls)),
        "exact_pf_pb_pk_orthogroups": int((patterns.orthology_class == "single_copy_1to1to1").sum()),
        "conserved_essential_orthogroups": int(len(groups)),
        "conserved_essential_members": int(len(members)),
        "target_supported_members": int(members.target_supported_registry.astype(bool).sum()),
        "target_supported_orthogroups": int(groups.any_target_supported.sum()),
        "all_three_strong_margin_orthogroups": int(groups.all_members_strong_margin.sum()),
        "any_tractability_annotation_orthogroups": int(groups.any_tractability_annotation.sum()),
        "any_pocket_screen_pass_orthogroups": int(groups.any_pocket_screen_pass.sum()),
        "random_seed": 20260831,
    }
    (OUT / "run_summary.json").write_text(json.dumps(group_summary, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Canonical exact-triplet conserved-essentiality screen\n\n"
        "This screen uses the frozen v37 canonical 9,142-gene species-matched atlas. Candidate membership is limited to exact Pf-Pb-Pk 1:1:1 orthogroups with E/E/E calls. "
        "Target-registry membership, human sequence context, product-based tractability and legacy structural fields are post-score annotations only. "
        "The target comparison uses a within-canonical-call, species-stratified permutation null; it is pharmacological concordance, not experimental validation or a therapeutic-efficacy claim.\n",
        encoding="utf-8",
    )
    flow.to_csv(SOURCE / "Fig6a_candidate_definition_flow.tsv", sep="\t", index=False)
    members.to_csv(SOURCE / "Fig6b_conserved_member_margins.tsv", sep="\t", index=False)
    species_stats.to_csv(SOURCE / "Fig6c_target_registry_comparison.tsv", sep="\t", index=False)
    groups.to_csv(SOURCE / "Fig6d_group_evidence_navigation.tsv", sep="\t", index=False)
    build_figure(flow, members, species_stats, permutation, groups)
    print(json.dumps(group_summary, indent=2))


if __name__ == "__main__":
    main()
