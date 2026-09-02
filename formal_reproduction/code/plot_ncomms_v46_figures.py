"""Draw v46 conceptual and atlas figures from submission-facing source tables.

This renderer changes presentation only. It neither trains a model nor alters
the leakage-safe numerical tables. All output is white-background, vector-first
and suitable for Nature Communications submission.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get(
    "PROTOESS_SUBMISSION_OUT",
    ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v46_reproducible",
))
SRC = OUT / "Source_Data"
FIG = OUT / "Main_Figures"

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.75,
})

COLORS = {"Pf": "#B94A48", "Pb": "#4F7C4A", "Pk": "#476F9E", "Tg": "#8B6BB5"}
INK, GREY, LIGHT = "#1F2933", "#667085", "#F5F7FA"


def box(ax, x, y, w, h, title, body, color="#4776A8"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                           facecolor="white", edgecolor=color, linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.5,
            weight="bold", color=color, wrap=True)
    ax.text(x + w / 2, y + h * 0.29, body, ha="center", va="center", fontsize=6.1,
            color=INK, wrap=True, linespacing=1.35)


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, color=color))


def export(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def draw_fig1():
    summary = pd.read_csv(SRC / "Fig1_study_design_summary.tsv", sep="\t").set_index("stage")["value"].to_dict()
    fig = plt.figure(figsize=(7.15, 4.25), facecolor="white")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.995, "Fig. 1 | Study design for species-aware essentiality mapping", fontsize=10.5,
            weight="bold", va="top", color=INK)
    # Data panel
    ax.text(0.01, 0.90, "a", fontsize=12, weight="bold", color=INK)
    ax.text(0.045, 0.90, "Evidence-aware data assembly", fontsize=9, weight="bold", color=INK)
    species = [("Pf", 0.08), ("Pb", 0.15), ("Pk", 0.22), ("Tg", 0.29)]
    for label, x in species:
        ax.add_patch(plt.Circle((x, 0.81), 0.024, color=COLORS[label], alpha=0.95))
        ax.text(x, 0.81, label, ha="center", va="center", color="white", fontsize=7, weight="bold")
    ax.text(0.185, 0.745, "curated experimental labels", ha="center", fontsize=6.9, color=GREY)
    box(ax, 0.03, 0.52, 0.31, 0.17, "Molecular feature layers",
        "sequence and PLM embeddings\nhomology, annotation and interactions", "#4776A8")
    ax.text(0.185, 0.47, "fold-local construction of label-derived features", ha="center", fontsize=6.5, color=GREY)
    # Questions panel
    ax.text(0.37, 0.90, "b", fontsize=12, weight="bold", color=INK)
    ax.text(0.405, 0.90, "Two complementary prediction questions", fontsize=9, weight="bold", color=INK)
    box(ax, 0.38, 0.65, 0.25, 0.17, "Matched-species\nprediction",
        "species labels available\nvalidation-fixed threshold", "#4F7C4A")
    box(ax, 0.68, 0.65, 0.25, 0.17, "LOSO\ntransfer",
        "all target labels withheld\ntarget-label-naive ranking", "#B06D2C")
    box(ax, 0.49, 0.42, 0.33, 0.13, "Integrative OOF stack",
        "five branches → L2 meta-learner", "#6C5A9E")
    arrow(ax, (0.505, 0.65), (0.60, 0.55)); arrow(ax, (0.805, 0.65), (0.70, 0.55))
    # outputs panel
    ax.text(0.01, 0.36, "c", fontsize=12, weight="bold", color=INK)
    ax.text(0.045, 0.36, "Genome-scale atlas and comparative interpretation", fontsize=9, weight="bold", color=INK)
    box(ax, 0.04, 0.12, 0.20, 0.17, "Malaria atlas",
        f"{int(summary['Malaria-atlas genes']):,} genes\n{int(summary['Predicted-essential calls']):,} essential calls", "#4776A8")
    box(ax, 0.29, 0.12, 0.20, 0.17, "Conserved predicted\nessentiality",
        f"{int(summary['Concordant predicted-essential E/E/E triplets']):,} E/E/E\nPf–Pb–Pk triplets", "#4F7C4A")
    box(ax, 0.54, 0.12, 0.20, 0.17, "Directional hypotheses",
        "high margin → annotation\nexclusion → refit stability", "#B06D2C")
    box(ax, 0.79, 0.12, 0.17, 0.17, "Interpretation",
        "pharmacology\nhost and structure", "#6C5A9E")
    arrow(ax, (0.62, 0.42), (0.14, 0.29)); arrow(ax, (0.62, 0.42), (0.39, 0.29)); arrow(ax, (0.62, 0.42), (0.64, 0.29)); arrow(ax, (0.72, 0.42), (0.875, 0.29))
    ax.text(0.50, 0.035, "Tg contributes to LOSO transfer analyses but is not included in the final malaria atlas.",
            ha="center", fontsize=6.5, color=GREY)
    export(fig, "Fig1_study_design")


def draw_fig2():
    performance = pd.read_csv(SRC / "Fig2a_species_matched_performance.tsv", sep="\t")
    branch = pd.read_csv(SRC / "Fig2b_branch_and_ablation_benchmarks.tsv", sep="\t")
    calibration = pd.read_csv(SRC / "Fig2c_calibration.tsv", sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.05), facecolor="white")
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.29, wspace=.43)
    # a: uncertainty belongs to the held-out test set, not to a refitted model.
    a = axes[0]
    x = np.arange(len(performance)); width = 0.34
    auc_err = np.vstack([performance.AUROC - performance.AUROC_ci95_low, performance.AUROC_ci95_high - performance.AUROC])
    prc_err = np.vstack([performance.AUPRC - performance.AUPRC_ci95_low, performance.AUPRC_ci95_high - performance.AUPRC])
    a.bar(x - width / 2, performance.AUROC, width, yerr=auc_err, capsize=2.5, color="#4776A8", label="AUROC")
    a.bar(x + width / 2, performance.AUPRC, width, yerr=prc_err, capsize=2.5, color="#A8653B", label="AUPRC")
    a.set_xticks(x, ["Pf\n(n=527)", "Pb\n(n=205)", "Pk\n(n=407)"])
    a.set_ylim(.65, 1.01); a.set_ylabel("Held-out score")
    a.set_title("a  Species-matched performance", loc="left", weight="bold", fontsize=8.5)
    # b: branch comparison, leave the legend outside data-heavy regions.
    b = axes[1]
    keep = branch[branch.model.isin(["OOF stack", "LightGBM", "CatBoost", "Transformer", "species-aware expert"])].copy()
    order = ["OOF stack", "LightGBM", "CatBoost", "Transformer", "species-aware expert"]
    palette = {"OOF stack":"#6C5A9E", "LightGBM":"#4776A8", "CatBoost":"#A8653B", "Transformer":"#4F7C4A", "species-aware expert":"#8B6BB5"}
    for name in order:
        part = keep[keep.model == name].set_index("short_species").reindex(["Pf", "Pb", "Pk"])
        b.plot(range(3), part.AUROC, marker="o", markersize=3.4, linewidth=1.2, label=name, color=palette[name])
    b.set_xticks(range(3), ["Pf", "Pb", "Pk"]); b.set_ylim(.74, .94); b.set_ylabel("AUROC")
    b.set_title("b  Branch benchmarks", loc="left", weight="bold", fontsize=8.5)
    b.spines[["top", "right"]].set_visible(False)
    # c: calibration values are retained as points so the main panel stays compact.
    c = axes[2]
    summary = calibration[calibration["kind"] == "summary"].copy()
    summary["short_species"] = summary["species"].map({"Plasmodium_falciparum": "Pf", "Plasmodium_berghei": "Pb", "Plasmodium_knowlesi": "Pk"})
    metric = "brier_score"
    groups = summary.set_index("short_species")[metric].reindex(["Pf", "Pb", "Pk"])
    c.bar(range(3), groups, color=[COLORS[i] for i in groups.index], width=.58)
    c.set_xticks(range(3), groups.index); c.set_ylabel(metric.replace("_", " ").title())
    c.set_title("c  Calibration summary", loc="left", weight="bold", fontsize=8.5)
    c.spines[["top", "right"]].set_visible(False)
    # Keep legends outside the axes: the short labels must remain readable in
    # reduced journal layouts and may not cover test-set data.
    handles_a, labels_a = a.get_legend_handles_labels()
    handles_b, labels_b = b.get_legend_handles_labels()
    fig.legend(handles_a, labels_a, frameon=False, fontsize=5.6, loc="lower left",
               bbox_to_anchor=(.08, .015), ncol=2, handlelength=1.4)
    fig.legend(handles_b, labels_b, frameon=False, fontsize=5.0, loc="lower center",
               bbox_to_anchor=(.67, .005), ncol=3, handlelength=1.25, columnspacing=.8)
    export(fig, "Fig2_species_matched_performance")


def draw_fig3():
    """Render LOSO and the species-prior-free Tg augmentation control."""
    loso = pd.read_csv(SRC / "Fig3a_four_species_LOSO.tsv", sep="\t")
    control = pd.read_csv(SRC / "Fig3b_shared_molecular_Tg_source_bootstrap.tsv", sep="\t")
    order = ["Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi", "Toxoplasma_gondii"]
    short = {"Plasmodium_falciparum": "Pf", "Plasmodium_berghei": "Pb", "Plasmodium_knowlesi": "Pk", "Toxoplasma_gondii": "Tg"}
    loso = loso.set_index("species").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.0), facecolor="white")
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.25, wspace=.43)

    a = axes[0]
    x = np.arange(len(loso)); width = 0.34
    auc_err = np.vstack([loso.AUROC - loso.AUROC_low, loso.AUROC_high - loso.AUROC])
    pr_err = np.vstack([loso.AUPRC - loso.AUPRC_low, loso.AUPRC_high - loso.AUPRC])
    a.bar(x - width / 2, loso.AUROC, width, yerr=auc_err, capsize=2.3, color="#4776A8", label="AUROC")
    a.bar(x + width / 2, loso.AUPRC, width, yerr=pr_err, capsize=2.3, color="#A8653B", label="AUPRC")
    a.set_xticks(x, [short[s] for s in order]); a.set_ylim(.55, 1.01); a.set_ylabel("Held-species score")
    a.set_title("a  Four-species LOSO", loc="left", weight="bold", fontsize=8.5)
    a.spines[["top", "right"]].set_visible(False)

    for axis, metric, panel in zip(axes[1:], ("AUROC", "AUPRC"), ("b", "c")):
        part = control.loc[control.metric.eq(metric)].copy()
        labels = [short[s] for s in part.held_out_species]
        values = part.delta.to_numpy()
        errors = np.vstack([values - part.bootstrap_ci_low.to_numpy(), part.bootstrap_ci_high.to_numpy() - values])
        bars = axis.bar(np.arange(len(part)), values, yerr=errors, capsize=2.5,
                        color=[COLORS[label] for label in labels], width=.58)
        axis.axhline(0, color=GREY, linestyle="--", linewidth=.8)
        axis.set_xticks(np.arange(len(part)), labels); axis.set_ylabel(f"Change in {metric}")
        axis.set_title(f"{panel}  Addition of Tg source data", loc="left", weight="bold", fontsize=8.5)
        axis.spines[["top", "right"]].set_visible(False)
        lower, upper = (part.bootstrap_ci_low.min(), part.bootstrap_ci_high.max())
        pad = max(.004, (upper - lower) * .22)
        axis.set_ylim(lower - pad, upper + pad)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = pad * .08 if value >= 0 else -pad * .08
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.3f}",
                      ha="center", va=va, fontsize=6.2)
    handles, labels_legend = a.get_legend_handles_labels()
    fig.legend(handles, labels_legend, frameon=False, fontsize=5.8, loc="lower left",
               bbox_to_anchor=(.08, .02), ncol=2, handlelength=1.4)
    export(fig, "Fig3_target_label_naive_transfer_controls")


def draw_fig5():
    rank = pd.read_csv(SRC / "Fig5a_LOSO_target_label_naive_target_ranking.tsv", sep="\t")
    matched = pd.read_csv(SRC / "Fig5b_LOSO_label_length_matched_target_control.tsv", sep="\t")
    recovery = pd.read_csv(SRC / "Supplementary_FigS5a_target_recovery.tsv", sep="\t")
    summary = rank[rank.top_fraction.eq(.20)].copy().set_index("species_short").reindex(["Pf", "Pb", "Pk"])
    labels = ["Pf", "Pb", "Pk"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.9), facecolor="white")
    fig.subplots_adjust(left=.08, right=.985, top=.88, bottom=.20, wspace=.45)
    a = axes[0]
    a.bar(range(3), summary.median_target_score_percentile_among_proteome, color=[COLORS[s] for s in labels], width=.58)
    a.axhline(.5, color=GREY, linestyle="--", linewidth=.8)
    a.set_xticks(range(3), labels); a.set_ylim(0, 1.03); a.set_ylabel("Median score percentile")
    a.set_title("a  Held-species target ranking", loc="left", weight="bold", fontsize=8.5)
    a.spines[["top", "right"]].set_visible(False)
    b = axes[1]
    m = matched.set_index("species_short").reindex(labels)
    delta = m.observed_minus_null.to_numpy(float)
    err = np.vstack([
        delta - m.difference_ci95_low.to_numpy(float),
        m.difference_ci95_high.to_numpy(float) - delta,
    ])
    b.bar(range(3), delta, yerr=err, capsize=2.5, color=[COLORS[s] for s in labels], width=.58)
    b.axhline(0, color=GREY, linestyle="--", linewidth=.8)
    b.set_xticks(range(3), labels); b.set_ylabel("Target − matched-null\nmedian percentile")
    b.set_title("b  Label- and length-matched null", loc="left", weight="bold", fontsize=8.5)
    b.spines[["top", "right"]].set_visible(False)
    c = axes[2]
    rec = recovery.set_index("species_short").reindex(labels)
    bars = c.bar(range(3), rec.recovery_fraction * 100, color=[COLORS[s] for s in labels], width=.58)
    c.set_xticks(range(3), labels); c.set_ylim(0, 108); c.set_ylabel("Registry recovery (%)")
    c.set_title("c  Matched-species atlas recovery", loc="left", weight="bold", fontsize=8.5)
    for bar, row in zip(bars, rec.itertuples()):
        c.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2, f"{row.recovered_by_canonical}/{row.target_supported_genes}", ha="center", fontsize=6.6)
    c.spines[["top", "right"]].set_visible(False)
    export(fig, "Fig5_pharmacological_concordance")


def draw_fig4():
    atlas = pd.read_csv(SRC / "Fig4a_atlas_scores_with_provenance.tsv", sep="\t")
    flow = pd.read_csv(SRC / "Fig4b_directional_filtering_flow.tsv", sep="\t").set_index("stage")["orthogroups"].to_dict()
    patterns = pd.read_csv(SRC / "Fig4c_directional_patterns.tsv", sep="\t")
    refit = pd.read_csv(SRC / "Fig4d_directional_refit_stability.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), facecolor="white")
    fig.subplots_adjust(left=.09, right=.985, top=.94, bottom=.09, hspace=.38, wspace=.18)
    # a atlas counts
    a = axes[0, 0]
    groups = atlas.groupby("species_short")["full_formal_stack_predicted_essential"].agg(["size", "sum"]).reindex(["Pf", "Pb", "Pk"])
    x = range(3); bars = a.bar(x, groups["sum"], color=[COLORS[i] for i in groups.index], width=0.62)
    a.set_xticks(list(x), ["Pf", "Pb", "Pk"]); a.set_ylabel("Predicted-essential genes")
    a.set_title("a  Genome-scale matched-species atlas", loc="left", weight="bold", fontsize=9)
    for b, total, pred in zip(bars, groups["size"], groups["sum"]):
        a.text(b.get_x()+b.get_width()/2, pred+max(groups["sum"])*0.025, f"{int(pred):,}\n/ {int(total):,}", ha="center", fontsize=7)
    a.spines[["top","right"]].set_visible(False); a.set_ylim(0, max(groups["sum"])*1.23)
    # b: a data-bearing stratification view rather than a textual flowchart.
    b = axes[0, 1]
    stage_labels = [
        "Exact 1:1:1 triplets", "Shared E/E/E layer", "High-margin\ndirectional", 
        "Directional after\nannotation exclusion", "Refit-stable\ndirectional",
    ]
    values = np.array([
        flow["Exact 1:1:1 orthogroups"], flow["Concordant predicted-essential E/E/E triplets"],
        flow["Strong directional, full model"], flow["Retained after annotation exclusion with same direction"],
        flow["Recovered in at least 80% of paired refits"],
    ], dtype=float)
    colours = ["#4776A8", "#4F7C4A", "#C97A3D", "#C97A3D", "#B06D2C"]
    ypos = np.arange(len(values))
    bars = b.barh(ypos, values, color=colours, height=.64)
    b.set_yticks(ypos, stage_labels, fontsize=6.7)
    b.invert_yaxis(); b.set_xlim(0, max(values) * 1.17)
    b.set_xlabel("Exact orthogroups", fontsize=7.4)
    b.set_title("b  Exact orthogroup resolution", loc="left", weight="bold", fontsize=9)
    b.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        b.text(value + max(values) * .018, bar.get_y() + bar.get_height() / 2, f"{int(value):,}",
               va="center", fontsize=7.2, color=INK)
    # c patterns
    c = axes[1,0]
    pat = patterns.loc[patterns.get("strong_full", True).astype(bool)].copy()
    counts = pat["full_pattern"].value_counts().head(6).sort_values()
    c.barh(range(len(counts)), counts.values, color="#C97A3D")
    c.set_yticks(range(len(counts)), counts.index, fontsize=7); c.set_xlabel("Exact orthogroups")
    c.set_title("c  High-margin directional call patterns", loc="left", weight="bold", fontsize=9)
    c.spines[["top","right"]].set_visible(False)
    # d refit
    d = axes[1,1]
    stable = refit["reference_pattern_recovery_frequency"]
    d.hist(stable, bins=12, color="#6C5A9E", edgecolor="white")
    d.axvline(.80, color="#B06D2C", linestyle="--", linewidth=1)
    d.set_xlabel("Reference-pattern recovery frequency"); d.set_ylabel("Orthogroups")
    d.set_title("d  Stability across 100 refits", loc="left", weight="bold", fontsize=9)
    d.spines[["top","right"]].set_visible(False)
    export(fig, "Fig4_atlas_and_directionality")


if __name__ == "__main__":
    if not OUT.exists():
        sys.exit("Build ncomms_submission_v46_reproducible before rendering its figures.")
    draw_fig1(); draw_fig2(); draw_fig3(); draw_fig4(); draw_fig5()
