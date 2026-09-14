"""Render the frozen-LOSO essentiality-to-target validation figure.

This figure is deliberately built from the v56 analysis tables only.  It does
not re-fit a model, tune a threshold, or use target annotations during scoring.
"""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "loso_full_workflow_target_validation_v56"
OUT = DATA / "figures"

COLORS = {"Pf": "#B44D4B", "Pb": "#4F7E4B", "Pk": "#4D77A6", "Tg": "#6D5AA7"}
OBSERVED = "#222222"
PREDICTED = "#2E7DA1"


def clean_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=8.5, width=0.8, length=3)


def panel_label(ax, letter):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(DATA / "loso_gene_level_performance.tsv", sep="\t")
    enrich = pd.read_csv(DATA / "loso_observed_vs_predicted_target_enrichment.tsv", sep="\t")
    curve = pd.read_csv(DATA / "loso_target_cumulative_recovery.tsv", sep="\t")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.8, 7.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.97, top=0.94, bottom=0.105,
                            hspace=0.62, wspace=0.38)

    # a: held-out binary essentiality performance.
    ax = fig.add_subplot(grid[0, 0])
    metric_cols = ["AUROC", "AUPRC", "MCC", "sensitivity", "specificity"]
    metric_labels = ["AUROC", "AUPRC", "MCC", "Sensitivity", "Specificity"]
    matrix = metrics.set_index("species_short").loc[["Pf", "Pb", "Pk", "Tg"], metric_cols]
    im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(metric_cols)), metric_labels, rotation=27, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), [f"{s}  (n={int(matrix.loc[s].name and metrics.loc[metrics.species_short.eq(s), 'held_labelled_genes'].iloc[0]):,})" for s in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value >= 0.64 else "#1A1A1A")
    ax.set_title("LOSO performance on labelled held-out genes", loc="left", pad=11, fontsize=10.5, fontweight="bold")
    ax.set_xticks([x - 0.5 for x in range(1, len(metric_cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 4)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Metric value", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    panel_label(ax, "a")

    # b: observed vs frozen-predicted target enrichment.
    ax = fig.add_subplot(grid[0, 1])
    rows = enrich.copy()
    rows = rows[rows["target_set"].eq("Tier 1+2 target-supported registry") | rows["target_set"].eq("Tier 1 direct targets")]
    rows = rows.assign(row_label=rows["species_short"] + " | " + rows["target_set"].map({
        "Tier 1 direct targets": "direct", "Tier 1+2 target-supported registry": "supported"}))
    desired = ["Pf | direct", "Pf | supported", "Pb | supported", "Pk | supported"]
    rows["row_label"] = pd.Categorical(rows["row_label"], categories=desired, ordered=True)
    rows = rows.sort_values("row_label")
    ymap = {label: 3 - idx for idx, label in enumerate(desired)}
    for state, colour, marker, xoffset, label in [
        ("Observed essentiality", OBSERVED, "o", -0.12, "Observed essential"),
        ("LOSO predicted essentiality", PREDICTED, "s", 0.12, "LOSO predicted essential"),
    ]:
        sub = rows[rows.essentiality_state.eq(state)]
        ys, xs, lo, hi = [], [], [], []
        for _, row in sub.iterrows():
            ys.append(ymap[str(row.row_label)] + xoffset)
            xs.append(row.haldane_odds_ratio)
            lo.append(row.haldane_odds_ratio - row.or_ci95_low)
            hi.append(row.or_ci95_high - row.haldane_odds_ratio)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt=marker, color=colour, ms=5.8,
                    elinewidth=1.1, capsize=2.8, label=label, zorder=3)
    ax.axvline(1, color="#777777", lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.35, 300)
    ax.set_yticks([ymap[x] for x in desired], ["Pf direct (n=11)", "Pf supported (n=44)", "Pb supported (n=6)", "Pk supported (n=49)"])
    ax.set_xlabel("Known-target enrichment odds ratio (95% CI)")
    ax.set_title("Known targets in observed and predicted essential sets", loc="left", pad=11, fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              frameon=False, fontsize=8, handletextpad=0.35, columnspacing=1.2)
    clean_axis(ax)
    panel_label(ax, "b")

    # c: paired target-essential fractions.
    ax = fig.add_subplot(grid[1, 0])
    summary = []
    for label in desired:
        sub = rows[rows.row_label.eq(label)]
        observed = sub.loc[sub.essentiality_state.eq("Observed essentiality")].iloc[0]
        predicted = sub.loc[sub.essentiality_state.eq("LOSO predicted essentiality")].iloc[0]
        summary.append((label, observed.target_essential_fraction, predicted.target_essential_fraction,
                        int(observed.target_n)))
    for idx, (label, observed, predicted, n) in enumerate(summary):
        y = len(summary) - 1 - idx
        ax.plot([observed, predicted], [y, y], color="#B7B7B7", lw=1.25, zorder=1)
        ax.scatter(observed, y, s=39, color=OBSERVED, marker="o", zorder=3)
        ax.scatter(predicted, y, s=40, color=PREDICTED, marker="s", zorder=3)
        ax.text(min(0.985, max(observed, predicted) + 0.035), y, f"n={n}", va="center", fontsize=7.7)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_yticks(range(3, -1, -1), ["Pf direct", "Pf supported", "Pb supported", "Pk supported"])
    ax.set_xlabel("Fraction of known targets classified as essential")
    ax.set_title("Target-member concordance", loc="left", pad=11, fontsize=10.5, fontweight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OBSERVED,
               markeredgecolor=OBSERVED, markersize=6, label="Observed essential"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PREDICTED,
               markeredgecolor=PREDICTED, markersize=6, label="LOSO predicted essential"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False,
       fontsize=8, handletextpad=0.35, columnspacing=1.2)
    clean_axis(ax)
    panel_label(ax, "c")

    # d: all pre-specified ranking increments.
    ax = fig.add_subplot(grid[1, 1])
    for species in ["Pf", "Pb", "Pk"]:
        sub = curve[curve.species_short.eq(species)].sort_values("top_fraction")
        ax.plot(sub.top_fraction * 100, sub.target_recall * 100, lw=2.0, marker="o", ms=3.8,
                color=COLORS[species], label=f"{species} (n={int(sub.mapped_target_genes.iloc[0])})")
    ax.plot([10, 100], [10, 100], color="#777777", lw=0.8, ls="--", label="Uniform ranking")
    ax.set_xlim(8, 102)
    ax.set_ylim(0, 104)
    ax.set_xticks(range(10, 101, 10))
    ax.set_yticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Top fraction of frozen LOSO ranking")
    ax.set_ylabel("Cumulative known-target recovery")
    ax.set_title("Threshold-free target recovery", loc="left", pad=11, fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              frameon=False, fontsize=8, handlelength=1.5, columnspacing=1.1)
    clean_axis(ax)
    panel_label(ax, "d")

    figure_stem = OUT / "Figure5_LOSO_full_workflow_target_validation"
    fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Stable copies for the submission package and source-data hand-off.
    source_dir = OUT / "source_data"
    source_dir.mkdir(exist_ok=True)
    metrics.to_csv(source_dir / "Figure5a_LOSO_binary_performance.tsv", sep="\t", index=False)
    enrich.to_csv(source_dir / "Figure5b_c_target_enrichment_and_member_concordance.tsv", sep="\t", index=False)
    curve.to_csv(source_dir / "Figure5d_LOSO_target_cumulative_recovery.tsv", sep="\t", index=False)
    print(figure_stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
