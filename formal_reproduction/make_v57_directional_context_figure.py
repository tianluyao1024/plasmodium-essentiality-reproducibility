"""Create the Figure 5 comparative context panel from frozen directional-group analyses.

This script performs no model fitting.  It assembles previously frozen refit,
orthology, sequence and GO outputs into a single Nature-style display item.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V56 = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v56_loso_full_workflow_validation"
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v57_directional_context_fig6"
FIG_DIR = OUT / "Main_Figures"
SOURCE_DIR = OUT / "Source_Data"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.75,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

PF = "#B8453D"
PB = "#4C7A48"
PK = "#4977A8"
ORANGE = "#C87932"
PURPLE = "#6B5A9E"
SLATE = "#40556D"
GREY = "#C7CDD4"


def copy_inputs() -> None:
    """Stage only frozen data tables needed for the display item."""
    for name in [
        "Supplementary_FigS8_orthomcl_support_summary.tsv",
        "Supplementary_FigS8_orthomcl_refit_stable_46_support.tsv",
        "Supplementary_TableS7_refit_stable_group_context.tsv",
        "Supplementary_TableS9_refit_stable_matched_controls.tsv",
        "Supplementary_TableS10_refit_stable_GO_enrichment.tsv",
        "Fig4d_directional_refit_stability.tsv",
    ]:
        src = V56 / "Source_Data" / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, SOURCE_DIR / name)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")


def pattern_state(pattern: str, position: int) -> int:
    return 1 if pattern.split("/")[position] == "E" else 0


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    copy_inputs()

    support = pd.read_csv(SOURCE_DIR / "Supplementary_FigS8_orthomcl_support_summary.tsv", sep="\t")
    support_refit = pd.read_csv(SOURCE_DIR / "Supplementary_FigS8_orthomcl_refit_stable_46_support.tsv", sep="\t")
    context = pd.read_csv(SOURCE_DIR / "Supplementary_TableS7_refit_stable_group_context.tsv", sep="\t")
    controls = pd.read_csv(SOURCE_DIR / "Supplementary_TableS9_refit_stable_matched_controls.tsv", sep="\t")
    go = pd.read_csv(SOURCE_DIR / "Supplementary_TableS10_refit_stable_GO_enrichment.tsv", sep="\t")
    stability = pd.read_csv(SOURCE_DIR / "Fig4d_directional_refit_stability.tsv", sep="\t")

    directional = context.loc[context["set"].eq("Refit-stable directional")].copy()
    matched = context.loc[context["set"].eq("Length-matched E/E/E control")].copy()
    if len(directional) != 46 or len(matched) != 46:
        raise ValueError("Expected 46 directional and 46 matched-control orthogroups")

    # a: support counts. b: all 92 pre-existing observations. c: GO only at BH FDR <0.05.
    support_row = support.loc[support["cohort"].eq("Refit-stable directional groups")].iloc[0]
    fig5a = pd.DataFrame(
        {
            "cohort": ["Refit-stable directional orthogroups"],
            "supported": [int(support_row["supported"])],
            "not_supported": [int(support_row["total"] - support_row["supported"])],
            "total": [int(support_row["total"])],
            "support_fraction": [float(support_row["support_fraction"])],
            "support_definition": ["All three genes occur in a shared external OrthoMCL 6.21 group"],
        }
    )
    fig5b = pd.concat(
        [
            directional.assign(cohort="Refit-stable directional"),
            matched.assign(cohort="Length-matched E/E/E control"),
        ],
        ignore_index=True,
    )[["cohort", "homology_group_id", "mean_pairwise_global_identity", "mean_protein_length"]]
    fig5c = go.loc[go["FDR_BH"].le(0.05)].copy()
    if len(fig5c) != 3:
        raise ValueError("Expected three BH-significant GO rows")

    matrix = directional.merge(
        support_refit[["homology_group_id", "orthomcl_supports_exact_triplet"]],
        on="homology_group_id",
        how="left",
    ).merge(
        stability[["homology_group_id", "reference_pattern_recovery_frequency"]],
        on="homology_group_id",
        how="left",
    )
    if matrix["orthomcl_supports_exact_triplet"].isna().any():
        raise ValueError("Missing external orthology support rows")
    matrix = matrix.sort_values(["full_pattern", "mean_pairwise_global_identity", "homology_group_id"], ascending=[True, True, True]).reset_index(drop=True)
    for i, species in enumerate(["Pf_call", "Pb_call", "Pk_call"]):
        matrix[species] = matrix["full_pattern"].map(lambda x: pattern_state(x, i))
    matrix["OrthoMCL_supported"] = matrix["orthomcl_supports_exact_triplet"].astype(int)
    matrix = matrix[
        [
            "homology_group_id",
            "full_pattern",
            "Pf_call",
            "Pb_call",
            "Pk_call",
            "OrthoMCL_supported",
            "mean_pairwise_global_identity",
            "mean_within_species_OrthoFinder_paralog_count",
            "reference_pattern_recovery_frequency",
        ]
    ]

    fig5a.to_csv(SOURCE_DIR / "Figure5a_external_orthology_support.tsv", sep="\t", index=False)
    fig5b.to_csv(SOURCE_DIR / "Figure5b_sequence_identity_matched_comparison.tsv", sep="\t", index=False)
    fig5c.to_csv(SOURCE_DIR / "Figure5c_GO_enrichment.tsv", sep="\t", index=False)
    matrix.to_csv(SOURCE_DIR / "Figure5d_group_level_evidence_matrix.tsv", sep="\t", index=False)

    # Core conclusion: directional groups are externally supported and have a distinct,
    # descriptive comparative context; neither annotation nor this figure alters atlas calls.
    fig = plt.figure(figsize=(7.25, 5.72), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.86, 1.14], height_ratios=[0.88, 1.12])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a
    supported = int(fig5a.loc[0, "supported"])
    unsupported = int(fig5a.loc[0, "not_supported"])
    ax_a.barh([0], [supported], color=SLATE, height=0.48, label="supported")
    ax_a.barh([0], [unsupported], left=[supported], color=GREY, height=0.48, label="not supported")
    ax_a.set_xlim(0, 46)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Refit-stable directional orthogroups")
    ax_a.set_xticks([0, 10, 20, 30, 40, 46])
    ax_a.text(supported / 2, 0, f"{supported}/46", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax_a.text(44.8, -0.43, "external OrthoMCL 6.21 support", ha="right", va="top", color=SLATE, fontsize=6.2)
    ax_a.spines["left"].set_visible(False)
    panel_label(ax_a, "a", "Independent orthology support")

    # b
    groups = [
        fig5b.loc[fig5b["cohort"].eq("Length-matched E/E/E control"), "mean_pairwise_global_identity"].to_numpy(),
        fig5b.loc[fig5b["cohort"].eq("Refit-stable directional"), "mean_pairwise_global_identity"].to_numpy(),
    ]
    violin = ax_b.violinplot(groups, positions=[0, 1], widths=0.72, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], [GREY, ORANGE]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(1.0)
    for x, values, color in zip([0, 1], groups, ["#7F8790", ORANGE]):
        # Fixed sinusoidal offsets prevent overplotting without introducing random data.
        offsets = 0.042 * np.sin(np.arange(len(values)) * 2.399 + x)
        ax_b.scatter(x + offsets, values, s=10, color=color, alpha=0.78, linewidths=0)
    ax_b.set_xlim(-0.58, 1.58)
    ax_b.set_ylim(0.22, 1.02)
    ax_b.set_xticks([0, 1], ["E/E/E\nmatched", "Directional"])
    ax_b.set_ylabel("Mean pairwise global identity")
    ax_b.text(0.5, 0.985, "median 0.747 vs 0.611; FDR = 0.049", ha="center", va="top", transform=ax_b.get_xaxis_transform(), fontsize=6.3)
    panel_label(ax_b, "b", "Lower sequence identity in directional triplets")

    # c
    go_plot = fig5c.sort_values("fold_enrichment", ascending=True).copy()
    labels = [
        "cytoplasmic dynein\ncomplex",
        "dynein intermediate-chain\nbinding",
        "microtubule",
    ]
    y = np.arange(len(go_plot))
    ax_c.hlines(y, 0, go_plot["fold_enrichment"], color="#D5D9DE", linewidth=1.5)
    ax_c.scatter(go_plot["fold_enrichment"], y, s=50, color=PURPLE, zorder=3)
    for yy, xx, fdr in zip(y, go_plot["fold_enrichment"], go_plot["FDR_BH"]):
        ax_c.text(xx + 0.25, yy, f"FDR {fdr:.3f}", va="center", fontsize=6.0, color=SLATE)
    ax_c.set_yticks(y, labels)
    ax_c.set_xlabel("Fold enrichment")
    ax_c.set_xlim(0, max(go_plot["fold_enrichment"]) + 5.4)
    ax_c.tick_params(axis="y", length=0)
    panel_label(ax_c, "c", "Functional annotation signals")

    # d: all 46 groups, with direct state calls and refit/orthology context.
    call_array = matrix[["Pf_call", "Pb_call", "Pk_call"]].to_numpy().T
    n_groups = len(matrix)
    ax_d.imshow(
        call_array,
        aspect="auto",
        cmap=ListedColormap(["#ECEFF2", "#34495E"]),
        vmin=0,
        vmax=1,
        interpolation="none",
        extent=(-0.5, n_groups - 0.5, 2.5, -0.5),
    )
    ax_d.imshow(
        matrix["OrthoMCL_supported"].to_numpy(float)[None, :],
        aspect="auto",
        cmap=ListedColormap(["#ECEFF2", "#4C7A48"]),
        vmin=0,
        vmax=1,
        interpolation="none",
        extent=(-0.5, n_groups - 0.5, 3.5, 2.5),
    )
    ax_d.imshow(
        matrix["reference_pattern_recovery_frequency"].to_numpy(float)[None, :],
        aspect="auto",
        cmap="Blues",
        vmin=0.8,
        vmax=1.0,
        interpolation="none",
        extent=(-0.5, n_groups - 0.5, 4.5, 3.5),
    )
    ax_d.set_xlim(-0.5, n_groups - 0.5)
    ax_d.set_ylim(4.5, -0.5)
    ax_d.set_yticks([0, 1, 2, 3, 4], ["Pf", "Pb", "Pk", "OrthoMCL", "refit"])
    ax_d.set_xticks([])
    ax_d.tick_params(axis="y", length=0)
    ax_d.set_xlabel("46 exact triplets, ordered by call pattern and identity", labelpad=6)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_title("Group-level evidence matrix", loc="left", pad=12, fontweight="bold")
    ax_d.text(-0.14, 1.08, "d", transform=ax_d.transAxes, fontsize=10, fontweight="bold", va="bottom")
    for yline in [2.5, 3.5]:
        ax_d.axhline(yline, color="white", linewidth=1.1)

    for ax in [ax_a, ax_b, ax_c]:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    base = FIG_DIR / "Fig5_refit_stable_directional_comparative_context"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    (SOURCE_DIR / "Figure5_figure_notes.md").write_text(
        "# Figure 5 source-data and analysis notes\n\n"
        "All values were generated previously from frozen atlas/refit outputs; no model was trained for this figure. "
        "Panel a reports external OrthoMCL 6.21 support. Panel b includes all 46 directional and 46 length-matched E/E/E groups. "
        "Panel c includes GO terms passing BH FDR <= 0.05. Panel d shows the exact state calls for all 46 groups, plus external orthology and refit recovery tracks.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
