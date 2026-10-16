"""Generate six publication-ready figures from the frozen release tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(r"D:\ProtoEss_NatureMicrobiology_submission_20261017")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig: plt.Figure, name: str, claim: str, evidence: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight", format="png")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight", format="tiff")
    fig.savefig(FIG / f"{name}.pdf", dpi=600, bbox_inches="tight", format="pdf")
    fig.savefig(FIG / f"{name}.svg", dpi=600, bbox_inches="tight", format="svg")
    (FIG / f"{name}_contract.json").write_text(json.dumps({"claim": claim, "evidence": evidence}, indent=2), encoding="utf-8")
    plt.close(fig)


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.20, 0.28, "95% redundancy\nreduction\n+ stratified split", "#e9f2fb"),
        (0.28, 0.55, 0.20, 0.28, "Raw 628 features\n9 routes\nfold-local dynamic features", "#e6f4ea"),
        (0.54, 0.55, 0.20, 0.28, "5-fold OOF\nLightGBM meta model\ntraining-only threshold", "#fff2cc"),
        (0.80, 0.55, 0.18, 0.28, "Frozen scores\nmain tests + atlas", "#fce4d6"),
        (0.22, 0.10, 0.22, 0.24, "Four-species LOSO\nPf / Pb / Pk / Tg", "#e9f2fb"),
        (0.52, 0.10, 0.22, 0.24, "Pk dual screen\nhuman-RBC condition", "#f4cccc"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="#555", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", weight="bold")
    for x1, y1, x2, y2 in [(0.22, .69, .28, .69), (.48, .69, .54, .69), (.74, .69, .80, .69), (.38, .55, .33, .34), (.65, .55, .63, .34)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444"})
    ax.text(.50, .94, "ProtoEss release architecture", transform=ax.transAxes, ha="center", weight="bold", fontsize=13)
    ax.text(.50, .02, "Test labels are used only for final scoring in the current run; historical profile selection was not fully blind.", transform=ax.transAxes, ha="center", fontsize=8, color="#8b0000")
    save(fig, "Figure_1_workflow", "The release separates model fitting, held-out scoring, cross-species transfer and condition analysis.", "Frozen run manifest, five-fold metrics, four-species LOSO and Pk dual-screen tables.")


def fig2() -> None:
    m = pd.read_csv(OUT / "metrics" / "frozen_fivefold_nine_route_metrics.tsv", sep="\t")
    m = m[m.split == "test"].copy()
    labels = ["Pf", "Pb", "Pk"]
    m["short"] = m.species.map(dict(zip(["Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi"], labels)))
    x = np.arange(3)
    w = .34
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - w / 2, m.AUROC, w, label="AUROC", color="#2f6690")
    ax.bar(x + w / 2, m.AUPRC, w, label="AUPRC", color="#f28e2b")
    ax.set_xticks(x, labels)
    ax.set_ylim(.65, 1.0)
    ax.set_ylabel("Held-out score")
    ax.set_title("Five-fold nine-route LightGBM: frozen test performance", weight="bold")
    ax.legend(frameon=False, ncol=2)
    for i, row in m.reset_index(drop=True).iterrows():
        ax.text(i - w / 2, row.AUROC + .012, f"{row.AUROC:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, row.AUPRC + .012, f"{row.AUPRC:.3f}", ha="center", fontsize=8)
    ax.text(.5, -.18, "Raw 628-feature matrix; original 95%-deduplicated stratified split", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "Figure_2_main_performance", "The frozen five-fold nine-route LightGBM ranks held-out essentiality labels across three malaria parasites.", "Recomputed from frozen_stack_predictions.tsv with thresholds recorded in the release manifest.")


def fig3() -> None:
    m = pd.read_csv(OUT / "loso" / "four_species_clade_loso_metrics.tsv", sep="\t")
    short = {"Plasmodium_falciparum": "Pf", "Plasmodium_berghei": "Pb", "Plasmodium_knowlesi": "Pk", "Toxoplasma_gondii": "Tg"}
    m["short"] = m.held_out.map(short)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(m))
    w = .36
    ax.bar(x - w / 2, m.test_AUROC, w, label="AUROC", color="#2f6690")
    ax.bar(x + w / 2, m.test_AUPRC, w, label="AUPRC", color="#7a5195")
    ax.set_xticks(x, m.short)
    ax.set_ylim(.55, 1.0)
    ax.set_ylabel("Held-out LOSO score")
    ax.set_title("Four-species clade leave-one-species-out transfer", weight="bold")
    ax.legend(frameon=False, ncol=2)
    for i, row in m.reset_index(drop=True).iterrows():
        ax.text(i - w / 2, row.test_AUROC + .015, f"{row.test_AUROC:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, row.test_AUPRC + .015, f"{row.test_AUPRC:.3f}", ha="center", fontsize=8)
    ax.text(.5, -.18, "Tg is included as the fourth held-out species; this is a companion analysis, not the main atlas test.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "Figure_3_loso_four_species", "Cross-species transfer is recipient-dependent and includes Toxoplasma gondii as a distant held-out species.", "clade_loso_full_stack_metrics.tsv, filtered to Pf/Pb/Pk/Tg.")


def fig4() -> None:
    app = pd.read_csv(OUT / "predictions" / "standardized_genomewide_predictions.tsv", sep="\t")
    counts = app.assign(essential=app.binary_call.eq("essential")).groupby("species").essential.agg(["sum", "count"])
    labels = ["Pf", "Pb", "Pk"]
    mapping = dict(zip(["Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi"], labels))
    counts["short"] = counts.index.map(mapping)
    counts = counts.set_index("short").loc[labels]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(3)
    ax.bar(x, counts["sum"], color=["#2f6690", "#59a14f", "#e15759"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Predicted-essential genes")
    ax.set_title("Genome-wide matched-species application", weight="bold")
    for i, v in enumerate(counts["sum"]):
        ax.text(i, v + counts["sum"].max() * .02, f"{int(v):,}", ha="center")
    ax.text(.5, -.18, "Calls are computational prioritization, not experimental essentiality evidence.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "Figure_4_genomewide_application", "Frozen matched-species scores define a versioned, queryable computational atlas.", "Standardized genome-wide prediction table generated from the frozen application output.")


def fig5() -> None:
    cat = pd.read_csv(OUT / "wp1_dual_screen" / "categorical_concordance.tsv", sep="\t")
    primary = cat[cat["set"] == "primary_exclude_tentative"].iloc[0]
    cont = pd.read_csv(OUT / "wp1_dual_screen" / "continuous_score_association.tsv", sep="\t").iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax1.bar(["Concordant", "Flipped"], [primary["n"] - primary["flips"], primary["flips"]], color=["#59a14f", "#e15759"])
    ax1.set_ylabel("Matched genes")
    ax1.set_title("Pk dual-screen calls")
    ax1.text(1, primary["flips"] + 60, f"{primary['flip_rate'] * 100:.2f}%", ha="center")
    ax2.axis("off")
    ax2.text(.05, .82, "Continuous association", weight="bold", fontsize=12)
    ax2.text(.05, .62, f"Spearman rho = {cont['spearman_rho_same_direction_essentiality_scores']:.3f}", fontsize=13)
    ax2.text(.05, .48, f"95% bootstrap CI: {cont['bootstrap_ci95_low']:.3f}–{cont['bootstrap_ci95_high']:.3f}")
    ax2.text(.05, .32, "Interpretation", weight="bold")
    ax2.text(.05, .19, "The two screens agree in direction overall,\nwhile a measurable condition-associated\ndiscordance remains.")
    fig.suptitle("P. knowlesi human- and macaque-red-cell condition comparison", weight="bold")
    save(fig, "Figure_5_pk_dual_screen", "Independent P. knowlesi screens show condition-associated essentiality discordance with positive continuous-score association.", "WP1 categorical_concordance.tsv and continuous_score_association.tsv; no A1H sequence-level claim.")


def fig6() -> None:
    p = pd.read_csv(OUT / "wp1_dual_screen" / "paired_cross_vs_within_pk_comparison.tsv", sep="\t")
    labels = ["Pf–Pk", "Pb–Pk"]
    rows = [p.iloc[0], p.iloc[1]]
    cross = [r["cross_species_rate"] for _, r in p.iloc[:2].iterrows()]
    within = [r["within_Pk_screen_rate"] for _, r in p.iloc[:2].iterrows()]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - .18, cross, .36, label="Cross-species", color="#2f6690")
    ax.bar(x + .18, within, .36, label="Within Pk dual screen", color="#f28e2b")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, .28)
    ax.set_ylabel("Label discordance")
    ax.set_title("Matched-group comparison of condition and species contrasts", weight="bold")
    ax.legend(frameon=False)
    ax.text(.5, -.18, "Descriptive contrast; not a causal partition of species and condition effects.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "Figure_6_species_condition_contrast", "Species and condition contrasts are compared on the same matched Pk gene groups.", "Paired cross-versus-within-Pk comparison table; descriptive, non-causal interpretation.")


def main() -> None:
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fn()
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
