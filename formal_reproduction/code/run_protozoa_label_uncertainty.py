"""Training-label uncertainty stress test for the four-species clade LOSO stack.

Only the source-species training labels are perturbed.  Validation and held
species test labels remain frozen.  Perturbations exchange equal positive and
negative subsets within each source species, preserving source prevalence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_clade_loso_full_stack as clade_stack


ROOT = Path.cwd()
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / os.environ.get("LABEL_NOISE_OUT_NAME", "label_uncertainty_clade_loso")
BASE = ROOT / "eukaryote_model_results" / "clade_loso_full_stack" / "protozoa"
WORK = OUT / "model_runs"
HELD = tuple(os.environ.get("LABEL_NOISE_HELD", "Plasmodium_falciparum,Plasmodium_berghei,Plasmodium_knowlesi,Toxoplasma_gondii").split(","))
FRACTIONS = tuple(float(x) for x in os.environ.get("LABEL_NOISE_FRACTIONS", "0.05,0.10,0.15").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("LABEL_NOISE_SEEDS", "20260824,20260825,20260826,20260827,20260828").split(","))


def save_state(state: dict) -> None:
    (OUT / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def baseline_rows() -> list[dict]:
    rows = []
    for held in HELD:
        data = json.loads((BASE / held / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"held_out": held, "noise_fraction": 0.0, "seed": 0, "run_type": "unperturbed_reference",
                     "training_records_exchanged": 0, "AUROC": data["test"]["AUROC"],
                     "AUPRC": data["test"]["AUPRC"], "F1": data["test"]["F1"]})
    return rows


def build_outputs(rows: pd.DataFrame) -> None:
    rows.to_csv(OUT / "label_uncertainty_all_runs.tsv", sep="\t", index=False)
    summaries = []
    for (held, fraction), part in rows.groupby(["held_out", "noise_fraction"], sort=True):
        for metric in ("AUROC", "AUPRC", "F1"):
            vals = part[metric].to_numpy(float)
            summaries.append({"held_out": held, "noise_fraction": fraction, "metric": metric,
                              "n_repeats": len(vals), "mean": vals.mean(), "sd": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                              "min": vals.min(), "max": vals.max()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "label_uncertainty_summary.tsv", sep="\t", index=False)
    summary.loc[summary.metric.isin(["AUROC", "AUPRC"])].to_csv(OUT / "label_uncertainty_summary_AUROC_AUPRC.tsv", sep="\t", index=False)

    colours = {"Plasmodium_falciparum": "#c55b3c", "Plasmodium_berghei": "#3e7cb1",
               "Plasmodium_knowlesi": "#5b9a5b", "Toxoplasma_gondii": "#9b6bb2"}
    labels = {"Plasmodium_falciparum": "Pf", "Plasmodium_berghei": "Pb",
              "Plasmodium_knowlesi": "Pk", "Toxoplasma_gondii": "Tg"}
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.79, wspace=0.24)
    for ax, metric, title in zip(axes, ("AUROC", "AUPRC"), ("a  AUROC", "b  AUPRC")):
        sub = summary.loc[summary.metric.eq(metric)]
        for held in HELD:
            cur = sub.loc[sub.held_out.eq(held)].sort_values("noise_fraction")
            ax.errorbar(cur.noise_fraction * 100, cur["mean"], yerr=cur.sd, marker="o", markersize=3.2,
                        linewidth=1.0, capsize=2, color=colours[held], label=labels[held])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.set_xlabel("Training-label exchange (%)")
        ax.set_ylabel(metric)
        ax.set_xticks([0, 5, 10, 15])
        ax.spines[["top", "right"]].set_visible(False)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, frameon=False, ncol=4, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.52, 0.995),
               columnspacing=1.1, handlelength=1.6)
    for ext in (".svg", ".pdf", ".tiff", ".png"):
        fig.savefig(OUT / f"Supplementary_FigS12_training_label_uncertainty{ext}", dpi=600 if ext in (".tiff", ".png") else None,
                    bbox_inches="tight")
    plt.close(fig)

    delta_rows = []
    for held in HELD:
        for metric in ("AUROC", "AUPRC", "F1"):
            base = rows.loc[(rows.held_out.eq(held)) & (rows.noise_fraction.eq(0)), metric].iloc[0]
            for fraction in FRACTIONS:
                vals = rows.loc[(rows.held_out.eq(held)) & (rows.noise_fraction.eq(fraction)), metric].to_numpy(float)
                delta_rows.append({"held_out": held, "noise_fraction": fraction, "metric": metric,
                                   "reference_unperturbed": base, "mean_delta": float(vals.mean() - base),
                                   "sd_across_repeats": float(vals.std(ddof=1)), "n_repeats": len(vals)})
    pd.DataFrame(delta_rows).to_csv(OUT / "label_uncertainty_delta_from_unperturbed.tsv", sep="\t", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    state = {"design": "source-species-stratified, prevalence-preserving training-label exchange; validation and test labels frozen",
             "fractions": list(FRACTIONS), "seeds": list(SEEDS), "held_species": list(HELD),
             "total_perturbed_runs": len(FRACTIONS) * len(SEEDS) * len(HELD), "completed_runs": 0, "status": "running"}
    rows = baseline_rows()
    frames, arrays, static_cols = clade_stack.load_clade("protozoa")
    joint = {s: pd.read_csv(clade_stack.JOINT / f"{s}_metadata.tsv", sep="\t", dtype={"record_id": str}) for s in ("train", "val", "test")}
    edges = clade_stack.load_edges(joint)
    hits = __import__("train_scope_strict_oof_stack").read_hits("clade", "protozoa", frames)
    old_out = clade_stack.OUT
    try:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                for held in HELD:
                    task_out = WORK / f"exchange_{int(fraction * 100):02d}pct_seed_{seed}" 
                    clade_stack.OUT = task_out
                    os.environ["CLOSO_TRAIN_LABEL_NOISE"] = str(fraction)
                    os.environ["CLOSO_NOISE_SEED"] = str(seed)
                    result = clade_stack.run_task("protozoa", held, frames, arrays, static_cols, hits, edges)
                    rows.append({"held_out": held, "noise_fraction": fraction, "seed": seed, "run_type": "perturbed_training_labels",
                                 "training_records_exchanged": result["training_records_exchanged"], "AUROC": result["test"]["AUROC"],
                                 "AUPRC": result["test"]["AUPRC"], "F1": result["test"]["F1"]})
                    state["completed_runs"] = len(rows) - len(HELD)
                    state["last_completed"] = {"held_out": held, "noise_fraction": fraction, "seed": seed}
                    save_state(state)
    finally:
        clade_stack.OUT = old_out
        os.environ.pop("CLOSO_TRAIN_LABEL_NOISE", None)
        os.environ.pop("CLOSO_NOISE_SEED", None)
    table = pd.DataFrame(rows)
    build_outputs(table)
    state["status"] = "complete"; state["completed_runs"] = len(FRACTIONS) * len(SEEDS) * len(HELD)
    save_state(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
