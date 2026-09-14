"""Validate the frozen protozoan LOSO workflow without fitting another model.

For each held-out species, this script reconstructs the saved score order,
uses the source-validation threshold recorded with the trained LOSO model, and
then overlays pharmacology membership.  It compares target enrichment in the
observed-essential and LOSO-predicted-essential subsets.  Target annotations
are read only after scores and thresholds have been loaded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "loso_full_workflow_target_validation_v56"
MANIFEST = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "clade" / "protozoa"
LOSO = ROOT / "eukaryote_model_results" / "leakage_safe_clade_loso_v46" / "protozoa"
TARGET_REGISTRY = Path(r"D:\ProtoEss_Reviewer_Inputs_v46\eukaryote_model_results\reviewer_context\target_registry_members.tsv")
TARGET_ROLES = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "three_species_pharmacology_registry_v1" / "pharmacology_unique_gene_registry_with_frozen_LOSO.tsv"

SEED = 5601
BOOTSTRAPS = 5_000
FULL_NAMES = {
    "Pf": "Plasmodium_falciparum",
    "Pb": "Plasmodium_berghei",
    "Pk": "Plasmodium_knowlesi",
    "Tg": "Toxoplasma_gondii",
}
REGISTRY_NAMES = {
    "Pf": "Plasmodium_falciparum_3D7",
    "Pb": "Plasmodium_berghei_ANKA",
    "Pk": "Plasmodium_knowlesi_H",
}


def load_held_metadata(species: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(MANIFEST / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str})
        for split in ("train", "val", "test")
    ]
    held = pd.concat(
        [frame.loc[frame.species.astype(str).eq(species)] for frame in frames],
        ignore_index=True,
    )
    return held[["record_id", "species", "gene_id", "label", "protein_length"]].copy()


def h_a_or_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    """Haldane-Anscombe odds ratio and Wald 95% interval."""
    aa, bb, cc, dd = (float(value) + 0.5 for value in (a, b, c, d))
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return math.exp(log_or), math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se), log_or


def enrichment_row(frame: pd.DataFrame, status_col: str) -> dict[str, float | int]:
    status = frame[status_col].astype(bool)
    target = frame["is_target"].astype(bool)
    a = int((status & target).sum())
    b = int((status & ~target).sum())
    c = int((~status & target).sum())
    d = int((~status & ~target).sum())
    odds, ci_low, ci_high, log_or = h_a_or_ci(a, b, c, d)
    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "target_and_essential": a,
        "nontarget_and_essential": b,
        "target_and_nonessential": c,
        "nontarget_and_nonessential": d,
        "target_n": int(target.sum()),
        "target_essential_fraction": a / int(target.sum()) if target.any() else np.nan,
        "haldane_odds_ratio": odds,
        "or_ci95_low": ci_low,
        "or_ci95_high": ci_high,
        "log_haldane_odds_ratio": log_or,
        "fisher_odds_ratio": float(fisher_or),
        "fisher_p_greater": float(fisher_p),
    }


def paired_log_or_bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap the predicted-minus-observed log odds ratio on matched rows."""
    target = frame.loc[frame.is_target.astype(bool)].reset_index(drop=True)
    background = frame.loc[~frame.is_target.astype(bool)].reset_index(drop=True)
    if target.empty or background.empty:
        return {"bootstrap_replicates": 0, "log_or_difference": np.nan, "difference_ci95_low": np.nan,
                "difference_ci95_high": np.nan, "bootstrap_p_two_sided": np.nan}

    observed = enrichment_row(frame, "observed_essential")["log_haldane_odds_ratio"]
    predicted = enrichment_row(frame, "loso_predicted_essential")["log_haldane_odds_ratio"]
    differences = np.empty(BOOTSTRAPS, dtype=float)
    n_target, n_background = len(target), len(background)
    for iteration in range(BOOTSTRAPS):
        sample = pd.concat(
            [
                target.iloc[rng.integers(0, n_target, size=n_target)],
                background.iloc[rng.integers(0, n_background, size=n_background)],
            ],
            ignore_index=True,
        )
        differences[iteration] = (
            enrichment_row(sample, "loso_predicted_essential")["log_haldane_odds_ratio"]
            - enrichment_row(sample, "observed_essential")["log_haldane_odds_ratio"]
        )
    return {
        "bootstrap_replicates": BOOTSTRAPS,
        "log_or_difference": float(predicted - observed),
        "difference_ci95_low": float(np.quantile(differences, 0.025)),
        "difference_ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_p_two_sided": float(2 * min((differences <= 0).mean(), (differences >= 0).mean())),
    }


def performance_row(short: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    y_true = frame.observed_essential.astype(int).to_numpy()
    score = frame.loso_score.to_numpy(float)
    y_pred = frame.loso_predicted_essential.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "species_short": short,
        "species": FULL_NAMES[short],
        "held_labelled_genes": int(len(frame)),
        "threshold_from_source_validation": float(frame.validation_fixed_loso_threshold.iloc[0]),
        "AUROC": float(roc_auc_score(y_true, score)),
        "AUPRC": float(average_precision_score(y_true, score)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def target_sets(registry: pd.DataFrame, role_table: pd.DataFrame, short: str) -> list[tuple[str, set[str]]]:
    """Return a direct-target set where available and the full Tier 1+2 set."""
    species = REGISTRY_NAMES[short]
    all_targets = set(
        registry.loc[
            registry.species.astype(str).eq(species) & registry.target_supported.astype(str).str.lower().eq("true"),
            "gene_id",
        ].astype(str)
    )
    direct = set(
        role_table.loc[
            role_table.species.astype(str).eq(species) & role_table.evidence_role.astype(str).eq("primary_direct_target"),
            "gene_id",
        ].astype(str)
    ) & all_targets
    output: list[tuple[str, set[str]]] = [("Tier 1+2 target-supported registry", all_targets)]
    if direct:
        output.insert(0, ("Tier 1 direct targets", direct))
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(TARGET_REGISTRY, sep="\t", dtype=str)
    role_table = pd.read_csv(TARGET_ROLES, sep="\t", dtype=str)
    rng = np.random.default_rng(SEED)

    gene_metrics: list[dict[str, float | int | str]] = []
    enrichments: list[dict[str, object]] = []
    member_audits: list[pd.DataFrame] = []
    cumulative_rows: list[dict[str, object]] = []
    target_coverage: list[dict[str, object]] = []

    for short, species in FULL_NAMES.items():
        held = load_held_metadata(species)
        probability = np.load(LOSO / species / "test_prob.npy")
        metrics = json.loads((LOSO / species / "metrics.json").read_text(encoding="utf-8"))
        if len(held) != len(probability):
            raise RuntimeError(f"{short}: held metadata and saved score vector do not align.")
        held["loso_score"] = probability.astype(float)
        held["validation_fixed_loso_threshold"] = float(metrics["threshold"])
        held["observed_essential"] = held.label.astype(str).eq("essential")
        held["loso_predicted_essential"] = held.loso_score.ge(held.validation_fixed_loso_threshold)
        held["source_mode"] = metrics.get("source_mode", "unknown")
        held["held_species_rows_used_for_model_training"] = False
        gene_metrics.append(performance_row(short, held))

        if short == "Tg":
            held["target_set"] = "No harmonised target registry used for enrichment analysis"
            held["is_target"] = False
            member_audits.append(held)
            continue

        for target_set, members in target_sets(registry, role_table, short):
            current = held.copy()
            current["target_set"] = target_set
            current["is_target"] = current.gene_id.astype(str).isin(members)
            current["target_annotation_used_for_model_fitting"] = False
            target_coverage.append({
                "species_short": short,
                "species": species,
                "target_set": target_set,
                "registry_members": int(len(members)),
                "members_mapped_to_held_labelled_rows": int(current.is_target.sum()),
                "held_labelled_rows": int(len(current)),
            })
            observed = enrichment_row(current, "observed_essential")
            predicted = enrichment_row(current, "loso_predicted_essential")
            bootstrap = paired_log_or_bootstrap(current, rng)
            for state, stats in (("Observed essentiality", observed), ("LOSO predicted essentiality", predicted)):
                enrichments.append({
                    "species_short": short,
                    "species": species,
                    "target_set": target_set,
                    "essentiality_state": state,
                    **stats,
                    **bootstrap,
                    "threshold_from_source_validation": float(metrics["threshold"]),
                    "held_species_labels_used_for_training": False,
                    "target_annotation_used_for_fitting_or_threshold_selection": False,
                })

            # The curve uses the broad Tier 1+2 registry only, avoiding a sparse
            # direct-only line in the cross-species panel.
            if target_set == "Tier 1+2 target-supported registry":
                ranked = current.sort_values("loso_score", ascending=False, kind="mergesort").reset_index(drop=True)
                target_n = int(ranked.is_target.sum())
                for fraction in np.arange(0.10, 1.01, 0.10):
                    top_n = int(math.ceil(len(ranked) * float(fraction)))
                    top = ranked.iloc[:top_n]
                    hits = int(top.is_target.sum())
                    a, b = hits, int(top_n - hits)
                    c, d = int(target_n - hits), int(len(ranked) - top_n - (target_n - hits))
                    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
                    cumulative_rows.append({
                        "species_short": short,
                        "species": species,
                        "top_fraction": float(fraction),
                        "held_labelled_genes": int(len(ranked)),
                        "mapped_target_genes": target_n,
                        "target_hits": hits,
                        "target_recall": hits / target_n if target_n else np.nan,
                        "fisher_odds_ratio": float(fisher_or),
                        "fisher_p_greater": float(fisher_p),
                    })
            member_audits.append(current)

    metrics_frame = pd.DataFrame(gene_metrics)
    enrich_frame = pd.DataFrame(enrichments)
    member_frame = pd.concat(member_audits, ignore_index=True)
    curve_frame = pd.DataFrame(cumulative_rows)
    coverage_frame = pd.DataFrame(target_coverage)

    metrics_frame.to_csv(OUT / "loso_gene_level_performance.tsv", sep="\t", index=False)
    enrich_frame.to_csv(OUT / "loso_observed_vs_predicted_target_enrichment.tsv", sep="\t", index=False)
    member_frame.to_csv(OUT / "loso_target_member_audit.tsv", sep="\t", index=False)
    curve_frame.to_csv(OUT / "loso_target_cumulative_recovery.tsv", sep="\t", index=False)
    coverage_frame.to_csv(OUT / "loso_target_registry_coverage.tsv", sep="\t", index=False)
    manifest = {
        "analysis": "frozen LOSO full-workflow essentiality-to-target concordance",
        "model_refit": False,
        "score_source": str(LOSO),
        "held_species_labels_used_for_training": False,
        "threshold_source": "source-species validation partition, fixed before held-species evaluation",
        "target_overlay_stage": "after frozen score generation",
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED, "design": "paired target-membership-stratified row bootstrap"},
        "target_sets": {
            "Tier 1": "primary direct targets among mapped Pf registry members",
            "Tier 1+2": "full target-supported three-species registry",
        },
        "Tg_target_enrichment": "not performed because no harmonised, evidence-tiered Tg registry was included in this pre-specified pharmacology analysis",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print(enrich_frame.to_string(index=False))


if __name__ == "__main__":
    main()
