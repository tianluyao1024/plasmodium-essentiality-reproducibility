"""Build the versioned Nature Microbiology submission and release bundle.

This script is intentionally a packaging and audit step. It consumes the
already-frozen five-fold nine-route artifacts, recomputes final metrics from
the saved prediction file, adds the four-species clade-LOSO table (including
Toxoplasma gondii), and writes one manifest with hashes for every published
table. It does not fit a new model or select a model using test labels.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


WORKSPACE = Path(r"C:\Users\tly\Desktop\植物\水稻日本")
FEATURE_ROOT = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
RUN_ROOT = FEATURE_ROOT / "nine_route_v3_5fold_fixed_params_20261016"
FINAL_EVAL = RUN_ROOT / "final_evaluation"
APPLICATION = RUN_ROOT / "full_application"
WP1 = FEATURE_ROOT / "wp1_dualscreen_rebuild_20261016"
LOSO = WORKSPACE / "eukaryote_model_results" / "clade_loso_full_stack" / "clade_loso_full_stack_metrics.tsv"
OUT = Path(r"D:\ProtoEss_NatureMicrobiology_submission_20261017")

SPECIES = [
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
]
LOS0_SPECIES = [
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
    "Toxoplasma_gondii",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def max_mcc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scores = [matthews_corrcoef(y, p >= t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric_row(group: pd.DataFrame, threshold: float) -> dict[str, object]:
    y = group["label"].astype(int).to_numpy()
    p = group["probability"].astype(float).to_numpy()
    pred = p >= threshold
    return {
        "species": group["species"].iloc[0],
        "split": group["split"].iloc[0],
        "n": int(len(group)),
        "n_essential": int(y.sum()),
        "threshold": threshold,
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "MCC": float(matthews_corrcoef(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def rebuild_metrics() -> pd.DataFrame:
    source = FINAL_EVAL / "frozen_stack_predictions.tsv"
    pred = pd.read_csv(source, sep="\t")
    label_parts = pred["record_id"].str.split("|", expand=True)
    pred["label"] = label_parts[1].map({"essential": 1, "nonessential": 0, "non-essential": 0})
    if pred["label"].isna().any():
        raise ValueError("Prediction records do not contain a recognized label token")
    rows = []
    thresholds = {"Plasmodium_falciparum": 0.430, "Plasmodium_berghei": 0.415, "Plasmodium_knowlesi": 0.540}
    for (species, split), group in pred.groupby(["species", "split"], sort=False):
        rows.append(metric_row(group, thresholds[species]))
    return pd.DataFrame(rows)


def standardized_application_table() -> pd.DataFrame:
    source = APPLICATION / "all_three_species_nine_route_genomewide.tsv"
    app = pd.read_csv(source, sep="\t")
    out = pd.DataFrame(
        {
            "species": app["species"],
            "gene_id": app["gene_id"],
            "score": app["nine_route_lightgbm_score"].astype(float),
            "binary_call": np.where(app["predicted_essential_at_training_cv_threshold"], "essential", "non-essential"),
            "threshold": app["training_cv_threshold"].astype(float),
            "model_version": "v20261017_5fold_9route_lightgbm",
            "feature_version": "raw628_v3_repaired_95dedup",
            "screen_condition": "source-screen consensus application; exact screen condition varies by species",
            "evidence_class": "computational_prioritization",
        }
    )
    return out.sort_values(["species", "gene_id"]).reset_index(drop=True)


def build_loso_table() -> pd.DataFrame:
    raw = pd.read_csv(LOSO, sep="\t")
    raw = raw[raw["held_out"].isin(LOS0_SPECIES)].copy()
    raw["analysis"] = "four-species clade leave-one-species-out"
    raw["model_version"] = "historical_clade_full_stack_v46; companion analysis"
    raw["note"] = (
        "Cross-species generalization evaluation; separate from the five-fold nine-route main test set."
    )
    cols = ["analysis", "clade", "held_out", "oof_AUROC", "validation_AUROC", "test_AUROC", "test_AUPRC", "test_F1", "threshold", "model_version", "note"]
    return raw[cols].sort_values("held_out").reset_index(drop=True)


def make_data_dictionary() -> pd.DataFrame:
    rows = [
        ("species", "Species name", "application and metric tables", "Plasmodium or Toxoplasma species label"),
        ("gene_id", "Species-native gene identifier", "genome-wide application table", "Stable source identifier"),
        ("score", "Frozen nine-route LightGBM probability", "genome-wide application table", "Continuous computational score in [0,1]"),
        ("binary_call", "Thresholded computational call", "genome-wide application table", "essential or non-essential"),
        ("threshold", "Training-only classification threshold", "application and metric tables", "Frozen max-MCC threshold"),
        ("model_version", "Model release identifier", "all published tables", "Versioned release label"),
        ("feature_version", "Feature matrix release identifier", "application table", "Raw 628-dimensional repaired matrix"),
        ("screen_condition", "Experimental condition represented by source labels", "application table", "Condition metadata; not an experimental claim"),
        ("evidence_class", "Evidence provenance", "application table", "Computational prioritization only"),
        ("test_AUROC", "Held-out ranking performance", "LOSO table", "LOSO test metric, not main five-fold test"),
        ("Pk dual-screen flip_rate", "Disagreement between two P. knowlesi screens", "WP1 table", "Condition-associated label discordance"),
    ]
    return pd.DataFrame(rows, columns=["field", "definition", "table_scope", "interpretation"])


def main() -> None:
    clean_dir(OUT)
    for sub in ["manifest", "metrics", "predictions", "wp1_dual_screen", "loso", "source_data", "old_vs_new"]:
        clean_dir(OUT / sub)

    metrics = rebuild_metrics()
    metrics.to_csv(OUT / "metrics" / "frozen_fivefold_nine_route_metrics.tsv", sep="\t", index=False)
    app = standardized_application_table()
    app.to_csv(OUT / "predictions" / "standardized_genomewide_predictions.tsv", sep="\t", index=False)

    copy_file(FINAL_EVAL / "frozen_stack_predictions.tsv", OUT / "predictions" / "frozen_stack_predictions.tsv")
    copy_file(FINAL_EVAL / "run_manifest.json", OUT / "manifest" / "fivefold_run_manifest.json")
    copy_file(RUN_ROOT / "homology_scopes" / "run_manifest.json", OUT / "manifest" / "homology_scope_manifest.json")
    copy_file(APPLICATION / "run_manifest.json", OUT / "manifest" / "application_run_manifest.json")
    copy_file(APPLICATION / "all_three_species_nine_route_genomewide.tsv", OUT / "predictions" / "all_three_species_nine_route_genomewide.tsv")

    for name in [
        "WP1_DUALSCREEN_VALIDATED_REPORT.md",
        "run_manifest.json",
        "aligned_dualscreen_gene_calls.tsv",
        "categorical_concordance.tsv",
        "continuous_score_association.tsv",
        "paired_cross_vs_within_pk_comparison.tsv",
        "ortholog_mapping_audit.tsv",
        "model_prediction_exact_triplets_summary.tsv",
        "cross_species_observed_label_discordance.tsv",
        "exact_triplets_observed_label_join.tsv",
    ]:
        copy_file(WP1 / name, OUT / "wp1_dual_screen" / name)

    loso = build_loso_table()
    loso.to_csv(OUT / "loso" / "four_species_clade_loso_metrics.tsv", sep="\t", index=False)
    copy_file(LOSO, OUT / "loso" / "source_clade_loso_full_stack_metrics.tsv")
    make_data_dictionary().to_csv(OUT / "source_data" / "data_dictionary.tsv", sep="\t", index=False)

    # Workbook is deliberately flat and auditable: no values are manually entered.
    workbook = OUT / "source_data" / "ProtoEss_NatureMicrobiology_source_data.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Frozen_metrics", index=False)
        app.to_excel(writer, sheet_name="Genomewide_predictions", index=False)
        loso.to_excel(writer, sheet_name="Four_species_LOSO", index=False)
        pd.read_csv(WP1 / "categorical_concordance.tsv", sep="\t").to_excel(writer, sheet_name="Pk_categorical", index=False)
        pd.read_csv(WP1 / "continuous_score_association.tsv", sep="\t").to_excel(writer, sheet_name="Pk_continuous", index=False)
        pd.read_csv(WP1 / "paired_cross_vs_within_pk_comparison.tsv", sep="\t").to_excel(writer, sheet_name="Pk_paired_comparison", index=False)
        make_data_dictionary().to_excel(writer, sheet_name="Data_dictionary", index=False)

    copied = [p for p in OUT.rglob("*") if p.is_file() and p.name != "final_freeze_manifest.json"]
    file_hashes = {str(p.relative_to(OUT)).replace("\\", "/"): sha256(p) for p in copied}
    source_manifest = json.loads((FINAL_EVAL / "run_manifest.json").read_text(encoding="utf-8"))
    wp1_manifest = json.loads((WP1 / "run_manifest.json").read_text(encoding="utf-8"))
    freeze = {
        "release_id": "v20261017_5fold_9route_lightgbm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_model": "five-fold nine-route LightGBM",
        "feature_count": 628,
        "feature_policy": "raw matrix values; no PCA or manual removal of homology30/50",
        "split_policy": "original 95%-deduplicated label-stratified random train/validation/test split",
        "test_boundary": source_manifest["test_labels"],
        "historical_non_blind_caveat": source_manifest["historical_profile_selection_caveat"],
        "meta_cv_caveat": source_manifest["known_caveat"],
        "main_species": source_manifest["species"],
        "pk_condition_analysis": {
            "label": "retrospective external-condition validation",
            "human_rbc_sequence_level_features_available": False,
            "interpretation": "ortholog-mapped condition analysis; not exact A1H sequence-level prediction",
            "source_manifest": wp1_manifest,
        },
        "loso_species_included": LOS0_SPECIES,
        "loso_note": "Toxoplasma gondii is included as the fourth held-out species in the companion clade-LOSO analysis.",
        "published_files_sha256": file_hashes,
    }
    (OUT / "manifest" / "final_freeze_manifest.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# ProtoEss Nature Microbiology release bundle\n\nRelease: `v20261017_5fold_9route_lightgbm`\n\nThis bundle is generated from the frozen five-fold nine-route LightGBM artifacts. The main model uses the original 628 raw features after 95% redundancy reduction and the original stratified random train/validation/test split. No PCA or manual removal of `homology30/50` is applied.\n\nThe four-species clade leave-one-species-out analysis includes **Plasmodium falciparum, P. berghei, P. knowlesi and Toxoplasma gondii**. It is a companion cross-species generalization analysis and is not the same evaluation as the main within-species five-fold test.\n\nThe P. knowlesi human-red-cell analysis is reported as an ortholog-mapped retrospective external-condition analysis. A complete A1H sequence-level feature matrix was not available in the audited input directories; therefore the bundle does not claim exact A1H sequence-level prediction.\n\nThe authoritative hash list is `manifest/final_freeze_manifest.json`.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "files": len(file_hashes), "loso_species": LOS0_SPECIES}, indent=2))


if __name__ == "__main__":
    main()
