"""Tune original-config LightGBM candidates on the corrected matrix using training CV only."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

import run_reproduced_stack_training_cv_tuning as cv  # noqa: E402
import run_repeated_paired_training_cv as safe_cv  # noqa: E402


OUT = Path(os.environ.get(
    "PROTOESS_LGBM_CV_OUT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\corrected_matrix_lgbm_training_cv_20261016",
))
CANDIDATE_NAMES = {
    "Plasmodium_falciparum": ("original_config", "bagging0.7"),
    "Plasmodium_berghei": ("original_config", "bagging0.7", "bagging0.85"),
    "Plasmodium_knowlesi": ("original_config", "bagging0.85"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_candidates = cv.prior_tuning.candidate_grid()
    seed_rows: list[dict] = []
    fold_rows: list[dict] = []
    summaries: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    for species in cv.SPECIES:
        print(f"[{species}] training-only repeated CV", flush=True)
        candidate_names = CANDIDATE_NAMES[species]
        candidates = [c for c in all_candidates if c["candidate"] in candidate_names]
        data = safe_cv.load_species_training(species)
        seed42_oof: dict[str, np.ndarray] = {}
        for seed in cv.SEEDS:
            oof, _ = cv.one_seed_cv(data, species, seed, candidates, fold_rows, seed_rows)
            if seed == 42:
                seed42_oof = oof
        seed_metrics = pd.DataFrame([row for row in seed_rows if row["species"] == species])
        winner, candidate_summary = cv.select_candidate(seed_metrics, candidates, species)
        summaries.append(candidate_summary.assign(species=species))
        winner_name = str(winner["candidate"])
        if winner_name not in seed42_oof:
            raise RuntimeError(f"{species}: selected seed-42 OOF predictions were not retained")

        train_oof = seed42_oof[winner_name]
        threshold = cv.shared.best_oof_threshold(data["y"], train_oof)
        full = cv.build_full_train_validation(species)
        model = cv.make_lgbm(winner, 42)
        model.fit(full["X"]["train"], full["y"]["train"])
        p_val = model.predict_proba(full["X"]["val"])[:, 1].astype(np.float32)
        y_val = full["y"]["val"]
        selected_rows.append({
            "species": species,
            "selected_candidate": winner_name,
            "selected_params": json.dumps({k: v for k, v in winner.items() if k != "candidate"}, sort_keys=True),
            "selection_rule": "1-SE on mean pooled OOF AUPRC across five repeated stratified 3-fold CVs; ties prefer the original LightGBM configuration",
            "cv_mean_pooled_OOF_AUPRC": float(candidate_summary.loc[candidate_summary.candidate.eq(winner_name), "mean_pooled_OOF_AUPRC"].iloc[0]),
            "cv_sd_pooled_OOF_AUPRC": float(candidate_summary.loc[candidate_summary.candidate.eq(winner_name), "sd_pooled_OOF_AUPRC"].iloc[0]),
            "original_cv_mean_pooled_OOF_AUPRC": float(candidate_summary.loc[candidate_summary.candidate.eq("original_config"), "mean_pooled_OOF_AUPRC"].iloc[0]),
            "train_oof_AUROC_seed42": float(roc_auc_score(data["y"], train_oof)),
            "train_oof_AUPRC_seed42": float(average_precision_score(data["y"], train_oof)),
            "validation_AUROC_diagnostic": float(roc_auc_score(y_val, p_val)),
            "validation_AUPRC_diagnostic": float(average_precision_score(y_val, p_val)),
            "validation_MCC_train_oof_threshold_diagnostic": float(matthews_corrcoef(y_val, p_val >= threshold)),
            "threshold_from_training_oof": float(threshold),
            "n_train": int(len(data["y"])),
            "n_validation": int(len(y_val)),
            "n_features": 628,
        })
        pd.DataFrame({
            "record_id": data["ids"],
            "label": data["y"],
            "probability_oof_seed42": train_oof,
            "threshold_from_training_oof": threshold,
        }).to_csv(prediction_dir / f"{species}_train_oof.tsv", sep="\t", index=False)
        pd.DataFrame({
            "record_id": full["ids"]["val"],
            "label": y_val,
            "probability": p_val,
            "threshold_from_training_oof": threshold,
        }).to_csv(prediction_dir / f"{species}_validation_diagnostic.tsv", sep="\t", index=False)
        model.booster_.save_model(str(OUT / f"{species}_selected_lightgbm.txt"))
        print(
            f"[{species}] selected={winner_name}; CV AP="
            f"{selected_rows[-1]['cv_mean_pooled_OOF_AUPRC']:.4f}; validation AP diagnostic="
            f"{selected_rows[-1]['validation_AUPRC_diagnostic']:.4f}",
            flush=True,
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "candidate_cv_summary.tsv", sep="\t", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "candidate_seed_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "candidate_fold_metrics.tsv", sep="\t", index=False)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / "selected_models_and_validation_diagnostics.tsv", sep="\t", index=False)
    (OUT / "run_manifest.json").write_text(json.dumps({
        "experiment": "LightGBM training-only hyperparameter selection on corrected original feature matrices",
        "feature_root": str(cv.shared.REPAIRED_FEATURES),
        "split": "frozen within-species 95%-deduplicated, label-stratified random split; no resplitting",
        "feature_schema": "original 628 raw-scale columns; preserve all original static values and four label-free homology size/species-count columns",
        "candidate_names_by_species": CANDIDATE_NAMES,
        "cv": {"type": "repeated stratified 3-fold", "seeds": list(cv.SEEDS)},
        "dynamic_feature_handling": "26 label-derived homology features, 21 MMseqs features, and 15 PPI features rebuilt fold-locally; final validation features use train labels only",
        "validation_used_for_candidate_or_threshold_selection": False,
        "validation_role": "post-selection diagnostic only; previously inspected in this project",
        "test_metadata_arrays_labels_or_predictions_read": False,
        "model": "single LightGBM, original manuscript configuration or 1-SE eligible candidate closest to it",
        "outputs": str(OUT),
        "selected_models": selected_rows,
    }, indent=2), encoding="utf-8")
    lines = [
        "# Corrected-Matrix LightGBM Training-CV Results",
        "",
        "The paired comparison uses only the frozen training rows with five repeated stratified three-fold CV. It includes the original LightGBM configuration and the candidates selected in the previous training-only search; the Pb best-mean and one-SE candidates are both included. All 628 original features are retained; label-derived columns are rebuilt within each fold. Validation scores are post-selection diagnostics, not used to choose candidates or thresholds. The test split is untouched.",
        "",
        "| Species | Selected candidate | Train-CV AP mean | Original-config Train-CV AP | Seed-42 OOF AP | Validation AP diagnostic | Validation AUROC diagnostic |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['species']} | {row['selected_candidate']} | {row['cv_mean_pooled_OOF_AUPRC']:.4f} | "
            f"{row['original_cv_mean_pooled_OOF_AUPRC']:.4f} | {row['train_oof_AUPRC_seed42']:.4f} | "
            f"{row['validation_AUPRC_diagnostic']:.4f} | {row['validation_AUROC_diagnostic']:.4f} |"
        )
    lines.extend(["", f"Detailed candidate scores: `{OUT / 'candidate_cv_summary.tsv'}`", f"Selected models: `{OUT / 'selected_models_and_validation_diagnostics.tsv'}`"])
    (OUT / "CORRECTED_LGBM_CV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed: {OUT}", flush=True)


if __name__ == "__main__":
    main()
