"""Evaluate the frozen nine-probability LightGBM meta-stack in all species.

Base-model outputs and candidate configurations are reused from training-only
artifacts. The held-out validation/test sets are used only for final scoring.
"""
from __future__ import annotations

import json
import sys
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

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
CODE = REPO / "formal_reproduction" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import tune_oof_probability_lightgbm_meta_20261016 as meta  # noqa: E402

DATA = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
SCOPE_FILE = DATA / "homology_scope_with_oberstaller_20261016" / "scope_comparison_predictions.tsv"
SOURCE_RUN = DATA / "corrected_raw_foldmatched_meta_20261016"
FINAL_CACHE = DATA / "corrected_raw_final_frozen_evaluation_20261016" / "prediction_cache"
FEATURES = DATA / "scope_feature_sets_v2" / "within_species"
MANIFESTS = WORKSPACE / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species"
REPRESENTATIVES = SOURCE_RUN / "meta" / "training_cv_profile_ensemble" / "profile_representatives.tsv"
OUTPUT = DATA / "all_species_nine_route_final_evaluation_20261016"
PROFILE = "combined_nine_base_probabilities"
BRANCH_COUNT = 5


def threshold_from_training_cv(y: np.ndarray, repeat_oof: np.ndarray) -> float:
    mean_oof = repeat_oof.mean(axis=0)
    grid = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y, mean_oof >= value) for value in grid]
    return float(grid[int(np.argmax(scores))])


def metric_row(species: str, split: str, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {
        "species": species,
        "split": split,
        "n": int(len(y)),
        "n_essential": int(y.sum()),
        "prevalence": float(y.mean()),
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold_from_nine_route_training_CV": threshold,
        "MCC": float(matthews_corrcoef(y, p >= threshold)),
        "F1": float(f1_score(y, p >= threshold, zero_division=0)),
        "profile": PROFILE,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scopes = pd.read_csv(SCOPE_FILE, sep="\t", dtype={"record_id": str})
    representatives = pd.read_csv(REPRESENTATIVES, sep="\t", header=None)
    representatives.columns = ["species", "profile", "candidate", "cv_auroc", "cv_auprc", "cv_score"]
    representatives = representatives.loc[representatives.profile == PROFILE].set_index("species")

    metric_rows: list[dict] = []
    cv_rows: list[dict] = []
    prediction_rows: list[dict] = []
    oof_rows: list[dict] = []
    run_species: dict[str, dict] = {}

    for species in meta.SPECIES:
        train_meta = pd.read_csv(MANIFESTS / species / "train_metadata.tsv", sep="\t", dtype={"record_id": str})
        val_meta = pd.read_csv(MANIFESTS / species / "val_metadata.tsv", sep="\t", dtype={"record_id": str})
        test_meta = pd.read_csv(MANIFESTS / species / "test_metadata.tsv", sep="\t", dtype={"record_id": str})

        train_ids, y_train, scope_oof, scope_names = meta.align_scope(scopes, species, "train")
        if tuple(scope_names) != tuple(meta.ALL_SCOPE_NAMES):
            raise ValueError(f"{species}: expected four scope columns, got {scope_names}")
        if not np.array_equal(train_ids, train_meta.record_id.astype(str).to_numpy()):
            raise ValueError(f"{species}: training OOF IDs differ from frozen manifest")
        if not np.array_equal(y_train, train_meta.label.eq("essential").astype(np.int8).to_numpy()):
            raise ValueError(f"{species}: training labels differ from frozen manifest")

        branch_oof = np.asarray(np.load(SOURCE_RUN / "five_branch" / species / "oof_branches.npy"), dtype=np.float32)
        if branch_oof.shape != (len(train_ids), BRANCH_COUNT):
            raise ValueError(f"{species}: unexpected five-branch OOF shape {branch_oof.shape}")
        x_train = np.column_stack([scope_oof, branch_oof]).astype(np.float32)

        matrix_dir = FEATURES / species
        cache = np.load(FINAL_CACHE / species / "full_refit_base_probabilities.npz")
        x_val = np.column_stack([cache["val_scopes"], cache["val_branches"]]).astype(np.float32)
        x_test = np.column_stack([cache["test_scopes"], cache["test_branches"]]).astype(np.float32)
        for split, expected in (("val", val_meta), ("test", test_meta)):
            matrix_ids = np.load(matrix_dir / f"ids_{split}.npy", allow_pickle=True).astype(str)
            if not np.array_equal(matrix_ids, expected.record_id.astype(str).to_numpy()):
                raise ValueError(f"{species}/{split}: matrix IDs differ from frozen manifest")
        if x_val.shape[0] != len(val_meta) or x_test.shape[0] != len(test_meta):
            raise ValueError(f"{species}: cached probability rows do not match frozen manifests")
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all() or not np.isfinite(x_test).all():
            raise ValueError(f"{species}: non-finite meta-input probability")

        candidate_name = str(representatives.loc[species, "candidate"])
        candidate = next((row for row in meta.CANDIDATES if row["name"] == candidate_name), None)
        if candidate is None:
            raise ValueError(f"{species}: unknown training-CV candidate {candidate_name}")

        repeat_oof, cv_metrics, _ = meta.cv_predictions(x_train, y_train, candidate, meta.CV, 5, 3)
        threshold = threshold_from_training_cv(y_train, repeat_oof)
        final_meta = meta.make_model(candidate, 42).fit(x_train, y_train)

        for repeat_row in cv_metrics.to_dict(orient="records"):
            cv_rows.append({"species": species, "profile": PROFILE, "candidate": candidate_name, **repeat_row})
        for rid, label, prob in zip(train_ids, y_train, repeat_oof.mean(axis=0)):
            oof_rows.append({"species": species, "record_id": rid, "label": int(label), "probability": float(prob)})

        for split, frame, matrix in (("validation", val_meta, x_val), ("test", test_meta, x_test)):
            y = frame.label.eq("essential").astype(np.int8).to_numpy()
            probabilities = final_meta.predict_proba(matrix)[:, 1]
            metric_rows.append(metric_row(species, split, y, probabilities, threshold))
            prediction_rows.extend(
                {"species": species, "split": split, "record_id": rid, "label": int(label), "probability": float(prob)}
                for rid, label, prob in zip(frame.record_id.astype(str), y, probabilities)
            )

        run_species[species] = {
            "profile": PROFILE,
            "input_count": 9,
            "input_order": list(meta.ALL_SCOPE_NAMES) + ["lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer"],
            "candidate_selected_from_training_cv_profile_representatives": candidate_name,
            "threshold_from_training_only_repeated_CV": threshold,
            "n_train": int(len(train_meta)),
            "n_validation": int(len(val_meta)),
            "n_test": int(len(test_meta)),
        }

    pd.DataFrame(metric_rows).to_csv(OUTPUT / "all_nine_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTPUT / "training_cv_repeat_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(oof_rows).to_csv(OUTPUT / "training_oof_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / "validation_test_predictions.tsv", sep="\t", index=False)
    manifest = {
        "purpose": "All-species nine-route final evaluation using training-CV profile representatives.",
        "profile": PROFILE,
        "input_count": 9,
        "base_probabilities": "four homology-label-scope LightGBM predictions plus five model-branch predictions",
        "meta_model": "LightGBM; candidate selected from training-only profile representatives; final refit on training OOF probabilities",
        "threshold": "MCC-maximizing threshold from repeated training-only OOF probabilities",
        "base_probability_source": str(FINAL_CACHE),
        "training_oof_source": str(SOURCE_RUN),
        "validation_test_labels_used_only_for_final_metrics": True,
        "historical_caveat": "The test split was accessed in earlier project work. The decision to use nine routes for all species followed inspection of P. knowlesi nine-route test metrics; therefore the test scores are post-hoc and not an untouched blind estimate.",
        "known_caveat": "Meta-level training CV uses precomputed base OOF probabilities and is not fully nested.",
        "species": run_species,
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nWrote all-nine evaluation to: {OUTPUT}")


if __name__ == "__main__":
    main()
