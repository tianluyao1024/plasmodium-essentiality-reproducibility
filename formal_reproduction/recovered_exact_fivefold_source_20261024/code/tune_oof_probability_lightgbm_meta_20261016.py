"""Training-OOF-only tuning of a small LightGBM meta-learner.

Input columns are out-of-fold base probabilities from corrected five-branch
and homology-scope LightGBM models. Validation is scored only after
configuration selection; test files are never opened.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


SCOPE_PREDICTIONS = Path(os.environ.get(
    "PROTOESS_HOMOLOGY_SCOPE_INPUT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\plasmodium_only_label_assisted_20261016\scope_comparison_predictions.tsv",
))
FIVE_BRANCH_ROOT = Path(os.environ.get(
    "PROTOESS_FIVE_BRANCH_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\model_five_branch\within_species",
))
OUTPUT = Path(os.environ.get(
    "PROTOESS_META_LGBM_OUT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\oof_lightgbm_meta_tuning_20261016",
))
MANIFEST_ROOT = Path.cwd() / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species"
SPECIES = (
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
)
ALL_SCOPE_NAMES = (
    "strict_within_species",
    "plasmodium_only_label_assisted",
    "cross_species_label_assisted",
    "cross_species_with_pk_oberstaller_labels",
)
CANDIDATES = (
    {"name": "leaves3_child20_trees100", "num_leaves": 3, "min_child_samples": 20, "n_estimators": 100, "learning_rate": 0.03, "reg_lambda": 10.0},
    {"name": "leaves5_child20_trees200", "num_leaves": 5, "min_child_samples": 20, "n_estimators": 200, "learning_rate": 0.03, "reg_lambda": 10.0},
    {"name": "leaves7_child30_trees200", "num_leaves": 7, "min_child_samples": 30, "n_estimators": 200, "learning_rate": 0.03, "reg_lambda": 15.0},
    {"name": "leaves3_child30_trees300", "num_leaves": 3, "min_child_samples": 30, "n_estimators": 300, "learning_rate": 0.02, "reg_lambda": 5.0},
    {"name": "leaves5_child30_trees100", "num_leaves": 5, "min_child_samples": 30, "n_estimators": 100, "learning_rate": 0.05, "reg_lambda": 5.0},
    {"name": "leaves7_child20_trees300", "num_leaves": 7, "min_child_samples": 20, "n_estimators": 300, "learning_rate": 0.03, "reg_lambda": 10.0},
    {"name": "leaves11_child30_trees300", "num_leaves": 11, "min_child_samples": 30, "n_estimators": 300, "learning_rate": 0.02, "reg_lambda": 20.0},
    {"name": "leaves3_child10_trees500", "num_leaves": 3, "min_child_samples": 10, "n_estimators": 500, "learning_rate": 0.01, "reg_lambda": 10.0},
)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5601)
CONFIRM_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=8261)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_scope(frame: pd.DataFrame, species: str, split: str):
    source_split = "train_oof" if split == "train" else "validation"
    part_all = frame.loc[(frame.species == species) & (frame.split == source_split)]
    ref_ids = None
    y = None
    columns = []
    available = set(part_all.scope.astype(str).unique())
    scope_names = tuple(name for name in ALL_SCOPE_NAMES if name in available)
    if len(scope_names) not in (3, 4):
        raise ValueError(f"{species}/{source_split}: expected 3 or 4 scope columns, found {scope_names}")
    for scope in scope_names:
        part = part_all.loc[part_all.scope == scope, ["record_id", "label", "probability"]]
        if part.empty or part.record_id.astype(str).duplicated().any():
            raise ValueError(f"{species}/{source_split}/{scope}: missing or duplicate IDs")
        part = part.assign(record_id=part.record_id.astype(str)).set_index("record_id")
        if ref_ids is None:
            ref_ids = part.index.to_numpy(dtype=str)
            y = part.label.astype(int).to_numpy()
        part = part.reindex(ref_ids)
        if part.label.isna().any() or not np.array_equal(part.label.astype(int).to_numpy(), y):
            raise ValueError(f"{species}/{source_split}/{scope}: ID/label mismatch")
        columns.append(part.probability.to_numpy(dtype=np.float32))
    return ref_ids, y, np.column_stack(columns), scope_names


def make_model(candidate: dict, seed: int):
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        n_jobs=int(os.environ.get("PROTOESS_LGBM_N_JOBS", "4")),
        verbosity=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        **{key: value for key, value in candidate.items() if key != "name"},
    )


def cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    candidate: dict,
    splitter,
    n_splits: int,
    n_repeats: int,
):
    split_predictions: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int]] = []
    for fold_index, (fit_ix, hold_ix) in enumerate(splitter.split(X, y)):
        model = make_model(candidate, 42 + fold_index)
        model.fit(X[fit_ix], y[fit_ix])
        prediction = model.predict_proba(X[hold_ix])[:, 1].astype(np.float32)
        split_predictions.append(np.column_stack([hold_ix, prediction]))
        fold_rows.append({
            "fold": fold_index,
            "repeat": fold_index // 5,
            "n": int(len(hold_ix)),
            "AUROC": float(roc_auc_score(y[hold_ix], prediction)),
            "AUPRC": float(average_precision_score(y[hold_ix], prediction)),
        })
    repeat_oof = np.zeros((n_repeats, len(y)), dtype=np.float32)
    for fold in range(n_splits * n_repeats):
        block = split_predictions[fold]
        indices = block[:, 0].astype(int)
        repeat_oof[fold // n_splits, indices] = block[:, 1]
    repeat_rows = []
    for repeat in range(n_repeats):
        repeat_rows.append({
            "repeat": repeat,
            "AUROC": float(roc_auc_score(y, repeat_oof[repeat])),
            "AUPRC": float(average_precision_score(y, repeat_oof[repeat])),
        })
    return repeat_oof, pd.DataFrame(repeat_rows), fold_rows


def select_one_se(candidate_results: list[dict[str, object]]) -> dict[str, object]:
    best = max(candidate_results, key=lambda row: float(row["mean_repeat_AUPRC"]))
    cutoff = float(best["mean_repeat_AUPRC"]) - float(best["se_repeat_AUPRC"])
    eligible = [row for row in candidate_results if float(row["mean_repeat_AUPRC"]) >= cutoff]
    return min(eligible, key=lambda row: (int(row["num_leaves"]), int(row["n_estimators"])))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y, p >= threshold) for threshold in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    scope_frame = pd.read_csv(SCOPE_PREDICTIONS, sep="\t", dtype={"record_id": str})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv_rows: list[dict[str, object]] = []
    fold_rows_all: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}

    for species in SPECIES:
        species_dir = FIVE_BRANCH_ROOT / species
        oof_path = species_dir / "oof_branches.npy"
        val_path = species_dir / "val_branches.npy"
        branch_oof = np.asarray(np.load(oof_path, mmap_mode="r"), dtype=np.float32)
        branch_val = np.asarray(np.load(val_path, mmap_mode="r"), dtype=np.float32)
        train_ids, y_train, scope_oof, scope_names = align_scope(scope_frame, species, "train")
        val_ids, y_val, scope_val, val_scope_names = align_scope(scope_frame, species, "val")
        if scope_names != val_scope_names:
            raise ValueError(f"{species}: train and validation scope columns differ")
        train_manifest = pd.read_csv(MANIFEST_ROOT / species / "train_metadata.tsv", sep="\t", dtype={"record_id": str})
        val_manifest = pd.read_csv(MANIFEST_ROOT / species / "val_metadata.tsv", sep="\t", dtype={"record_id": str})
        if not np.array_equal(train_ids, train_manifest.record_id.astype(str).to_numpy()) or not np.array_equal(val_ids, val_manifest.record_id.astype(str).to_numpy()):
            raise ValueError(f"{species}: predictions do not align to frozen train/validation manifests")
        if not np.array_equal(y_train, train_manifest.label.eq("essential").astype(int).to_numpy()) or not np.array_equal(y_val, val_manifest.label.eq("essential").astype(int).to_numpy()):
            raise ValueError(f"{species}: labels differ from frozen train/validation manifests")
        if branch_oof.shape != (len(train_ids), 5) or branch_val.shape != (len(val_ids), 5):
            raise ValueError(f"{species}: five-branch prediction shape mismatch")
        n_scopes = len(scope_names)
        branch_ix = tuple(range(n_scopes, n_scopes + 5))
        X_train_all = np.column_stack([scope_oof, branch_oof])
        X_val_all = np.column_stack([scope_val, branch_val])
        profiles = {
            "corrected_five_branch": branch_ix,
            "three_scope_lgbm": (0, 1, 2),
            ("combined_eight_base_probabilities" if n_scopes == 3
             else "combined_eight_without_oberstaller"): (0, 1, 2) + branch_ix,
        }
        if n_scopes == 4:
            profiles["four_scope_lgbm"] = (0, 1, 2, 3)
            profiles["combined_nine_base_probabilities"] = tuple(range(9))

        for profile_name, indices in profiles.items():
            X = X_train_all[:, indices]
            X_val = X_val_all[:, indices]
            summaries: list[dict[str, object]] = []
            candidate_predictions: dict[str, np.ndarray] = {}
            for candidate in CANDIDATES:
                repeat_oof, repeat_metrics, fold_metrics = cv_predictions(X, y_train, candidate, CV, 5, 3)
                mean_ap = float(repeat_metrics.AUPRC.mean())
                se_ap = float(repeat_metrics.AUPRC.std(ddof=1) / np.sqrt(len(repeat_metrics)))
                row = {
                    "species": species,
                    "profile": profile_name,
                    "candidate": candidate["name"],
                    **{key: value for key, value in candidate.items() if key != "name"},
                    "mean_repeat_AUPRC": mean_ap,
                    "se_repeat_AUPRC": se_ap,
                    "mean_repeat_AUROC": float(repeat_metrics.AUROC.mean()),
                    "se_repeat_AUROC": float(repeat_metrics.AUROC.std(ddof=1) / np.sqrt(len(repeat_metrics))),
                }
                summaries.append(row)
                candidate_predictions[candidate["name"]] = repeat_oof
                for repeat_row in repeat_metrics.to_dict(orient="records"):
                    fold_rows_all.append({"species": species, "profile": profile_name, "candidate": candidate["name"], **repeat_row})
            cv_rows.extend(summaries)
            winner = select_one_se(summaries)
            candidate = next(item for item in CANDIDATES if item["name"] == winner["candidate"])
            confirmation_oof, confirmation_metrics, _ = cv_predictions(X, y_train, candidate, CONFIRM_CV, 5, 2)
            train_oof_mean = confirmation_oof.mean(axis=0)
            threshold = best_threshold(y_train, train_oof_mean)
            final_model = make_model(candidate, 42)
            final_model.fit(X, y_train)
            val_probability = final_model.predict_proba(X_val)[:, 1]
            validation_rows.append({
                "species": species,
                "profile": profile_name,
                "selected_candidate_one_se": candidate["name"],
                "train_cv_mean_AUPRC": winner["mean_repeat_AUPRC"],
                "train_cv_se_AUPRC": winner["se_repeat_AUPRC"],
                "confirmation_cv_mean_AUPRC": float(confirmation_metrics.AUPRC.mean()),
                "confirmation_cv_se_AUPRC": float(confirmation_metrics.AUPRC.std(ddof=1) / np.sqrt(len(confirmation_metrics))),
                "threshold_from_repeated_train_oof": threshold,
                "validation_AUROC": float(roc_auc_score(y_val, val_probability)),
                "validation_AUPRC": float(average_precision_score(y_val, val_probability)),
                "validation_MCC": float(matthews_corrcoef(y_val, val_probability >= threshold)),
            })
            for i, record_id in enumerate(val_ids):
                predictions.append({
                    "species": species,
                    "profile": profile_name,
                    "record_id": record_id,
                    "label": int(y_val[i]),
                    "probability": float(val_probability[i]),
                    "threshold_from_repeated_train_oof": threshold,
                })
        input_hashes[species] = {
            "oof_branches_sha256": sha256(oof_path),
            "val_branches_sha256": sha256(val_path),
        }

    profile_selection_rows = []
    for species in SPECIES:
        rows = [row for row in validation_rows if row["species"] == species]
        best = max(rows, key=lambda row: row["train_cv_mean_AUPRC"])
        cutoff = best["train_cv_mean_AUPRC"] - best["train_cv_se_AUPRC"]
        eligible = [row for row in rows if row["train_cv_mean_AUPRC"] >= cutoff]
        profile_sizes = {
            "corrected_five_branch": 5,
            "three_scope_lgbm": 3,
            "combined_eight_base_probabilities": 8,
            "four_scope_lgbm": 4,
            "combined_eight_without_oberstaller": 8,
            "combined_nine_base_probabilities": 9,
        }
        selected = min(eligible, key=lambda row: (profile_sizes[row["profile"]], row["profile"]))
        profile_selection_rows.append({
            "species": species,
            "selected_profile_one_se": selected["profile"],
            "best_profile_cv_AUPRC": best["train_cv_mean_AUPRC"],
            "one_se_cutoff": cutoff,
            "selected_profile_train_cv_AUPRC": selected["train_cv_mean_AUPRC"],
            "selected_profile_confirmation_cv_AUPRC": selected["confirmation_cv_mean_AUPRC"],
            "selected_profile_validation_AUROC": selected["validation_AUROC"],
            "selected_profile_validation_AUPRC": selected["validation_AUPRC"],
            "validation_used_for_profile_selection": False,
        })

    pd.DataFrame(cv_rows).to_csv(OUTPUT / "meta_candidate_cv_summary.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows_all).to_csv(OUTPUT / "meta_candidate_repeat_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(validation_rows).to_csv(OUTPUT / "selected_meta_validation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(predictions).to_csv(OUTPUT / "selected_meta_validation_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(profile_selection_rows).to_csv(OUTPUT / "selected_profile_by_training_cv.tsv", sep="\t", index=False)
    manifest = {
        "experiment": "training-only repeated CV tuning of LightGBM meta models over OOF base probabilities",
        "scope_prediction_file": str(SCOPE_PREDICTIONS),
        "scope_prediction_sha256": sha256(SCOPE_PREDICTIONS),
        "five_branch_root": str(FIVE_BRANCH_ROOT),
        "features": "base-model probabilities only; no raw validation features used in candidate selection",
        "scope_names": list(scope_names),
        "candidate_profiles": {key: list(value) for key, value in profiles.items()},
        "candidate_configs": list(CANDIDATES),
        "cv": {"type": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 3, "random_state": 5601},
        "confirmation_cv": {"type": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 2, "random_state": 8261},
        "selection": "one-standard-error rule over mean repeated train-OOF AUPRC; simpler leaves/tree count preferred",
        "threshold": "MCC grid on the mean repeated train-OOF probability per record",
        "only_branch_arrays_read": ["oof_branches.npy", "val_branches.npy"],
        "test_arrays_read": False,
        "test_labels_read": False,
        "validation_role": "final score only after candidate selection; not used to select parameters or profile",
        "caveat": "Validation was inspected in prior work and is not a blind estimate.",
        "input_hashes": input_hashes,
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(validation_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
