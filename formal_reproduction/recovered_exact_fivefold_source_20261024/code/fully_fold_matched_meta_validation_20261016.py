"""Rebuild corrected raw-feature branches and refit the LightGBM meta model.

All five species-matched branches are regenerated from the repaired 628-column
matrices. Dynamic homology, MMseqs and PPI features are rebuilt within each
training fold. The resulting OOF probabilities are joined to the four corrected
homology-scope OOF probabilities; model/profile selection uses training rows
only. Validation predictions use matching fold-ensemble base models. No test
arrays, IDs, metadata, or labels are loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path.cwd()
CODE = ROOT / "plasmodium-essentiality-reproducibility" / "formal_reproduction" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

FEATURE_ROOT = Path(os.environ.get(
    "PROTOESS_CORRECTED_FEATURE_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\scope_feature_sets_v2",
))
SOURCE_FEATURE_ROOT = Path(os.environ.get(
    "PROTOESS_SOURCE_FEATURE_ROOT",
    str(ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2"),
))
os.environ["PROTOESS_SCOPE_FEATURE_ROOT"] = str(FEATURE_ROOT)
os.environ["PROTOESS_SOURCE_SCOPE_FEATURE_ROOT"] = str(SOURCE_FEATURE_ROOT)
N_SPLITS = int(os.environ.get("PROTOESS_RAW_BRANCH_FOLDS", "3"))
FOLD_SEED = int(os.environ.get("PROTOESS_RAW_BRANCH_SEED", "42"))

import compare_homology_label_scopes as scopes  # noqa: E402
import train_scope_strict_oof_stack as strict  # noqa: E402
import train_scope_transformer_oof as transformer  # noqa: E402
import tune_oof_probability_lightgbm_meta_20261016 as meta  # noqa: E402


STALE_BRANCH_ROOT = Path(os.environ.get(
    "PROTOESS_STALE_BRANCH_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\model_five_branch\within_species",
))
SCOPE_PREDICTIONS = Path(os.environ.get(
    "PROTOESS_NINE_SCOPE_PREDICTIONS",
    r"D:\ProtoEss_feature_matrix_repair_20261015\homology_scope_with_oberstaller_20261016\scope_comparison_predictions.tsv",
))
SCOPE_FOLD_PREDICTIONS = Path(os.environ.get(
    "PROTOESS_NINE_SCOPE_FOLD_PREDICTIONS",
    r"D:\ProtoEss_feature_matrix_repair_20261015\fold_ensemble_scope_validation_diagnostic_20261016\fold_ensemble_scope_validation_predictions.tsv",
))
OUTPUT = Path(os.environ.get(
    "PROTOESS_RAW_BRANCH_OUTPUT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\corrected_raw_foldmatched_meta_20261016",
))
BRANCH_OUTPUT = OUTPUT / "five_branch"
META_OUTPUT = OUTPUT / "meta"

PROFILE_INDICES = {
    "corrected_five_branch": tuple(range(4, 9)),
    "three_scope_lgbm": (0, 1, 2),
    "combined_eight_without_oberstaller": (0, 1, 2, 4, 5, 6, 7, 8),
    "four_scope_lgbm": (0, 1, 2, 3),
    "combined_nine_base_probabilities": tuple(range(9)),
}
TARGETS = {
    "Plasmodium_falciparum": {"AUROC": 0.811392, "AUPRC": 0.865720},
    "Plasmodium_berghei": {"AUROC": 0.918908, "AUPRC": 0.928316},
    "Plasmodium_knowlesi": {"AUROC": 0.893703, "AUPRC": 0.900711},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(species: str) -> dict[str, str]:
    matrix = FEATURE_ROOT / "within_species" / species
    manifest = scopes.MANIFESTS / "within_species" / species
    hits = SOURCE_FEATURE_ROOT / "within_species" / species
    files = {
        "X_train": matrix / "X_train.npy",
        "X_val": matrix / "X_val.npy",
        "ids_train": matrix / "ids_train.npy",
        "ids_val": matrix / "ids_val.npy",
        "feature_names": matrix / "feature_names.json",
        "train_manifest": manifest / "train_metadata.tsv",
        "validation_manifest": manifest / "val_metadata.tsv",
        "train_hits": hits / "train_vs_scope_train.tsv",
        "validation_hits": hits / "val_vs_scope_train.tsv",
    }
    files.update({
        "strict_branch_code": Path(strict.__file__),
        "transformer_code": Path(transformer.__file__),
        "feature_profile_code": CODE / "feature_profiles.py",
        "rebuild_script": Path(__file__),
    })
    return {name: sha256(path) for name, path in files.items()}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_scope_probabilities(frame: pd.DataFrame, species: str, split: str, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_split = "train_oof" if split == "train" else "validation"
    part_all = frame.loc[(frame.species == species) & (frame.split == source_split)]
    columns = []
    labels = None
    for scope in meta.ALL_SCOPE_NAMES:
        part = part_all.loc[part_all.scope == scope, ["record_id", "label", "probability"]].copy()
        if part.record_id.astype(str).duplicated().any():
            raise ValueError(f"{species}/{source_split}/{scope}: duplicated record IDs")
        part["record_id"] = part.record_id.astype(str)
        part = part.set_index("record_id").reindex(ids)
        if part.probability.isna().any() or part.label.isna().any():
            raise ValueError(f"{species}/{source_split}/{scope}: incomplete aligned predictions")
        current_labels = part.label.astype(int).to_numpy()
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError(f"{species}/{source_split}/{scope}: labels differ by scope")
        columns.append(part.probability.to_numpy(dtype=np.float32))
    return np.column_stack(columns), labels


def align_fold_scope_validation(frame: pd.DataFrame, species: str, ids: np.ndarray) -> np.ndarray:
    species_rows = frame.loc[frame.species == species]
    columns = []
    for scope in meta.ALL_SCOPE_NAMES:
        part = species_rows.loc[species_rows.scope == scope, ["record_id", "probability"]].copy()
        if part.record_id.astype(str).duplicated().any():
            raise ValueError(f"{species}/{scope}: duplicate fold-ensemble validation IDs")
        part["record_id"] = part.record_id.astype(str)
        aligned = part.set_index("record_id").reindex(ids)
        if aligned.probability.isna().any():
            raise ValueError(f"{species}/{scope}: incomplete fold-ensemble validation predictions")
        columns.append(aligned.probability.to_numpy(dtype=np.float32))
    return np.column_stack(columns)


def reconstruct_species(species: str) -> dict[str, object]:
    train = scopes.read_metadata("within_species", species, "train")
    val = scopes.read_metadata("within_species", species, "val")
    train_ids = train.record_id.astype(str).to_numpy()
    val_ids = val.record_id.astype(str).to_numpy()
    y_train = train.label.eq("essential").astype(np.int8).to_numpy()
    y_val = val.label.eq("essential").astype(np.int8).to_numpy()
    species_dir = FEATURE_ROOT / "within_species" / species
    if not (species_dir / "DONE.json").exists():
        raise FileNotFoundError(f"Missing corrected feature matrix: {species_dir}")
    matrix_train_ids = np.load(species_dir / "ids_train.npy", allow_pickle=True).astype(str)
    matrix_val_ids = np.load(species_dir / "ids_val.npy", allow_pickle=True).astype(str)
    if not np.array_equal(matrix_train_ids, train_ids) or not np.array_equal(matrix_val_ids, val_ids):
        raise ValueError(f"{species}: corrected matrix IDs differ from frozen manifests")
    X_train = np.asarray(np.load(species_dir / "X_train.npy", mmap_mode="r"), dtype=np.float32)
    X_val = np.asarray(np.load(species_dir / "X_val.npy", mmap_mode="r"), dtype=np.float32)
    feature_names = json.loads((species_dir / "feature_names.json").read_text(encoding="utf-8"))
    if X_train.shape[1] != 628 or X_val.shape[1] != 628 or len(feature_names) != 628:
        raise ValueError(f"{species}: expected the original 628-column raw feature matrix")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError(f"{species}: non-finite value in corrected raw features")

    hashes = source_hashes(species)
    out_dir = BRANCH_OUTPUT / species
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "fold_checkpoint.npz"
    if checkpoint.exists() and (out_dir / "checkpoint_source_hashes.json").exists():
        saved_hashes = json.loads((out_dir / "checkpoint_source_hashes.json").read_text(encoding="utf-8"))
        if saved_hashes == hashes:
            cached = np.load(checkpoint)
            rebuilt_oof = np.asarray(cached["oof"], dtype=np.float32)
            validation_by_fold = np.asarray(cached["validation_by_fold"], dtype=np.float32)
            complete = np.asarray(cached["fold_complete"], dtype=bool)
            if rebuilt_oof.shape != (len(train_ids), 5) or validation_by_fold.shape != (N_SPLITS, len(val_ids), 5) or complete.shape != (N_SPLITS,):
                raise ValueError(f"{species}: invalid fold checkpoint dimensions")
        else:
            rebuilt_oof = np.zeros((len(train_ids), 5), dtype=np.float32)
            validation_by_fold = np.zeros((N_SPLITS, len(val_ids), 5), dtype=np.float32)
            complete = np.zeros(N_SPLITS, dtype=bool)
    else:
        rebuilt_oof = np.zeros((len(train_ids), 5), dtype=np.float32)
        validation_by_fold = np.zeros((N_SPLITS, len(val_ids), 5), dtype=np.float32)
        complete = np.zeros(N_SPLITS, dtype=bool)
    (out_dir / "checkpoint_source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    frames = {"train": train, "val": val}
    hits_train = scopes.read_hits(species, "train")
    hits_val = scopes.read_hits(species, "val")
    ppi_edges = scopes.read_target_species_edges(species, set(train_ids) | set(val_ids))
    species_train = train.species.astype(str).to_numpy()
    species_val = val.species.astype(str).to_numpy()
    k_species = int(X_train.shape[1] - 582 - 9 - 21 - 15)
    static_cols = 582 + 9 + k_species
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(train_ids, y_train))
    fold_rows = []
    old_oof_path = STALE_BRANCH_ROOT / species / "oof_branches.npy"
    old_oof = np.asarray(np.load(old_oof_path, mmap_mode="r"), dtype=np.float32) if old_oof_path.exists() else None

    for fold, (inner_ix, hold_ix) in enumerate(folds):
        if complete[fold]:
            print(f"{species}: fold {fold + 1}/{N_SPLITS} already checkpointed", flush=True)
            continue
        inner_ids = set(train_ids[inner_ix])
        hold_ids = train_ids[hold_ix]
        Xi_raw = strict.fold_features(
            X_train[inner_ix].copy(), frames, hits_train, ppi_edges,
            inner_ids, train_ids[inner_ix], static_cols,
        )
        Xh_raw = strict.fold_features(
            X_train[hold_ix].copy(), frames, hits_train, ppi_edges,
            inner_ids, hold_ids, static_cols,
        )
        Xv_raw = strict.fold_features(
            X_val.copy(), frames, hits_val, ppi_edges,
            inner_ids, val_ids, static_cols,
        )
        X_eval_raw = np.concatenate([Xh_raw, Xv_raw], axis=0)
        X_fit, X_eval, _, _ = strict.fit_profile_matrix(
            Xi_raw, X_eval_raw, feature_names, profile=None, seed=100 + fold,
        )
        n_hold = len(hold_ix)
        X_hold = X_eval[:n_hold]
        X_val_fold = X_eval[n_hold:]

        lgbm_model = strict.lgbm(X_fit, y_train[inner_ix], 100 + fold)
        cat_model = strict.cat(X_fit, y_train[inner_ix], 200 + fold)
        hold_lgbm = lgbm_model.predict_proba(X_hold)[:, 1]
        val_lgbm = lgbm_model.predict_proba(X_val_fold)[:, 1]
        hold_cat = cat_model.predict_proba(X_hold)[:, 1]
        val_cat = cat_model.predict_proba(X_val_fold)[:, 1]

        hold_expert = hold_lgbm.copy()
        val_expert = val_lgbm.copy()
        for sp in sorted(set(species_train)):
            inner_local = np.flatnonzero(species_train[inner_ix] == sp)
            hold_local = np.flatnonzero(species_train[hold_ix] == sp)
            val_local = np.flatnonzero(species_val == sp)
            if len(inner_local) >= 30 and np.unique(y_train[inner_ix][inner_local]).size == 2:
                expert = strict.lgbm(X_fit[inner_local], y_train[inner_ix][inner_local], 300 + fold)
                if len(hold_local):
                    hold_expert[hold_local] = expert.predict_proba(X_hold[hold_local])[:, 1]
                if len(val_local):
                    val_expert[val_local] = expert.predict_proba(X_val_fold[val_local])[:, 1]

        hold_prior = np.zeros(n_hold, dtype=np.float32)
        val_prior = np.zeros(len(val_ids), dtype=np.float32)
        global_prior = float(y_train[inner_ix].mean())
        for sp in sorted(set(species_train)):
            inner_local = np.flatnonzero(species_train[inner_ix] == sp)
            hold_local = np.flatnonzero(species_train[hold_ix] == sp)
            val_local = np.flatnonzero(species_val == sp)
            prior = float(y_train[inner_ix][inner_local].mean()) if len(inner_local) else global_prior
            hold_prior[hold_local] = prior
            val_prior[val_local] = prior

        fit_groups, eval_groups, group_names = transformer.fit_profile_groups(
            Xi_raw, X_eval_raw, feature_names, profile=None, seed=42 + fold,
        )
        modalities = transformer.transformer_modalities(eval_groups, group_names)
        hold_modalities = [group[:n_hold] for group in modalities]
        val_modalities = [group[n_hold:] for group in modalities]
        hold_transformer, val_transformer, best_epoch = transformer.train_predict(
            transformer.transformer_modalities(fit_groups, group_names),
            y_train[inner_ix],
            hold_modalities,
            y_train[hold_ix],
            val_modalities,
            42 + fold,
        )

        rebuilt_oof[hold_ix] = np.column_stack([
            hold_lgbm, hold_cat, hold_expert, hold_prior, hold_transformer,
        ]).astype(np.float32)
        validation_by_fold[fold] = np.column_stack([
            val_lgbm, val_cat, val_expert, val_prior, val_transformer,
        ]).astype(np.float32)
        complete[fold] = True
        atomic_npz(checkpoint, oof=rebuilt_oof, validation_by_fold=validation_by_fold, fold_complete=complete)
        fold_rows.append({
            "species": species,
            "fold": fold,
            "n_inner": int(len(inner_ix)),
            "n_holdout": int(len(hold_ix)),
            "transformer_best_epoch": int(best_epoch),
        })
        print(f"{species}: rebuilt corrected raw fold {fold + 1}/{N_SPLITS}; Transformer epoch={best_epoch}", flush=True)

    if not complete.all():
        raise RuntimeError(f"{species}: one or more branch folds were not completed")
    val_branches = validation_by_fold.mean(axis=0).astype(np.float32)
    np.save(out_dir / "oof_branches.npy", rebuilt_oof)
    np.save(out_dir / "val_branches.npy", val_branches)
    np.save(out_dir / "val_branches_by_fold.npy", validation_by_fold)
    pd.DataFrame(fold_rows).to_csv(out_dir / "training_log.tsv", sep="\t", index=False)

    stale_diff = None
    if old_oof is not None and old_oof.shape == rebuilt_oof.shape:
        # Legacy branch artifacts lack ID sidecars. Frozen manifests establish
        # the same row order, but this is a source-difference audit, not a gate.
        stale_diff = np.max(np.abs(old_oof - rebuilt_oof), axis=0).astype(float).tolist()
    result_manifest = {
        "species": species,
        "source": str(species_dir),
        "feature_count": 628,
        "feature_values": "original raw-scale matrix values; no PCA or feature screening",
        "dynamic_features": "homology30/50, MMseqs, and PPI rebuilt within each outer training fold",
        "folds": {"n_splits": N_SPLITS, "shuffle": True, "random_state": FOLD_SEED},
        "branch_order": ["lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer"],
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "source_hashes": hashes,
        "legacy_oof_max_abs_difference_by_branch_non_gating": stale_diff,
        "legacy_arrays_have_id_sidecars": False,
    }
    (out_dir / "reconstruction_manifest.json").write_text(json.dumps(result_manifest, indent=2), encoding="utf-8")
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "y_train": y_train,
        "y_val": y_val,
        "oof": rebuilt_oof,
        "val_by_fold": validation_by_fold,
        "val_mean": val_branches,
        "manifest": result_manifest,
    }


def select_meta_and_score(scope_frame: pd.DataFrame, fold_scope_frame: pd.DataFrame, branches: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    all_candidate_rows = []
    prediction_rows = []
    metrics_rows = []
    for species in meta.SPECIES:
        base = branches[species]
        train_ids = base["train_ids"]
        val_ids = base["val_ids"]
        y_train = base["y_train"]
        y_val = base["y_val"]
        scope_train, y_scope_train = load_scope_probabilities(scope_frame, species, "train", train_ids)
        scope_full_val, y_scope_val = load_scope_probabilities(scope_frame, species, "val", val_ids)
        scope_fold_val = align_fold_scope_validation(fold_scope_frame, species, val_ids)
        if not np.array_equal(y_scope_train, y_train) or not np.array_equal(y_scope_val, y_val):
            raise ValueError(f"{species}: scope labels differ from frozen train/validation manifests")
        X_train_all = np.column_stack([scope_train, base["oof"]]).astype(np.float32)
        X_val_fold_all = np.column_stack([scope_fold_val, base["val_mean"]]).astype(np.float32)
        X_val_full_scope_all = np.column_stack([scope_full_val, base["val_mean"]]).astype(np.float32)

        species_candidates = []
        for profile_name, indices in PROFILE_INDICES.items():
            X_profile = X_train_all[:, indices]
            for candidate in meta.CANDIDATES:
                _, repeat_metrics, _ = meta.cv_predictions(X_profile, y_train, candidate, meta.CV, 5, 3)
                row = {
                    "species": species,
                    "profile": profile_name,
                    "candidate": candidate["name"],
                    "probability_count": len(indices),
                    **{key: value for key, value in candidate.items() if key != "name"},
                    "mean_repeat_AUPRC": float(repeat_metrics.AUPRC.mean()),
                    "sd_repeat_AUPRC": float(repeat_metrics.AUPRC.std(ddof=1)),
                    "mean_repeat_AUROC": float(repeat_metrics.AUROC.mean()),
                    "sd_repeat_AUROC": float(repeat_metrics.AUROC.std(ddof=1)),
                }
                species_candidates.append(row)
        candidate_frame = pd.DataFrame(species_candidates)
        ap_sd = float(candidate_frame.mean_repeat_AUPRC.std(ddof=0))
        auc_sd = float(candidate_frame.mean_repeat_AUROC.std(ddof=0))
        candidate_frame["z_AUPRC"] = (candidate_frame.mean_repeat_AUPRC - candidate_frame.mean_repeat_AUPRC.mean()) / max(ap_sd, 1e-12)
        candidate_frame["z_AUROC"] = (candidate_frame.mean_repeat_AUROC - candidate_frame.mean_repeat_AUROC.mean()) / max(auc_sd, 1e-12)
        candidate_frame["balanced_training_cv_score"] = 0.5 * (candidate_frame.z_AUPRC + candidate_frame.z_AUROC)
        candidate_frame["complexity"] = candidate_frame.num_leaves.astype(int) * candidate_frame.n_estimators.astype(int)
        candidate_frame = candidate_frame.sort_values(
            ["balanced_training_cv_score", "mean_repeat_AUPRC", "probability_count", "complexity"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        all_candidate_rows.extend(candidate_frame.to_dict(orient="records"))
        chosen = candidate_frame.iloc[0]
        selected_profile = str(chosen.profile)
        selected_candidate = next(item for item in meta.CANDIDATES if item["name"] == str(chosen.candidate))
        selected_indices = PROFILE_INDICES[selected_profile]
        model = meta.make_model(selected_candidate, 42)
        model.fit(X_train_all[:, selected_indices], y_train)
        for variant, Xval in (
            ("all_fold_matched", X_val_fold_all),
            ("foldmatched_branches_fullrefit_scope_sensitivity", X_val_full_scope_all),
        ):
            probability = model.predict_proba(Xval[:, selected_indices])[:, 1]
            auc = float(roc_auc_score(y_val, probability))
            ap = float(average_precision_score(y_val, probability))
            target = TARGETS[species]
            metrics_rows.append({
                "species": species,
                "selection": "balanced_training_CV_across_5_profiles_x_8_LightGBM_configs",
                "validation_inference": variant,
                "selected_profile": selected_profile,
                "selected_candidate": selected_candidate["name"],
                "training_CV_AUPRC": float(chosen.mean_repeat_AUPRC),
                "training_CV_AUROC": float(chosen.mean_repeat_AUROC),
                "validation_AUROC": auc,
                "validation_AUPRC": ap,
                "original_AUROC_reference": target["AUROC"],
                "original_AUPRC_reference": target["AUPRC"],
                "AUROC_gap": auc - target["AUROC"],
                "AUPRC_gap": ap - target["AUPRC"],
                "validation_used_for_selection": False,
            })
            for rid, label, value in zip(val_ids, y_val, probability):
                prediction_rows.append({
                    "species": species,
                    "validation_inference": variant,
                    "profile": selected_profile,
                    "candidate": selected_candidate["name"],
                    "record_id": rid,
                    "label": int(label),
                    "probability": float(value),
                })
        selection_rows.append({
            "species": species,
            "selected_profile": selected_profile,
            "selected_candidate": selected_candidate["name"],
            "training_CV_AUPRC": float(chosen.mean_repeat_AUPRC),
            "training_CV_AUROC": float(chosen.mean_repeat_AUROC),
            "balanced_training_CV_score": float(chosen.balanced_training_cv_score),
            "selection_rule": "max equal-weight standardized mean AUPRC/AUROC from repeated training-only CV",
        })
        print(f"{species}: selected {selected_profile}/{selected_candidate['name']} from training CV", flush=True)
    return pd.DataFrame(selection_rows), pd.DataFrame(all_candidate_rows), pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-branches-only",
        action="store_true",
        help="Rebuild five-branch OOF/validation probabilities, skipping all meta-model search.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BRANCH_OUTPUT.mkdir(parents=True, exist_ok=True)
    META_OUTPUT.mkdir(parents=True, exist_ok=True)
    branches = {}
    for species in meta.SPECIES:
        branches[species] = reconstruct_species(species)
    if args.rebuild_branches_only:
        manifest = {
            "purpose": "Rebuild five base-branch OOF probabilities for the frozen nine-route model; no meta search.",
            "feature_matrix": str(FEATURE_ROOT),
            "source_feature_files": str(SOURCE_FEATURE_ROOT),
            "feature_count": 628,
            "feature_values": "raw matrix values; no PCA or feature selection",
            "folds": {"n_splits": N_SPLITS, "shuffle": True, "random_state": FOLD_SEED},
            "test_arrays_ids_metadata_or_labels_opened": False,
            "species": {species: item["manifest"] for species, item in branches.items()},
        }
        (OUTPUT / "branch_rebuild_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Rebuilt five-branch OOF probabilities under {BRANCH_OUTPUT}", flush=True)
        return
    scope_frame = pd.read_csv(SCOPE_PREDICTIONS, sep="\t", dtype={"record_id": str})
    fold_scope_frame = pd.read_csv(SCOPE_FOLD_PREDICTIONS, sep="\t", dtype={"record_id": str})
    selection, candidates, metrics, predictions = select_meta_and_score(scope_frame, fold_scope_frame, branches)

    selection.to_csv(META_OUTPUT / "training_cv_selected_meta.tsv", sep="\t", index=False)
    candidates.to_csv(META_OUTPUT / "all_training_cv_candidates.tsv", sep="\t", index=False)
    metrics.to_csv(META_OUTPUT / "foldmatched_validation_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(META_OUTPUT / "foldmatched_validation_predictions.tsv", sep="\t", index=False)
    audit = pd.DataFrame([
        {
            "species": species,
            "feature_count": item["manifest"]["feature_count"],
            "feature_values": item["manifest"]["feature_values"],
            "legacy_oof_max_abs_difference_by_branch_non_gating": json.dumps(item["manifest"]["legacy_oof_max_abs_difference_by_branch_non_gating"]),
            "legacy_arrays_have_id_sidecars": item["manifest"]["legacy_arrays_have_id_sidecars"],
            "train_ids_sha256": item["manifest"]["train_ids_sha256"],
            "validation_ids_sha256": item["manifest"]["validation_ids_sha256"],
        }
        for species, item in branches.items()
    ])
    audit.to_csv(BRANCH_OUTPUT / "legacy_vs_corrected_source_audit.tsv", sep="\t", index=False)
    run_manifest = {
        "purpose": "Rebuild five branches from corrected raw matrices, train-select LightGBM meta profile/config, then score frozen validation once.",
        "feature_matrix": str(FEATURE_ROOT),
        "feature_count": 628,
        "feature_scaling_or_PCA": "none; raw matrix values preserved",
        "dynamic_feature_policy": "homology30/50, MMseqs, and PPI recomputed using only each fold's inner-training labels",
        "base_oof_and_validation_inference": f"same {N_SPLITS} folds, seed {FOLD_SEED}; validation prediction is mean of fold models",
        "meta_model_selection": "five base-probability profiles by eight fixed LightGBM configurations; repeated stratified CV on training rows only; balanced standardized AUROC/AUPRC",
        "validation_used_for_selection": False,
        "test_arrays_ids_metadata_or_labels_opened": False,
        "known_caveat": "Meta-level CV is not fully nested because base OOF probabilities are precomputed; the frozen validation set was viewed during prior project iterations and is diagnostic, not a blind estimate.",
        "inputs_sha256": {
            "scope_oof_predictions": sha256(SCOPE_PREDICTIONS),
            "fold_ensemble_scope_validation_predictions": sha256(SCOPE_FOLD_PREDICTIONS),
        },
        "species": {species: item["manifest"] for species, item in branches.items()},
        "outputs": [
            "five_branch/<species>/oof_branches.npy",
            "five_branch/<species>/val_branches.npy",
            "five_branch/<species>/val_branches_by_fold.npy",
            "five_branch/legacy_vs_corrected_source_audit.tsv",
            "meta/training_cv_selected_meta.tsv",
            "meta/all_training_cv_candidates.tsv",
            "meta/foldmatched_validation_metrics.tsv",
            "meta/foldmatched_validation_predictions.tsv",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
