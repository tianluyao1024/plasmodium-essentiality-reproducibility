"""Run the Nature Communications functional-annotation-excluded analysis.

The experiment mirrors the frozen within-species three-fold OOF stack for the
three malaria parasites.  The only feature intervention is masking the
GO/Pfam/InterPro block (columns 287:415).  MMseqs2, STRING/PPI, sequence,
protein-language-model and species channels remain available.

The runner is checkpointed at species/fold level.  It writes ``state.json``
after every durable step so that ``watch_ncomms_revision.py`` can resume an
interrupted run without overwriting completed work.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run_plasmodium_cluster_blocked_full_oof import (
    all_vs_all,
    catboost,
    choose_threshold,
    dynamic_x,
    edge_bundle,
    lgbm,
    metrics,
    read_all,
)
from score_final_application_models import APP_META, RUN, app_arrays, app_ppi_dynamic, app_static
from train_scope_transformer_oof import split_groups, train_predict


ROOT = Path(__file__).resolve().parent
# Optional overrides isolate an updated canonical downstream freeze from the
# historical annotation-exclusion analysis.
OUT = Path(os.environ.get(
    "NCOMMS_ANNOTATION_EXCLUDED_OUT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_annotation_excluded"),
))
PRED_OUT = OUT / "genomewide_predictions"
MODEL_OUT = OUT / "models"
CHECKPOINTS = OUT / "checkpoints"
STATE = OUT / "state.json"
MM_HITS = RUN / "app_vs_train.tsv"
SPECIES = (
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
)
ALIASES = {
    "Plasmodium_falciparum": "Plasmodium_falciparum_3D7",
    "Plasmodium_berghei": "Plasmodium_berghei_ANKA",
    "Plasmodium_knowlesi": "Plasmodium_knowlesi_H",
}
ANNOTATION_SLICE = slice(287, 415)
SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_state(**updates: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    if STATE.exists():
        try:
            current = json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(updates)
    current["updated_at_epoch"] = time.time()
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE)


def mask_annotation(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    if result.shape[1] < ANNOTATION_SLICE.stop:
        raise ValueError(f"Expected at least {ANNOTATION_SLICE.stop} features, found {result.shape[1]}")
    result[:, ANNOTATION_SLICE] = 0.0
    return result


def read_frames(species: str) -> dict[str, pd.DataFrame]:
    base = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species" / species
    return {
        split: pd.read_csv(base / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str}).fillna("")
        for split in ("train", "val", "test")
    }


def bootstrap_pair(y: np.ndarray, full: np.ndarray, excluded: np.ndarray, repeats: int = 1000) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260820)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rows: list[dict[str, object]] = []
    for metric_name, function in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
        observed = float(function(y, excluded) - function(y, full))
        values = np.empty(repeats, dtype=float)
        for index in range(repeats):
            sampled = np.concatenate([
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ])
            values[index] = function(y[sampled], excluded[sampled]) - function(y[sampled], full[sampled])
        rows.append({
            "metric": metric_name,
            "delta_excluded_minus_full": observed,
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_replicates": repeats,
        })
    return rows


def oof_predictions(species: str, raw: np.ndarray, meta: pd.DataFrame, hits: pd.DataFrame, edges: object) -> np.ndarray:
    y = meta.label.eq("essential").astype(np.int8).to_numpy()
    ids = meta.record_id.astype(str).to_numpy()
    train = np.flatnonzero(meta.formal_split.eq("train"))
    id_to_species = dict(zip(ids, meta.species.astype(str)))
    output = np.zeros((len(train), 5), dtype=np.float32)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(train, y[train]))
    folder = CHECKPOINTS / species
    folder.mkdir(parents=True, exist_ok=True)
    for fold, (inner_local, held_local) in enumerate(folds):
        checkpoint = folder / f"oof_fold_{fold}.npy"
        held_file = folder / f"oof_fold_{fold}_held.npy"
        if checkpoint.exists() and held_file.exists():
            held = np.load(held_file)
            output[held, :] = np.load(checkpoint)
            print(species, "resume OOF fold", fold, flush=True)
            continue
        inner, held_global = train[inner_local], train[held_local]
        x_inner = mask_annotation(dynamic_x(raw[inner], ids[inner], ids[inner], y[inner], id_to_species, hits, edges))
        x_held = mask_annotation(dynamic_x(raw[held_global], ids[held_global], ids[inner], y[inner], id_to_species, hits, edges))
        fold_values = np.zeros((len(held_local), 5), dtype=np.float32)
        global_lgbm = lgbm(x_inner, y[inner], 100 + fold)
        global_cat = catboost(x_inner, y[inner], 200 + fold)
        expert = lgbm(x_inner, y[inner], 300 + fold)
        fold_values[:, 0] = global_lgbm.predict_proba(x_held)[:, 1]
        fold_values[:, 1] = global_cat.predict_proba(x_held)[:, 1]
        fold_values[:, 2] = expert.predict_proba(x_held)[:, 1]
        fold_values[:, 3] = float(y[inner].mean())
        fold_values[:, 4] = train_predict(
            split_groups(x_inner, 1), y[inner], split_groups(x_held, 1), y[held_global], split_groups(x_held, 1),
            400 + fold,
        )[0]
        np.save(checkpoint, fold_values)
        np.save(held_file, held_local)
        output[held_local, :] = fold_values
        write_state(status="running", species=species, stage="oof", completed_fold=fold)
    if not np.isfinite(output).all() or np.any(output[:, :3] == 0):
        raise RuntimeError(f"Incomplete OOF matrix for {species}")
    np.save(folder / "oof_base.npy", output)
    return output


def application_matrix(species: str, frames: dict[str, pd.DataFrame], arrays: dict[str, np.ndarray],
                       mm_hits: pd.DataFrame, ppi_dynamic: np.ndarray) -> np.ndarray:
    static = app_static("within_species", species, frames, arrays)
    train_ids = frames["train"].record_id.astype(str).to_numpy()
    y = frames["train"].label.eq("essential").astype(float).to_numpy()
    selected = mm_hits[mm_hits.target.isin(set(train_ids))].copy()
    selected["identity"] = selected.pident.astype(float) / 100.0
    selected["coverage"] = selected.alnlen.astype(float) / selected[["qlen", "tlen"]].astype(float).max(axis=1)
    selected["score"] = selected.identity * selected.coverage
    labels = dict(zip(train_ids, y))
    target_species = dict(zip(train_ids, frames["train"].species.astype(str)))
    query_species = dict(zip(
        APP_META.record_id.astype(str),
        APP_META.species.astype(str).map(lambda value: {v: k for k, v in ALIASES.items()}.get(value, value)),
    ))
    app_ids = APP_META.record_id.astype(str).to_numpy()
    from train_scope_strict_oof_stack import aggregate_mm
    dynamic_mm = aggregate_mm(selected[selected["query"].isin(set(app_ids))], app_ids, labels, target_species, query_species)
    result = np.concatenate([static, dynamic_mm, ppi_dynamic], axis=1).astype(np.float32)
    return mask_annotation(result)


def train_species(species: str, app_parts: dict[str, np.ndarray], mm_hits: pd.DataFrame,
                  ppi_dynamic: np.ndarray) -> dict[str, object]:
    frames = read_frames(species)
    combined_meta, raw = read_all(species)
    y_all = combined_meta.label.eq("essential").astype(np.int8).to_numpy()
    ids = combined_meta.record_id.astype(str).to_numpy()
    train = np.flatnonzero(combined_meta.formal_split.eq("train"))
    val = np.flatnonzero(combined_meta.formal_split.eq("val"))
    test = np.flatnonzero(combined_meta.formal_split.eq("test"))
    cache = CHECKPOINTS / species
    cache.mkdir(parents=True, exist_ok=True)
    hits = all_vs_all(species, cache)
    edges = edge_bundle(species)
    oof = oof_predictions(species, raw, combined_meta, hits, edges)

    id_to_species = dict(zip(ids, combined_meta.species.astype(str)))
    x_train = mask_annotation(dynamic_x(raw[train], ids[train], ids[train], y_all[train], id_to_species, hits, edges))
    x_val = mask_annotation(dynamic_x(raw[val], ids[val], ids[train], y_all[train], id_to_species, hits, edges))
    x_test = mask_annotation(dynamic_x(raw[test], ids[test], ids[train], y_all[train], id_to_species, hits, edges))
    x_app = application_matrix(species, frames, app_parts, mm_hits, ppi_dynamic)
    if x_app.shape[1] != x_train.shape[1]:
        raise RuntimeError(f"Feature mismatch for {species}: train={x_train.shape}, app={x_app.shape}")

    tree_lgbm = lgbm(x_train, y_all[train], SEED)
    tree_cat = catboost(x_train, y_all[train], SEED)
    expert = lgbm(x_train, y_all[train], SEED + 1)
    val_transformer, test_transformer, _ = train_predict(
        split_groups(x_train, 1), y_all[train], split_groups(x_val, 1), y_all[val], split_groups(x_test, 1), 900,
    )
    _, app_transformer, _ = train_predict(
        split_groups(x_train, 1), y_all[train], split_groups(x_val, 1), y_all[val], split_groups(x_app, 1), 900,
    )

    def branches(matrix: np.ndarray, transformer: np.ndarray) -> np.ndarray:
        return np.column_stack([
            tree_lgbm.predict_proba(matrix)[:, 1],
            tree_cat.predict_proba(matrix)[:, 1],
            expert.predict_proba(matrix)[:, 1],
            np.repeat(y_all[train].mean(), len(matrix)),
            transformer,
        ])

    val_branches = branches(x_val, val_transformer)
    test_branches = branches(x_test, test_transformer)
    app_branches = branches(x_app, app_transformer)
    meta_model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    meta_model.fit(oof, y_all[train])
    val_scores = meta_model.predict_proba(val_branches)[:, 1]
    test_scores = meta_model.predict_proba(test_branches)[:, 1]
    app_scores = meta_model.predict_proba(app_branches)[:, 1]
    threshold = choose_threshold(y_all[val], val_scores)
    test_metrics = metrics(y_all[test], test_scores, threshold)

    species_app = APP_META.species.astype(str).eq(ALIASES[species]).to_numpy()
    prediction = APP_META.loc[species_app, ["record_id", "species", "gene_id", "protein_id", "protein_length"]].copy()
    prediction["functional_annotation_excluded_score"] = app_scores[species_app]
    prediction["functional_annotation_excluded_threshold"] = threshold
    prediction["functional_annotation_excluded_margin"] = prediction["functional_annotation_excluded_score"] - threshold
    prediction["functional_annotation_excluded_predicted_essential"] = prediction["functional_annotation_excluded_score"] >= threshold
    prediction.to_csv(PRED_OUT / f"{species}_genomewide.tsv", sep="\t", index=False)
    pd.DataFrame({
        "record_id": ids[test], "label": y_all[test], "score": test_scores, "threshold": threshold,
    }).to_csv(OUT / f"{species}_test_predictions.tsv", sep="\t", index=False)
    joblib.dump({
        "lgbm": tree_lgbm,
        "catboost": tree_cat,
        "species_expert": expert,
        "meta_model": meta_model,
        "threshold": threshold,
        "masked_columns": [ANNOTATION_SLICE.start, ANNOTATION_SLICE.stop],
        "architecture": "three-fold OOF five-branch stack; GO/Pfam/InterPro block masked",
    }, MODEL_OUT / f"{species}_tree_meta.joblib")
    np.save(cache / "val_scores.npy", val_scores)
    np.save(cache / "test_scores.npy", test_scores)
    np.save(cache / "app_scores.npy", app_scores)
    return {
        "species": species,
        "train_n": int(len(train)), "validation_n": int(len(val)), "test_n": int(len(test)),
        "threshold": threshold, **{f"test_{key}": value for key, value in test_metrics.items()},
        "genomewide_n": int(species_app.sum()),
        "genomewide_predicted_essential_n": int(prediction.functional_annotation_excluded_predicted_essential.sum()),
    }


def compare_formal_predictions(results: pd.DataFrame) -> None:
    canonical = os.environ.get("NCOMMS_CANONICAL_FIVE_BRANCH_ROOT", "")
    full_root = (Path(canonical) / "within_species") if canonical else (ROOT / "eukaryote_model_results" / "scope_full_oof_stack" / "within_species")
    rows: list[dict[str, object]] = []
    boot: list[dict[str, object]] = []
    for species in SPECIES:
        frames = read_frames(species)
        y = frames["test"].label.eq("essential").astype(np.int8).to_numpy()
        excluded = pd.read_csv(OUT / f"{species}_test_predictions.tsv", sep="\t").score.to_numpy()
        full = np.load(full_root / species / "test_prob.npy")
        full_threshold = float(json.loads((full_root / species / "metrics.json").read_text())["threshold"])
        excluded_threshold = float(results.loc[results.species.eq(species), "threshold"].iloc[0])
        rows.append({
            "species": species,
            "spearman_score": float(pd.Series(full).corr(pd.Series(excluded), method="spearman")),
            "classification_agreement": float(np.mean((full >= full_threshold) == (excluded >= excluded_threshold))),
            "threshold_crossing_rate": float(np.mean((full >= full_threshold) != (excluded >= excluded_threshold))),
            "full_predicted_essential_n": int((full >= full_threshold).sum()),
            "excluded_predicted_essential_n": int((excluded >= excluded_threshold).sum()),
        })
        for item in bootstrap_pair(y, full, excluded):
            boot.append({"species": species, **item})
    pd.DataFrame(rows).to_csv(OUT / "formal_prediction_stability.tsv", sep="\t", index=False)
    pd.DataFrame(boot).to_csv(OUT / "paired_bootstrap_full_vs_excluded.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=SPECIES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for folder in (OUT, PRED_OUT, MODEL_OUT, CHECKPOINTS):
        folder.mkdir(parents=True, exist_ok=True)
    seed_all()
    write_state(status="running", stage="loading_application_features", pid=os.getpid(), error=None)
    app_parts = app_arrays()
    mm_hits = pd.read_csv(
        MM_HITS, sep="\t", header=None,
        names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"],
        dtype={"query": str, "target": str},
    )
    ppi_dynamic = app_ppi_dynamic({}, app_parts)
    selected_species = (args.species,) if args.species else SPECIES
    completed_rows: list[dict[str, object]] = []
    summary_path = OUT / "functional_annotation_excluded_metrics.tsv"
    if summary_path.exists() and not args.force:
        completed_rows = pd.read_csv(summary_path, sep="\t").to_dict("records")
    completed_names = {str(row["species"]) for row in completed_rows}
    try:
        for species in selected_species:
            prediction_path = PRED_OUT / f"{species}_genomewide.tsv"
            if species in completed_names and prediction_path.exists() and not args.force:
                print("resume: completed", species, flush=True)
                continue
            write_state(status="running", stage="training_species", species=species, pid=os.getpid())
            row = train_species(species, app_parts, mm_hits, ppi_dynamic)
            completed_rows = [item for item in completed_rows if item.get("species") != species] + [row]
            pd.DataFrame(completed_rows).to_csv(summary_path, sep="\t", index=False)
            write_state(status="running", stage="species_complete", species=species, completed_species=[r["species"] for r in completed_rows])
        result_frame = pd.DataFrame(completed_rows)
        if set(result_frame.species.astype(str)) == set(SPECIES):
            compare_formal_predictions(result_frame)
            combined = pd.concat([pd.read_csv(PRED_OUT / f"{species}_genomewide.tsv", sep="\t") for species in SPECIES], ignore_index=True)
            combined.to_csv(OUT / "functional_annotation_excluded_genomewide_all_three.tsv", sep="\t", index=False)
            manifest = {
                "analysis_name": "functional-annotation-excluded robustness analysis",
                "excluded_inputs": ["GO", "Pfam", "InterPro"],
                "retained_inputs": ["ESM2", "ProtT5", "sequence", "k-mer", "MMseqs2", "STRING/PPI", "species channel"],
                "oof_folds": 3,
                "formal_splits_reused": True,
                "threshold_selection": "validation-set MCC maximisation",
                "genomewide_records": int(len(combined)),
                "interpretation": "orthogonal computational anti-circularity analysis; not experimental validation",
            }
            (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            write_state(status="complete", stage="annotation_excluded_complete", species=None, pid=os.getpid())
    except Exception as error:
        error_text = traceback.format_exc()
        (OUT / "last_error.log").write_text(error_text, encoding="utf-8")
        write_state(status="failed", stage="exception", error=str(error), traceback=error_text, pid=os.getpid())
        raise


if __name__ == "__main__":
    main()
