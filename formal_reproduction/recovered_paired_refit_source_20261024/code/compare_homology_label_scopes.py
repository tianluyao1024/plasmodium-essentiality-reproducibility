"""Compare strict and cross-species label-assisted homology scopes.

Uses the frozen within-species train/validation split and the original
LightGBM configuration. All label-derived features are rebuilt fold-locally
for OOF predictions. Test arrays, metadata, and labels are never loaded.
Run from the workspace root (the parent of the repository directory).
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path.cwd()
CODE = ROOT / "plasmodium-essentiality-reproducibility" / "formal_reproduction" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_strict_oof_dynamic_stack import TAX, aggregate_mm, aggregate_ppi  # noqa: E402
from protoess_safe_homology import raw_homology_features  # noqa: E402


DATA = ROOT / "eukaryote_model_datasets"
MANIFESTS = DATA / "unified_v3_experiment_manifests_pre95human"
SOURCE_FEATURES = DATA / "scope_feature_sets_v2"
REPAIRED_FEATURES = Path(os.environ.get(
    "PROTOESS_SCOPE_FEATURE_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\scope_feature_sets_v2",
))
PPI_DIR = ROOT / "downloads" / "string_v12"
OBERST_CALLS = Path(
    r"D:\ProtoEss_feature_matrix_repair_20261015\wp1_dualscreen_rebuild_20261016\aligned_dualscreen_gene_calls.tsv"
)
OUT = Path(os.environ.get(
    "PROTOESS_HOMOLOGY_SCOPE_OUT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\homology_scope_comparison_20261015",
))
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
SPLITS = ("train", "val")
N_JOBS = int(os.environ.get("PROTOESS_LGBM_N_JOBS", "8"))
N_SPLITS = int(os.environ.get("PROTOESS_SCOPE_FOLDS", "3"))
FOLD_SEED = int(os.environ.get("PROTOESS_SCOPE_SEED", "42"))

LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "class_weight": "balanced",
    "verbosity": -1,
}


def read_metadata(scope: str, species: str, split: str) -> pd.DataFrame:
    path = MANIFESTS / scope / species / f"{split}_metadata.tsv"
    return pd.read_csv(path, sep="\t", dtype={"record_id": str})


def read_all_training_metadata() -> pd.DataFrame:
    rows = []
    for directory in sorted((MANIFESTS / "within_species").iterdir()):
        path = directory / "train_metadata.tsv"
        if directory.is_dir() and path.exists():
            rows.append(pd.read_csv(path, sep="\t", dtype={"record_id": str})[
                ["record_id", "species", "label"]
            ])
    if not rows:
        raise FileNotFoundError("No within-species training manifests found")
    result = pd.concat(rows, ignore_index=True)
    if result.record_id.astype(str).duplicated().any():
        raise ValueError("Training record IDs are not globally unique")
    return result


def build_oberstaller_training_pool(all_train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace P. knowlesi training labels with definite Oberstaller calls only."""
    pk_train = read_metadata("within_species", "Plasmodium_knowlesi", "train")
    calls = pd.read_csv(
        OBERST_CALLS,
        sep="\t",
        usecols=["gene_id", "ober_call", "is_tentative"],
        dtype=str,
    )
    calls = calls.loc[
        calls.is_tentative.str.lower().eq("false")
        & calls.ober_call.isin({"essential", "nonessential"})
    ].drop_duplicates("gene_id")
    matched = pk_train[["record_id", "species", "primary_id", "label"]].merge(
        calls,
        left_on="primary_id",
        right_on="gene_id",
        how="inner",
        validate="one_to_one",
    )
    matched["elsworth_label"] = matched.label
    matched["source_label"] = matched.ober_call
    matched["label"] = matched.ober_call
    other_species = all_train.loc[
        all_train.species.astype(str) != "Plasmodium_knowlesi",
        ["record_id", "species", "label"],
    ]
    pk_source = matched[["record_id", "species", "label"]]
    result = pd.concat([other_species, pk_source], ignore_index=True)
    if result.record_id.astype(str).duplicated().any():
        raise ValueError("Oberstaller training pool contains duplicate record IDs")
    disagreements = matched.elsworth_label.ne(matched.source_label)
    stats = {
        "source_file": str(OBERST_CALLS),
        "eligible_definite_calls": int(len(calls)),
        "matched_pk_training_records": int(len(matched)),
        "pk_training_records": int(len(pk_train)),
        "coverage": float(len(matched) / len(pk_train)),
        "label_disagreements_vs_elsworth_within_matched_training_records": int(disagreements.sum()),
        "disagreement_rate_within_matched_training_records": float(disagreements.mean()),
        "validation_or_test_records_joined": 0,
    }
    return result, stats


def read_hits(species: str, split: str) -> pd.DataFrame:
    path = SOURCE_FEATURES / "within_species" / species / f"{split}_vs_scope_train.tsv"
    hits = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"],
        dtype={"query": str, "target": str},
    )
    hits["identity"] = hits.pident.astype(float) / 100.0
    hits["coverage"] = hits.alnlen.astype(float) / hits[["qlen", "tlen"]].astype(float).max(axis=1)
    hits["score"] = hits.identity * hits.coverage
    return hits


def read_target_species_edges(species: str, record_ids: set[str]):
    tax = TAX[species]
    mapping = pd.read_csv(
        PPI_DIR / f"current_map_{tax}.tsv",
        sep="\t",
        header=None,
        names=["record", "sid", "pid", "aln", "e", "bits", "qlen", "tlen"],
        dtype={"record": str, "sid": str},
    )
    mapping = mapping[mapping.record.isin(record_ids)].drop_duplicates("record")
    sid_to_record = dict(zip(mapping.sid, mapping.record))
    sources: list[str] = []
    targets: list[str] = []
    scores: list[float] = []
    links_path = PPI_DIR / f"{tax}.protein.links.v12.0.txt.gz"
    with gzip.open(links_path, "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            source_sid, target_sid, score = line.split()
            source = sid_to_record.get(source_sid)
            target = sid_to_record.get(target_sid)
            if source is not None and target is not None and source != target:
                sources.append(source)
                targets.append(target)
                scores.append(float(score) / 1000.0)
    return [(species, np.asarray(sources, dtype=str), np.asarray(targets, dtype=str), np.asarray(scores, dtype=np.float32))]


def scaled_homology(query_fit: pd.DataFrame, query_eval: pd.DataFrame, allowed_labels: pd.DataFrame):
    raw_fit = raw_homology_features(query_fit[["record_id", "species"]], allowed_labels)
    raw_eval = raw_homology_features(query_eval[["record_id", "species"]], allowed_labels)
    mean = raw_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw_fit.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
    return ((raw_fit - mean) / std).astype(np.float32), ((raw_eval - mean) / std).astype(np.float32)


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "MCC": float(matthews_corrcoef(y, p >= threshold)),
        "F1": float(f1_score(y, p >= threshold, zero_division=0)),
    }


def best_oof_threshold(y: np.ndarray, p: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y, p >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def make_model(seed: int):
    return lgb.LGBMClassifier(random_state=seed, n_jobs=N_JOBS, **LGBM_PARAMS)


def run_species(species: str, all_train: pd.DataFrame, all_train_ober: pd.DataFrame,
                ober_stats: dict) -> tuple[list[dict], list[dict]]:
    split_dir = MANIFESTS / "within_species" / species
    source_dir = SOURCE_FEATURES / "within_species" / species
    repaired_dir = REPAIRED_FEATURES / "within_species" / species
    frames = {split: read_metadata("within_species", species, split) for split in SPLITS}
    ids = {split: frames[split].record_id.astype(str).to_numpy() for split in SPLITS}
    y = {split: frames[split].label.eq("essential").astype(np.int8).to_numpy() for split in SPLITS}
    X = {}
    for split in SPLITS:
        matrix_ids = np.load(repaired_dir / f"ids_{split}.npy", allow_pickle=True).astype(str)
        if not np.array_equal(matrix_ids, ids[split]):
            raise RuntimeError(f"{species} {split}: feature rows do not match the frozen manifest")
        X[split] = np.asarray(np.load(repaired_dir / f"X_{split}.npy", mmap_mode="r"), dtype=np.float32)
    names = json.loads((repaired_dir / "feature_names.json").read_text(encoding="utf-8"))
    if X["train"].shape[1] != 628 or X["val"].shape[1] != 628 or len(names) != 628:
        raise RuntimeError(f"{species}: expected all 628 original raw feature columns")

    hom_ix = np.asarray([i for i, name in enumerate(names) if name.startswith(("homology30_", "homology50_"))])
    hom_local = np.asarray([
        j for j, index in enumerate(hom_ix)
        if int(names[index].rsplit("_", 1)[1]) >= 2
    ], dtype=np.int64)
    hom_label_ix = hom_ix[hom_local]
    mm_ix = np.asarray([i for i, name in enumerate(names) if name.startswith("mmseqs_dynamic_")])
    ppi_ix = np.asarray([i for i, name in enumerate(names) if name.startswith("ppi_dynamic_")])
    if (len(hom_ix), len(mm_ix), len(ppi_ix)) != (30, 21, 15):
        raise RuntimeError(f"{species}: unexpected dynamic feature counts")

    train = frames["train"].copy()
    train_ids = ids["train"]
    val_ids = ids["val"]
    train_sp = dict(zip(train.record_id.astype(str), train.species.astype(str)))
    query_sp = dict(zip(
        pd.concat([frames["train"], frames["val"]]).record_id.astype(str),
        pd.concat([frames["train"], frames["val"]]).species.astype(str),
    ))
    hits_train = read_hits(species, "train")
    hits_val = read_hits(species, "val")
    edges = read_target_species_edges(species, set(train_ids) | set(val_ids))

    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(train_ids, y["train"]))
    results: list[dict] = []
    prediction_rows: list[dict] = []

    for mode in (
        "strict_within_species",
        "plasmodium_only_label_assisted",
        "cross_species_label_assisted",
        "cross_species_with_pk_oberstaller_labels",
    ):
        oof = np.zeros(len(train), dtype=np.float32)
        fold_metrics = []
        for fold, (inner_ix, hold_ix) in enumerate(folds):
            inner = train.iloc[inner_ix][["record_id", "species", "label"]].copy()
            hold = train.iloc[hold_ix][["record_id", "species"]].copy()
            inner_ids = inner.record_id.astype(str).to_numpy()
            hold_ids = hold.record_id.astype(str).to_numpy()
            if mode == "strict_within_species":
                allowed = inner
            else:
                label_pool = all_train
                if mode == "plasmodium_only_label_assisted":
                    label_pool = label_pool.loc[
                        label_pool.species.astype(str).str.startswith("Plasmodium_")
                    ]
                elif mode == "cross_species_with_pk_oberstaller_labels":
                    label_pool = all_train_ober
                allowed = label_pool.loc[
                    ~label_pool.record_id.astype(str).isin(set(hold_ids))
                ].copy()

            h_inner, h_hold = scaled_homology(inner, hold, allowed)
            label_map = dict(zip(inner_ids, inner.label.eq("essential").astype(float)))
            species_map = dict(zip(inner_ids, inner.species.astype(str)))
            qsp_inner = {rid: query_sp[rid] for rid in inner_ids}
            qsp_hold = {rid: query_sp[rid] for rid in hold_ids}
            dyn_mm_inner = aggregate_mm(hits_train, inner_ids, label_map, species_map, qsp_inner)
            dyn_mm_hold = aggregate_mm(hits_train, hold_ids, label_map, species_map, qsp_hold)
            dyn_ppi_inner = aggregate_ppi(inner_ids, label_map, edges)
            dyn_ppi_hold = aggregate_ppi(hold_ids, label_map, edges)

            x_inner = X["train"][inner_ix].copy()
            x_hold = X["train"][hold_ix].copy()
            x_inner[:, hom_label_ix] = h_inner[:, hom_local]
            x_hold[:, hom_label_ix] = h_hold[:, hom_local]
            x_inner[:, mm_ix] = dyn_mm_inner
            x_hold[:, mm_ix] = dyn_mm_hold
            x_inner[:, ppi_ix] = dyn_ppi_inner
            x_hold[:, ppi_ix] = dyn_ppi_hold
            model = make_model(100 + fold)
            model.fit(x_inner, y["train"][inner_ix])
            oof[hold_ix] = model.predict_proba(x_hold)[:, 1]
            fold_metrics.append({
                "fold": fold,
                "n": int(len(hold_ix)),
                "AUROC": float(roc_auc_score(y["train"][hold_ix], oof[hold_ix])),
                "AUPRC": float(average_precision_score(y["train"][hold_ix], oof[hold_ix])),
            })
        threshold = best_oof_threshold(y["train"], oof)

        # Validation labels are never part of any homology label pool.
        full_query = train[["record_id", "species"]]
        val_query = frames["val"][["record_id", "species"]]
        if mode == "strict_within_species":
            allowed_full = train
        elif mode == "plasmodium_only_label_assisted":
            allowed_full = all_train.loc[
                all_train.species.astype(str).str.startswith("Plasmodium_")
            ].copy()
        elif mode == "cross_species_with_pk_oberstaller_labels":
            allowed_full = all_train_ober
        else:
            allowed_full = all_train
        h_train, h_val = scaled_homology(full_query, val_query, allowed_full)
        labels_full = dict(zip(train_ids, y["train"].astype(float)))
        mm_train = aggregate_mm(hits_train, train_ids, labels_full, train_sp, {rid: query_sp[rid] for rid in train_ids})
        mm_val = aggregate_mm(hits_val, val_ids, labels_full, train_sp, {rid: query_sp[rid] for rid in val_ids})
        ppi_train = aggregate_ppi(train_ids, labels_full, edges)
        ppi_val = aggregate_ppi(val_ids, labels_full, edges)
        x_train = X["train"].copy()
        x_val = X["val"].copy()
        x_train[:, hom_label_ix] = h_train[:, hom_local]
        x_val[:, hom_label_ix] = h_val[:, hom_local]
        x_train[:, mm_ix] = mm_train
        x_val[:, mm_ix] = mm_val
        x_train[:, ppi_ix] = ppi_train
        x_val[:, ppi_ix] = ppi_val
        final_model = make_model(42)
        final_model.fit(x_train, y["train"])
        p_val = final_model.predict_proba(x_val)[:, 1].astype(np.float32)

        results.append({
            "species": species,
            "scope": mode,
            "features": int(x_train.shape[1]),
            "oof_threshold_MCC": threshold,
            "oof": metric_row(y["train"], oof, threshold),
            "validation": metric_row(y["val"], p_val, threshold),
            "fold_metrics": fold_metrics,
            "external_training_species_count": int(allowed_full.species.nunique()),
            "oberstaller_training_source": ober_stats if mode == "cross_species_with_pk_oberstaller_labels" else None,
            "validation_labels_used_for_features_or_threshold": False,
        })
        for index, record_id in enumerate(train_ids):
            prediction_rows.append({
                "species": species, "scope": mode, "split": "train_oof",
                "record_id": record_id, "label": int(y["train"][index]),
                "probability": float(oof[index]), "threshold": threshold,
            })
        for index, record_id in enumerate(val_ids):
            prediction_rows.append({
                "species": species, "scope": mode, "split": "validation",
                "record_id": record_id, "label": int(y["val"][index]),
                "probability": float(p_val[index]), "threshold": threshold,
            })
        print(species, mode, "OOF", results[-1]["oof"], "VAL", results[-1]["validation"], flush=True)

    # Audit that the assisted full-train homology representation reconstructs
    # the legacy matrix values. The legacy matrices are read for train/val only.
    legacy = {}
    for split in SPLITS:
        old_ids = np.load(source_dir / f"ids_{split}.npy", allow_pickle=True).astype(str)
        if not np.array_equal(old_ids, ids[split]):
            raise RuntimeError(f"{species} {split}: legacy matrix IDs differ from frozen manifests")
        old_x = np.asarray(np.load(source_dir / f"X_{split}.npy", mmap_mode="r"), dtype=np.float32)
        if split == "train":
            current_x = x_train
        else:
            current_x = x_val
        legacy[split] = float(np.max(np.abs(old_x[:, hom_ix] - current_x[:, hom_ix])))
    for row in results:
        row["historical_source_vs_current_split_rebuilt_cross_homology_max_abs_diff_train"] = legacy["train"]
        row["historical_source_vs_current_split_rebuilt_cross_homology_max_abs_diff_validation"] = legacy["val"]
    return results, prediction_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_train = read_all_training_metadata()
    all_train_ober, ober_stats = build_oberstaller_training_pool(all_train)
    summaries: list[dict] = []
    predictions: list[dict] = []
    for species in SPECIES:
        rows, pred = run_species(species, all_train, all_train_ober, ober_stats)
        summaries.extend(rows)
        predictions.extend(pred)

    metrics = pd.json_normalize(summaries, sep=".")
    metrics.to_csv(OUT / "scope_comparison_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "scope_comparison_predictions.tsv", sep="\t", index=False)
    manifest = {
        "experiment": "strict and label-assisted homology scopes, including a P. knowlesi Oberstaller-source replacement",
        "split": "frozen within_species train/validation manifests; same split IDs and 95%-deduplicated data",
        "features": "all 628 original raw-scale columns; no PCA or feature selection",
        "model": {**LGBM_PARAMS, "random_state": "42 final; 100+fold for OOF"},
        "folds": {"n_splits": N_SPLITS, "shuffle": True, "random_state": FOLD_SEED},
        "dynamic_features_rebuilt_fold_locally": ["homology30/50", "MMseqs", "PPI"],
        "assisted_scopes": {
            "plasmodium_only_label_assisted": "Plasmodium train labels only; target-species OOF holdout fold excluded; validation features use train labels only",
            "cross_species_label_assisted": "all included species train labels; target-species OOF holdout fold excluded; validation features use train labels only",
            "cross_species_with_pk_oberstaller_labels": "all other species train labels plus definite Oberstaller labels for matched P. knowlesi training records, replacing Elsworth P. knowlesi labels; target-species OOF holdout records excluded",
        },
        "oberstaller_training_source": ober_stats,
        "strict_scope": "target-species train labels only; target-species OOF holdout fold excluded",
        "test_arrays_read": False,
        "test_labels_read": False,
        "validation_threshold_selection": "threshold chosen from pooled train-only OOF MCC",
        "warning": "The fixed validation set was used in previous project iterations and is diagnostic, not an untouched blind estimate.",
        "historical_matrix_warning": "Historical homology features use a different global split; the source-vs-rebuilt max-difference fields are descriptive and are not an exact-equivalence test.",
        "training_species_count": int(all_train.species.nunique()),
        "results": summaries,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved scope comparison to {OUT}", flush=True)


if __name__ == "__main__":
    main()

