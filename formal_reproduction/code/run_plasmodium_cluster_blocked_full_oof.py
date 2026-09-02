"""Cluster-blocked sensitivity analysis using the locked five-branch full OOF stack.

This is intentionally separate from the earlier three-branch exploratory run.  The
only experimental change from the formal within-species analysis is the outer and
inner group assignment: all members of an MMseqs2 30% or 50% cluster remain in one
fold.  Dynamic MMseqs2 and PPI features are rebuilt within each training fold.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from build_strict_oof_dynamic_stack import TAX, aggregate_mm, aggregate_ppi
from train_scope_transformer_oof import split_groups, train_predict

ROOT = Path.cwd()
MAN = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species"
FEAT = ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2" / "within_species"
MMSEQS = ROOT / "tools" / "mmseqs2" / "mmseqs" / "bin" / "mmseqs.exe"
EDGE_DIR = ROOT / "eukaryote_model_results" / "strict_oof_dynamic_stack" / "mapped_edges"
BASE_OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "cluster_blocked_full_oof"
# Repeated cluster allocations are intentionally written to separate directories.
# With no environment variable this retains the original, published sensitivity run.
_repeat = os.environ.get("FULL_CLUSTER_REPEAT", "").strip()
OUT = BASE_OUT / "repeated_allocations" / _repeat if _repeat else BASE_OUT
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
IDENTITIES = (30, 50)
SEED = int(os.environ.get("FULL_CLUSTER_SEED", "42"))


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lgbm(x: np.ndarray, y: np.ndarray, seed: int) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.60,
        reg_alpha=0.10, reg_lambda=5.0, class_weight="balanced", random_state=seed,
        n_jobs=8, verbosity=-1,
    )
    model.fit(x, y)
    return model


def catboost(x: np.ndarray, y: np.ndarray, seed: int) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.04, loss_function="Logloss",
        l2_leaf_reg=8, random_seed=seed, verbose=False, thread_count=4,
        allow_writing_files=False,
    )
    model.fit(x, y)
    return model


def read_cluster_map(identity: int) -> dict[str, str]:
    path = ROOT / "eukaryote_model_datasets" / f"mmseqs{identity}" / f"cluster{identity}_cluster.tsv"
    data = pd.read_csv(path, sep="\t", header=None, names=["representative", "member"], dtype=str)
    return dict(zip(data.member, data.representative))


def read_all(species: str) -> tuple[pd.DataFrame, np.ndarray]:
    frames, arrays = [], []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(MAN / species / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str})
        frame["formal_split"] = split
        frames.append(frame)
        arrays.append(np.asarray(np.load(FEAT / species / f"X_{split}.npy"), dtype=np.float32))
    return pd.concat(frames, ignore_index=True), np.concatenate(arrays, axis=0)


def all_fasta(species: str, out: Path) -> Path:
    target = out / "all_labelled_proteins.fasta"
    if target.exists():
        return target
    seen: set[str] = set()
    with target.open("w", encoding="utf-8", newline="\n") as write:
        for split in ("train", "val", "test"):
            path = FEAT / species / f"{split}.fasta"
            header, seq = None, []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(">"):
                    if header is not None and header not in seen:
                        write.write(f">{header}\n{''.join(seq)}\n")
                        seen.add(header)
                    header, seq = line[1:].split()[0], []
                else:
                    seq.append(line.strip())
            if header is not None and header not in seen:
                write.write(f">{header}\n{''.join(seq)}\n")
                seen.add(header)
    return target


def all_vs_all(species: str, out: Path) -> pd.DataFrame:
    result = out / "all_vs_all.tsv"
    cols = ["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"]
    if not result.exists():
        fasta = all_fasta(species, out)
        tmp = out / "tmp_all_vs_all"
        subprocess.run([
            str(MMSEQS), "easy-search", str(fasta), str(fasta), str(result), str(tmp),
            "--format-output", "query,target,pident,alnlen,evalue,bits,qlen,tlen",
            "--max-seqs", "100", "-s", "7.5", "-e", "0.01", "--threads", "8",
        ], check=True)
    h = pd.read_csv(result, sep="\t", names=cols, dtype={"query": str, "target": str})
    h["identity"] = h.pident.astype(float) / 100.0
    h["coverage"] = h.alnlen.astype(float) / h[["qlen", "tlen"]].astype(float).max(axis=1)
    h["score"] = h.identity * h.coverage
    return h


def edge_bundle(species: str):
    tax = TAX[species]
    path = EDGE_DIR / f"{tax}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached STRING edge map: {path}")
    z = np.load(path, allow_pickle=True)
    return [(species, z["src"].astype(str), z["dst"].astype(str), z["score"].astype(np.float32))]


def assign_folds(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, hold) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        folds[hold] = fold
    if (folds < 0).any():
        raise RuntimeError("Cluster fold assignment incomplete")
    return folds


def dynamic_x(static_x: np.ndarray, query_ids: np.ndarray, target_ids: np.ndarray, target_y: np.ndarray,
              id_to_species: dict[str, str], hits: pd.DataFrame, edges) -> np.ndarray:
    labels = dict(zip(target_ids.astype(str), target_y.astype(float)))
    target_species = {rid: id_to_species[rid] for rid in labels}
    query_species = {rid: id_to_species[rid] for rid in query_ids.astype(str)}
    mm = aggregate_mm(hits[hits["query"].isin(set(query_ids.astype(str)))], query_ids, labels, target_species, query_species)
    ppi = aggregate_ppi(query_ids, labels, edges)
    x = static_x.copy()
    k = int(x.shape[1] - 582 - 9 - 21 - 15)
    static_cols = 582 + 9 + k
    x[:, static_cols:static_cols + 21] = mm
    x[:, static_cols + 21:static_cols + 36] = ppi
    return x


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    return float(grid[int(np.argmax([matthews_corrcoef(y, p >= threshold) for threshold in grid]))])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)), "MCC": float(matthews_corrcoef(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def bootstrap(y: np.ndarray, p: np.ndarray, repeats: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(20260812)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    auc, ap = [], []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)])
        auc.append(roc_auc_score(y[ix], p[ix])); ap.append(average_precision_score(y[ix], p[ix]))
    return {"AUROC_ci_low": float(np.quantile(auc, .025)), "AUROC_ci_high": float(np.quantile(auc, .975)),
            "AUPRC_ci_low": float(np.quantile(ap, .025)), "AUPRC_ci_high": float(np.quantile(ap, .975))}


def run_one(species: str, identity: int) -> dict[str, object]:
    out = OUT / f"mmseqs{identity}" / species
    out.mkdir(parents=True, exist_ok=True)
    finished = out / "metrics.json"
    if finished.exists() and os.environ.get("FULL_CLUSTER_FORCE", "0") != "1":
        return json.loads(finished.read_text(encoding="utf-8"))
    meta, raw_x = read_all(species)
    y = meta.label.eq("essential").astype(np.int8).to_numpy()
    groups = meta.record_id.astype(str).map(read_cluster_map(identity)).fillna(meta.record_id.astype(str)).to_numpy()
    folds = assign_folds(y, groups)
    meta["homology_cluster"] = groups; meta["cluster_fold"] = folds
    meta["cluster_split"] = np.where(folds == 0, "test", np.where(folds == 1, "validation", "training"))
    meta.to_csv(out / "cluster_blocked_manifest.tsv", sep="\t", index=False)
    train, val, test = np.flatnonzero(folds >= 2), np.flatnonzero(folds == 1), np.flatnonzero(folds == 0)
    ids = meta.record_id.astype(str).to_numpy(); id_to_species = dict(zip(ids, meta.species.astype(str)))
    hits, edges = all_vs_all(species, out), edge_bundle(species)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 5), dtype=np.float32)
    for fold, (inner, hold) in enumerate(inner_cv.split(train, y[train], groups[train])):
        inner_global, hold_global = train[inner], train[hold]
        xi = dynamic_x(raw_x[inner_global], ids[inner_global], ids[inner_global], y[inner_global], id_to_species, hits, edges)
        xh = dynamic_x(raw_x[hold_global], ids[hold_global], ids[inner_global], y[inner_global], id_to_species, hits, edges)
        global_model = lgbm(xi, y[inner_global], 100 + fold)
        cat_model = catboost(xi, y[inner_global], 200 + fold)
        # Within a single species, the species expert is trained separately but has the same eligible records.
        expert_model = lgbm(xi, y[inner_global], 300 + fold)
        oof[hold, 0] = global_model.predict_proba(xh)[:, 1]
        oof[hold, 1] = cat_model.predict_proba(xh)[:, 1]
        oof[hold, 2] = expert_model.predict_proba(xh)[:, 1]
        oof[hold, 3] = y[inner_global].mean()
        oof[hold, 4] = train_predict(
            split_groups(xi, 1), y[inner_global], split_groups(xh, 1), y[hold_global], split_groups(xh, 1), 400 + fold
        )[0]
        print(f"{species} {identity}% OOF fold {fold + 1}/3 complete", flush=True)
    x_train = dynamic_x(raw_x[train], ids[train], ids[train], y[train], id_to_species, hits, edges)
    x_val = dynamic_x(raw_x[val], ids[val], ids[train], y[train], id_to_species, hits, edges)
    x_test = dynamic_x(raw_x[test], ids[test], ids[train], y[train], id_to_species, hits, edges)
    global_model, cat_model, expert_model = lgbm(x_train, y[train], SEED), catboost(x_train, y[train], SEED), lgbm(x_train, y[train], SEED + 1)
    transformer_val, transformer_test, _ = train_predict(split_groups(x_train, 1), y[train], split_groups(x_val, 1), y[val], split_groups(x_test, 1), 900 + identity)
    val_base = np.column_stack([global_model.predict_proba(x_val)[:, 1], cat_model.predict_proba(x_val)[:, 1], expert_model.predict_proba(x_val)[:, 1], np.repeat(y[train].mean(), len(val)), transformer_val])
    test_base = np.column_stack([global_model.predict_proba(x_test)[:, 1], cat_model.predict_proba(x_test)[:, 1], expert_model.predict_proba(x_test)[:, 1], np.repeat(y[train].mean(), len(test)), transformer_test])
    meta_model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED).fit(oof, y[train])
    p_val, p_test = meta_model.predict_proba(val_base)[:, 1], meta_model.predict_proba(test_base)[:, 1]
    threshold = choose_threshold(y[val], p_val)
    result: dict[str, object] = {
        "species": species, "cluster_identity_percent": identity,
        "model": "locked full OOF stack: Transformer + LightGBM + CatBoost + species expert + species prior + logistic meta-learner",
        "feature_mode": "formal 628-dimensional feature set; fold-specific dynamic MMseqs2 and PPI features",
        "n_features": int(raw_x.shape[1]), "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "n_train_clusters": int(pd.Series(groups[train]).nunique()), "n_val_clusters": int(pd.Series(groups[val]).nunique()), "n_test_clusters": int(pd.Series(groups[test]).nunique()),
        "threshold": threshold, **{f"val_{k}": v for k, v in metrics(y[val], p_val, threshold).items()},
        **{f"test_{k}": v for k, v in metrics(y[test], p_test, threshold).items()}, **bootstrap(y[test], p_test),
    }
    np.save(out / "oof_base_full.npy", oof); np.save(out / "val_prob.npy", p_val); np.save(out / "test_prob.npy", p_test)
    prediction = meta.loc[test, ["record_id", "species", "gene_id", "label", "homology_cluster"]].copy()
    prediction["probability"] = p_test; prediction["threshold"] = threshold; prediction["predicted_label"] = np.where(p_test >= threshold, "essential", "nonessential")
    prediction.to_csv(out / "test_predictions.tsv", sep="\t", index=False)
    finished.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    seed_all(); OUT.mkdir(parents=True, exist_ok=True)
    only_species, only_identity = os.environ.get("FULL_CLUSTER_SPECIES"), os.environ.get("FULL_CLUSTER_IDENTITY")
    state = {"architecture": "locked_formal_full_oof", "jobs": []}
    for identity in IDENTITIES:
        if only_identity and identity != int(only_identity):
            continue
        for species in SPECIES:
            if only_species and species != only_species:
                continue
            state["jobs"].append({"species": species, "identity": identity, "status": "running"})
            (OUT / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            try:
                result = run_one(species, identity)
                state["jobs"][-1].update({"status": "complete", "test_AUROC": result["test_AUROC"], "test_AUPRC": result["test_AUPRC"]})
            except Exception as exc:
                state["jobs"][-1].update({"status": "failed", "error": repr(exc)})
                (OUT / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
                raise
            (OUT / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("mmseqs*/*/metrics.json"))]
    pd.DataFrame(rows).sort_values(["cluster_identity_percent", "species"]).to_csv(OUT / "full_oof_cluster_blocked_metrics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
