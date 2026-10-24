from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_strict_oof_dynamic_stack import aggregate_mm, aggregate_ppi, load_edges
from feature_profiles import BLOCKS, fit_profile_matrix
from protoess_safe_homology import fold_safe_homology

ROOT = Path.cwd()
FEAT = Path(os.environ.get("PROTOESS_SCOPE_FEATURE_ROOT", ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2"))
AUX_FEAT = Path(os.environ.get("PROTOESS_SOURCE_SCOPE_FEATURE_ROOT", ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2"))
MAN = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human"
JOINT = MAN / "joint" / "all_15_species"
# This optional override lets formal-reproduction runs write into an isolated
# result tree without mutating the frozen historical branch outputs.
OUT = Path(os.environ.get("STRICT_OOF_OUT", ROOT / "eukaryote_model_results" / "scope_strict_oof_stack"))
MM_BASE = ROOT / "eukaryote_model_datasets" / "features" / "unified_v3_current_mmseqs_neighbors"


def lgbm(X, y, seed=42):
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.03, num_leaves=31, min_child_samples=30,
        subsample=.85, colsample_bytree=.6, reg_alpha=.1, reg_lambda=5, class_weight="balanced", random_state=seed,
        n_jobs=int(os.environ.get("PROTOESS_LGBM_N_JOBS", "8")), verbosity=-1)
    clf.fit(X, y)
    return clf


def cat(X, y, seed=42):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=.04, loss_function="Logloss", l2_leaf_reg=8,
        random_seed=seed, verbose=False, thread_count=int(os.environ.get("PROTOESS_CATBOOST_THREADS", "4")), allow_writing_files=False)
    clf.fit(X, y)
    return clf


def threshold(y, p):
    grid = np.linspace(.05, .95, 181)
    return float(grid[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid]))])


def metric(y, p, t):
    return {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)), "F1": float(f1_score(y, p >= t, zero_division=0))}


def read_frames(scope, name):
    base = MAN / scope / name
    return {s: pd.read_csv(base / ("test_common_metadata.tsv" if scope == "loso" and s == "test" else f"{s}_metadata.tsv"), sep="\t", dtype={"record_id": str}) for s in ("train", "val", "test")}


def read_hits(scope, name, frames):
    base = AUX_FEAT / scope / name
    rows = []
    for s in ("train", "val", "test"):
        hit_path = (MM_BASE / f"{s}_vs_train.tsv") if scope in {"joint", "loso"} else (base / f"{s}_vs_scope_train.tsv")
        h = pd.read_csv(hit_path, sep="\t", header=None,
            names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"], dtype={"query": str, "target": str})
        h["identity"] = h.pident.astype(float) / 100.0
        h["coverage"] = h.alnlen.astype(float) / h[["qlen", "tlen"]].astype(float).max(axis=1)
        h["score"] = h.identity * h.coverage; h["scope_split"] = s; rows.append(h)
    return pd.concat(rows, ignore_index=True)


def fold_features(X, frames, hits, edges, inner_ids, query_ids, full_static_cols):
    labels = dict(zip(frames["train"].record_id.astype(str), frames["train"].label.eq("essential").astype(float)))
    inner_labels = {k: labels[k] for k in inner_ids}
    target_sp = dict(zip(frames["train"].record_id.astype(str), frames["train"].species.astype(str)))
    query_sp = dict(zip(pd.concat(list(frames.values())).record_id.astype(str), pd.concat(list(frames.values())).species.astype(str)))
    h = hits[hits["query"].isin(set(query_ids))]
    mm = aggregate_mm(h, query_ids, inner_labels, {k: target_sp[k] for k in inner_labels}, {k: query_sp[k] for k in query_ids})
    ppi = aggregate_ppi(query_ids, inner_labels, edges)
    z = X.copy()
    homology = fold_safe_homology(frames, inner_ids, np.asarray(query_ids, dtype=str))
    label_columns = np.r_[2:15, 17:30]
    z[:, 543 + label_columns] = homology[:, label_columns]
    z[:, full_static_cols:full_static_cols + 21] = mm
    z[:, full_static_cols + 21:full_static_cols + 36] = ppi
    return z


def train_one(scope, name, edges):
    out = OUT / scope / name; out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and os.environ.get("STACK_FORCE", "0") != "1": return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    frames = read_frames(scope, name)
    X = {s: np.asarray(np.load(FEAT / scope / name / f"X_{s}.npy"), dtype=np.float32) for s in frames}
    feature_names = json.loads((FEAT / scope / name / "feature_names.json").read_text(encoding="utf-8"))
    profile_path = os.environ.get("PROTOESS_FEATURE_PROFILE_PATH", "")
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path else None
    y = {s: frames[s].label.eq("essential").astype(np.int8).to_numpy() for s in frames}
    ids = {s: frames[s].record_id.astype(str).to_numpy() for s in frames}
    # X layout: 582 static groups + 9 label-free STRING structure + K species bits, then MMseqs21 and PPI15.
    k_species = int(X["train"].shape[1] - 582 - 9 - 21 - 15); static_cols = 582 + 9 + k_species
    hits = read_hits(scope, name, frames)
    train_ids = ids["train"]; skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros((len(train_ids), 4), dtype=np.float32)
    for fold, (inner_ix, hold_ix) in enumerate(skf.split(train_ids, y["train"])):
        inner_ids = set(train_ids[inner_ix]); hold_ids = train_ids[hold_ix]
        Xi = fold_features(X["train"][inner_ix].copy(), frames, hits[hits.scope_split.eq("train")], edges, inner_ids, train_ids[inner_ix], static_cols)
        Xh = fold_features(X["train"][hold_ix].copy(), frames, hits[hits.scope_split.eq("train")], edges, inner_ids, hold_ids, static_cols)
        Xi, Xh, _, _ = fit_profile_matrix(Xi, Xh, feature_names, profile, seed=100 + fold)
        m1 = lgbm(Xi, y["train"][inner_ix], 100 + fold); m2 = cat(Xi, y["train"][inner_ix], 200 + fold)
        oof[hold_ix, 0] = m1.predict_proba(Xh)[:, 1]; oof[hold_ix, 1] = m2.predict_proba(Xh)[:, 1]
        # Species expert prediction within each fold; unseen species falls back to the global branch.
        for sp in sorted(frames["train"].species.astype(str).unique()):
            ii = inner_ix[frames["train"].species.to_numpy()[inner_ix].astype(str) == sp]
            hh = hold_ix[frames["train"].species.to_numpy()[hold_ix].astype(str) == sp]
            if len(ii) >= 30 and len(np.unique(y["train"][ii])) == 2 and len(hh):
                ex = lgbm(Xi[np.isin(inner_ix, ii)], y["train"][ii], 300 + fold)
                oof[hh, 2] = ex.predict_proba(Xh[np.isin(hold_ix, hh)])[:, 1]
        missing = oof[hold_ix, 2] == 0
        oof[hold_ix[missing], 2] = oof[hold_ix[missing], 0]
        prior = y["train"][inner_ix].mean()
        for sp in frames["train"].species.astype(str).unique():
            ii = inner_ix[frames["train"].species.to_numpy()[inner_ix].astype(str) == sp]
            hh = hold_ix[frames["train"].species.to_numpy()[hold_ix].astype(str) == sp]
            oof[hh, 3] = y["train"][ii].mean() if len(ii) else prior
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(oof, y["train"])
    # Full-train base predictions use full-scope dynamic features, which are valid for val/test.
    Xtrain, Xeval, _, _ = fit_profile_matrix(
        X["train"], np.concatenate([X["val"], X["test"]]), feature_names, profile, seed=42
    )
    Xval = Xeval[:len(X["val"])]
    Xtest = Xeval[len(X["val"]):]
    m1 = lgbm(Xtrain, y["train"]); m2 = cat(Xtrain, y["train"])
    pval = np.column_stack([m1.predict_proba(Xval)[:, 1], m2.predict_proba(Xval)[:, 1]])
    ptest = np.column_stack([m1.predict_proba(Xtest)[:, 1], m2.predict_proba(Xtest)[:, 1]])
    exv = np.zeros(len(frames["val"])); ext = np.zeros(len(frames["test"]))
    for sp in sorted(frames["train"].species.astype(str).unique()):
        ii = np.flatnonzero(frames["train"].species.astype(str).to_numpy() == sp); iv = np.flatnonzero(frames["val"].species.astype(str).to_numpy() == sp); it = np.flatnonzero(frames["test"].species.astype(str).to_numpy() == sp)
        if len(ii) >= 30 and len(np.unique(y["train"][ii])) == 2:
            ex = lgbm(Xtrain[ii], y["train"][ii], 342); exv[iv] = ex.predict_proba(Xval[iv])[:, 1] if len(iv) else 0; ext[it] = ex.predict_proba(Xtest[it])[:, 1] if len(it) else 0
    exv[exv == 0] = pval[:, 0][exv == 0]; ext[ext == 0] = ptest[:, 0][ext == 0]
    prior = y["train"].mean(); priors = frames["train"].assign(y=y["train"]).groupby("species").y.mean().to_dict()
    prv = np.asarray([priors.get(s, prior) for s in frames["val"].species], dtype=np.float32); prt = np.asarray([priors.get(s, prior) for s in frames["test"].species], dtype=np.float32)
    Bv = np.column_stack([pval[:, 0], pval[:, 1], exv, prv]); Bt = np.column_stack([ptest[:, 0], ptest[:, 1], ext, prt])
    pv = meta.predict_proba(Bv)[:, 1]; pt = meta.predict_proba(Bt)[:, 1]; th = threshold(y["val"], pv)
    result = {"scope": scope, "name": name, "n_features": int(Xtrain.shape[1]), "feature_profile": profile or {"profile_id": "all_raw", "include_blocks": list(BLOCKS), "pca_components": {}}, "feature_profile_path": profile_path or None, "meta": "logistic_on_3fold_oof", "base_models": ["lgbm", "catboost", "species_expert", "species_prior"], "threshold": th, "oof": metric(y["train"], meta.predict_proba(oof)[:, 1], th), "validation": metric(y["val"], pv, th), "test": metric(y["test"], pt, th)}
    np.save(out / "oof_base.npy", oof); np.save(out / "val_base.npy", Bv); np.save(out / "test_base.npy", Bt); np.save(out / "val_prob.npy", pv); np.save(out / "test_prob.npy", pt); (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(scope, name, result["test"], flush=True); return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joint = {s: pd.read_csv(JOINT / f"{s}_metadata.tsv", sep="\t", dtype={"record_id": str}) for s in ("train", "val", "test")}
    jobs = []
    for scope in ("joint", "clade", "within_species", "loso"):
        for p in sorted((FEAT / scope).iterdir()):
            if p.is_dir() and (p / "DONE.json").exists(): jobs.append((scope, p.name))
    only = os.environ.get("STACK_ONLY", "")
    if only: jobs = [j for j in jobs if f"{j[0]}:{j[1]}" == only]
    active_species = set()
    for scope, name in jobs:
        active_species.update(read_frames(scope, name)["train"].species.astype(str).unique())
    edges = load_edges(joint, active_species)
    rows = [train_one(s, n, edges) for s, n in jobs]
    pd.json_normalize(rows).to_csv(OUT / "scope_strict_oof_stack_metrics.tsv", sep="\t", index=False)


if __name__ == "__main__": main()

