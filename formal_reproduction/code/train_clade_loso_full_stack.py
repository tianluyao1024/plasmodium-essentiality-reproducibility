from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_strict_oof_dynamic_stack import load_edges
from train_scope_strict_oof_stack import cat, fold_features, lgbm, metric, threshold
from train_scope_transformer_oof import split_groups, train_predict

ROOT = Path.cwd()
MAN = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "clade"
FEAT = ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2" / "clade"
OUT = ROOT / "eukaryote_model_results" / os.environ.get("CLOSO_OUT_SUBDIR", "clade_loso_full_stack")
JOINT = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "joint" / "all_15_species"

CLADES = ("animal", "fungi", "plant", "protozoa")


def perturb_training_labels(frames, y):
    """Apply a prevalence-preserving, source-species-stratified label exchange.

    This optional reviewer-sensitivity mode never changes validation/test labels.
    Within every source species it exchanges equal-sized essential and
    non-essential subsets, thereby preserving each source species' prevalence.
    """
    fraction = float(os.environ.get("CLOSO_TRAIN_LABEL_NOISE", "0"))
    seed = int(os.environ.get("CLOSO_NOISE_SEED", "0"))
    if fraction <= 0:
        return pd.DataFrame(columns=["record_id", "species", "original_label", "perturbed_label", "noise_fraction", "noise_seed"])
    if not 0 < fraction < 1:
        raise ValueError("CLOSO_TRAIN_LABEL_NOISE must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train = frames["train"].copy()
    noisy = y["train"].copy()
    changed = []
    for species in sorted(train.species.astype(str).unique()):
        idx = np.flatnonzero(train.species.astype(str).to_numpy() == species)
        pos = idx[noisy[idx] == 1]
        neg = idx[noisy[idx] == 0]
        # Two exchanged sets give the requested fraction of changed labels.
        n_exchange = min(len(pos), len(neg), int(round(fraction * len(idx) / 2)))
        if n_exchange == 0:
            continue
        selected_pos = rng.choice(pos, size=n_exchange, replace=False)
        selected_neg = rng.choice(neg, size=n_exchange, replace=False)
        chosen = np.concatenate([selected_pos, selected_neg])
        original = noisy[chosen].copy()
        noisy[chosen] = 1 - noisy[chosen]
        for j, before in zip(chosen, original):
            changed.append({
                "record_id": str(train.record_id.iloc[j]), "species": species,
                "original_label": "essential" if before else "nonessential",
                "perturbed_label": "essential" if noisy[j] else "nonessential",
                "noise_fraction": fraction, "noise_seed": seed,
            })
    if not changed:
        raise RuntimeError("No training labels were exchanged for requested noise fraction")
    y["train"] = noisy
    train.loc[:, "label"] = np.where(noisy == 1, "essential", "nonessential")
    frames["train"] = train
    return pd.DataFrame(changed)


def load_clade(name):
    base = MAN / name
    frames = {s: pd.read_csv(base / f"{s}_metadata.tsv", sep="\t", dtype={"record_id": str}) for s in ("train", "val", "test")}
    arrays = {s: np.asarray(np.load(FEAT / name / f"X_{s}.npy"), dtype=np.float32) for s in frames}
    static_cols = arrays["train"].shape[1] - 21 - 15
    return frames, arrays, static_cols


def task_data(name, held, frames, arrays, static_cols, hits):
    """Create a held-species transfer task without exposing held labels.

    ``CLOSO_SOURCE_MODE=malaria_only`` is a deliberate ablation used only for
    Plasmodium held-out tests: it removes the distant Tg source while retaining
    exactly the same held-out test records. The default is the original
    all-other-species clade LOSO design.
    """
    out_f, out_x = {}, {}
    source_mode = os.environ.get("CLOSO_SOURCE_MODE", "all_other_species")
    if source_mode == "malaria_only":
        malaria = {"Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi"}
        if held not in malaria:
            raise ValueError("malaria_only source mode is defined only for a held-out Plasmodium species")
        allowed = malaria - {held}
        train_mask = frames["train"].species.astype(str).isin(allowed).to_numpy()
        val_mask = frames["val"].species.astype(str).isin(allowed).to_numpy()
    elif source_mode == "all_other_species":
        train_mask = (frames["train"].species.astype(str).to_numpy() != held)
        val_mask = (frames["val"].species.astype(str).to_numpy() != held)
    else:
        raise ValueError(f"Unknown CLOSO_SOURCE_MODE={source_mode!r}")
    test_parts = []
    for s in ("train", "val", "test"):
        m = frames[s].species.astype(str).to_numpy() == held
        if m.any():
            test_parts.append((frames[s].loc[m].copy(), arrays[s][m]))
    out_f["train"] = frames["train"].loc[train_mask].reset_index(drop=True)
    out_f["val"] = frames["val"].loc[val_mask].reset_index(drop=True)
    out_f["test"] = pd.concat([x[0] for x in test_parts], ignore_index=True)
    def make_x(parts):
        z = np.concatenate([x[:, :static_cols] for x in parts], axis=0)
        return np.concatenate([z, np.zeros((len(z), 36), dtype=np.float32)], axis=1)
    out_x["train"] = make_x([arrays["train"][train_mask]])
    out_x["val"] = make_x([arrays["val"][val_mask]])
    out_x["test"] = make_x([x[1] for x in test_parts])

    # This reviewer control leaves all molecular and label-free network
    # features intact, while removing the four clade-level species one-hot
    # columns. The expert and prevalence branches are disabled in run_task.
    if os.environ.get("CLOSO_SHARED_FEATURES_ONLY", "0") == "1":
        species_bits = static_cols - 591
        if species_bits <= 0:
            raise RuntimeError(f"Expected clade-level species one-hot columns, found {species_bits}")
        for split, values in out_x.items():
            static = np.concatenate([values[:, :591], values[:, static_cols:]], axis=1)
            out_x[split] = static.astype(np.float32, copy=False)
        static_cols = 591
    return out_f, out_x, static_cols


def run_task(clade, held, frames, arrays, static_cols, hits, edges):
    out = OUT / clade / held
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and os.environ.get("CLOSO_FORCE", "0") != "1":
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    f, X, static_cols = task_data(clade, held, frames, arrays, static_cols, hits)
    y = {s: f[s].label.eq("essential").astype(np.int8).to_numpy() for s in f}
    label_audit = perturb_training_labels(f, y)
    if len(label_audit):
        label_audit.to_csv(out / "training_label_exchange_audit.tsv", sep="\t", index=False)
    ids = {s: f[s].record_id.astype(str).to_numpy() for s in f}
    train_ids = ids["train"]
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    shared_only = os.environ.get("CLOSO_SHARED_FEATURES_ONLY", "0") == "1"
    tree_oof = np.zeros((len(train_ids), 2 if shared_only else 4), dtype=np.float32)
    tr_hits = hits[hits.scope_split.eq("train")]
    for fold, (ii, hh) in enumerate(skf.split(train_ids, y["train"])):
        inner = set(train_ids[ii]); hold = train_ids[hh]
        xi = fold_features(X["train"][ii].copy(), f, tr_hits, edges, inner, train_ids[ii], static_cols)
        xh = fold_features(X["train"][hh].copy(), f, tr_hits, edges, inner, hold, static_cols)
        m1, m2 = lgbm(xi, y["train"][ii], 100 + fold), cat(xi, y["train"][ii], 200 + fold)
        tree_oof[hh, 0] = m1.predict_proba(xh)[:, 1]; tree_oof[hh, 1] = m2.predict_proba(xh)[:, 1]
        if not shared_only:
            for sp in sorted(f["train"].species.astype(str).unique()):
                a = ii[f["train"].species.to_numpy()[ii].astype(str) == sp]; b = hh[f["train"].species.to_numpy()[hh].astype(str) == sp]
                if len(a) >= 30 and len(np.unique(y["train"][a])) == 2 and len(b):
                    ex = lgbm(xi[np.isin(ii, a)], y["train"][a], 300 + fold)
                    tree_oof[b, 2] = ex.predict_proba(xh[np.isin(hh, b)])[:, 1]
                if len(b): tree_oof[b, 3] = y["train"][a].mean() if len(a) else y["train"][ii].mean()
            tree_oof[hh, 2] = np.where(tree_oof[hh, 2] == 0, tree_oof[hh, 0], tree_oof[hh, 2])
    meta_tree = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42).fit(tree_oof, y["train"])
    full_train = fold_features(X["train"].copy(), f, tr_hits, edges, set(train_ids), train_ids, static_cols)
    valx = fold_features(X["val"].copy(), f, hits, edges, set(train_ids), ids["val"], static_cols)
    testx = fold_features(X["test"].copy(), f, hits, edges, set(train_ids), ids["test"], static_cols)
    a, b = lgbm(full_train, y["train"]), cat(full_train, y["train"])
    tree_val = np.column_stack([a.predict_proba(valx)[:, 1], b.predict_proba(valx)[:, 1]])
    tree_test = np.column_stack([a.predict_proba(testx)[:, 1], b.predict_proba(testx)[:, 1]])
    if not shared_only:
        tree_val = np.column_stack([tree_val, tree_val[:, 0], np.full(len(valx), y["train"].mean())])
        tree_test = np.column_stack([tree_test, tree_test[:, 0], np.full(len(testx), y["train"].mean())])
        tree_val[:, 2] = tree_val[:, 0]; tree_test[:, 2] = tree_test[:, 0]
    tree_val_p = meta_tree.predict_proba(tree_val)[:, 1]; tree_test_p = meta_tree.predict_proba(tree_test)[:, 1]
    tr = {"train": full_train, "val": valx, "test": testx}
    groups = {s: split_groups(tr[s], int(static_cols - 591)) for s in tr}
    trans_oof = np.zeros(len(train_ids), dtype=np.float32)
    for fold, (ii, hh) in enumerate(skf.split(train_ids, y["train"])):
        trans_oof[hh] = train_predict([g[ii] for g in groups["train"]], y["train"][ii], [g[hh] for g in groups["train"]], y["train"][hh], [g[hh] for g in groups["train"]], 42 + fold)[0]
    _, trans_val, _ = train_predict(groups["train"], y["train"], groups["val"], y["val"], groups["val"], 42)
    _, trans_test, _ = train_predict(groups["train"], y["train"], groups["val"], y["val"], groups["test"], 42)
    oof = np.column_stack([tree_oof, trans_oof]); val = np.column_stack([tree_val, trans_val]); test = np.column_stack([tree_test, trans_test])
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42).fit(oof, y["train"])
    pv, pt = meta.predict_proba(val)[:, 1], meta.predict_proba(test)[:, 1]; th = threshold(y["val"], pv)
    result = {"clade": clade, "held_out": held, "source_mode": os.environ.get("CLOSO_SOURCE_MODE", "all_other_species"), "shared_molecular_features_only": shared_only, "species_onehot_removed": shared_only, "species_expert_removed": shared_only, "species_prevalence_removed": shared_only, "training_label_exchange_fraction": float(os.environ.get("CLOSO_TRAIN_LABEL_NOISE", "0")), "training_label_exchange_seed": int(os.environ.get("CLOSO_NOISE_SEED", "0")), "training_records_exchanged": int(len(label_audit)), "test": metric(y["test"], pt, th), "validation": metric(y["val"], pv, th), "oof": metric(y["train"], meta.predict_proba(oof)[:, 1], th), "threshold": th}
    np.save(out / "oof_prob.npy", meta.predict_proba(oof)[:, 1]); np.save(out / "val_prob.npy", pv); np.save(out / "test_prob.npy", pt)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(clade, held, result["test"], flush=True); return result


def main():
    joint = {s: pd.read_csv(JOINT / f"{s}_metadata.tsv", sep="\t", dtype={"record_id": str}) for s in ("train", "val", "test")}
    edges = load_edges(joint); rows = []
    for clade in CLADES:
        frames, arrays, static_cols = load_clade(clade)
        hits = __import__("train_scope_strict_oof_stack").read_hits("clade", clade, frames)
        for held in sorted(set(pd.concat(frames.values()).species.astype(str))):
            only = os.environ.get("CLOSO_ONLY", "")
            if only and only != f"{clade}:{held}":
                continue
            rows.append(run_task(clade, held, frames, arrays, static_cols, hits, edges))
    OUT.mkdir(parents=True, exist_ok=True); pd.json_normalize(rows).to_csv(OUT / "clade_loso_full_stack_metrics.tsv", sep="\t", index=False)


if __name__ == "__main__": main()
