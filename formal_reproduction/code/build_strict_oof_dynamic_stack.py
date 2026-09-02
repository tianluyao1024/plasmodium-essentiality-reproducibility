from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path.cwd()
ALIGN = ROOT / "eukaryote_model_datasets" / "features" / "unified_v3_aligned_features"
PPI_DIR = ROOT / "downloads" / "string_v12"
MM_DIR = ROOT / "eukaryote_model_datasets" / "features" / "unified_v3_current_mmseqs_neighbors"
PPI_FEATURE_DIR = ROOT / "eukaryote_model_datasets" / "features" / "unified_v3_current_string_features"
MAN = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "joint" / "all_15_species"
OUT = ROOT / "eukaryote_model_results" / "strict_oof_dynamic_stack"
EDGE_DIR = OUT / "mapped_edges"

TAX = {
    "Arabidopsis_thaliana": 3702, "Caenorhabditis_elegans": 6239,
    "Candida_albicans__SC5314": 237561, "Candida_glabrata__CBS138": 284593,
    "Drosophila_melanogaster": 7227, "Homo_sapiens": 9606, "Mus_musculus": 10090,
    "Oryza_sativa_japonica": 39947,
    "Pichia_pastoris___Komagataella_phaffii__species_level": 644223,
    "Plasmodium_berghei": 5823, "Plasmodium_falciparum": 36329,
    "Plasmodium_knowlesi": 5851, "Saccharomyces_cerevisiae__species_level": 4932,
    "Schizosaccharomyces_pombe__species_level": 284812, "Toxoplasma_gondii": 432359,
}
STATIC = ("esm2", "prott5", "basic_seq", "go_ipr_pfam", "kmer_graph", "homology30", "homology50", "string_network")


def load_frames():
    return {s: pd.read_csv(MAN / f"{s}_metadata.tsv", sep="\t") for s in ("train", "val", "test")}


def load_static(frames):
    master = np.asarray(np.load(ALIGN / "master_train_ids.npy", allow_pickle=True)).astype(str)
    pos = {x: i for i, x in enumerate(master)}
    idx = {s: np.asarray([pos[str(x)] for x in frames[s].record_id], dtype=np.int64) for s in frames}
    parts = {s: [] for s in frames}
    for g in STATIC:
        x = np.asarray(np.load(ALIGN / g / "train_emb_aligned.npy", mmap_mode="r"), dtype=np.float32)
        m = np.load(ALIGN / g / "train_present.npy", mmap_mode="r").astype(bool)
        tr = np.nan_to_num(x[idx["train"]].copy(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = m[idx["train"]]
        mean = np.where(valid[:, None], tr, 0).sum(0) / valid.sum(0).clip(min=1)
        centered = np.where(valid[:, None], tr - mean, 0).astype(np.float64)
        std = np.sqrt((centered ** 2).sum(0) / valid.sum(0).clip(min=1)).astype(np.float32).clip(min=1e-6)
        svd = None
        if tr.shape[1] > 128:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=128, n_iter=4, random_state=42)
            svd.fit(tr)
        for s in frames:
            z = np.nan_to_num(x[idx[s]].copy(), nan=0.0, posinf=0.0, neginf=0.0)
            z = np.where(m[idx[s], None], z, mean)
            z = ((z - mean) / std).astype(np.float32)
            if svd is not None:
                z = svd.transform(z).astype(np.float32)
            parts[s].append(z)
    species = sorted(frames["train"].species.unique())
    sp_to = {x: i for i, x in enumerate(species)}
    for s in frames:
        one = np.zeros((len(frames[s]), len(species)), dtype=np.float32)
        for i, x in enumerate(frames[s].species):
            one[i, sp_to[x]] = 1
        parts[s].append(one)
        # Nine STRING mapping/network features are label-free; keep them in the
        # strict OOF model so its input space matches the formal dynamic model.
        parts[s].append(np.load(PPI_FEATURE_DIR / f"{s}_emb.npy").astype(np.float32)[:, :9])
    return {s: np.concatenate(parts[s], axis=1) for s in frames}


def load_mm_hits(split: str, train_ids: set[str]):
    path = MM_DIR / f"{split}_vs_train.tsv"
    h = pd.read_csv(path, sep="\t", header=None, names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"])
    h = h[h.target.isin(train_ids)].copy()
    h["identity"] = h.pident.astype(float) / 100.0
    h["coverage"] = h.alnlen.astype(float) / h[["qlen", "tlen"]].astype(float).max(axis=1)
    h["score"] = h.identity * h.coverage
    return h


def aggregate_mm(h: pd.DataFrame, query_ids: np.ndarray, target_y: dict[str, float], target_sp: dict[str, str], query_sp: dict[str, str]) -> np.ndarray:
    ids = pd.Index(query_ids.astype(str), name="query")
    z = h[h.target.isin(target_y)].copy()
    z = z[z["query"] != z["target"]]
    z["y"] = z.target.map(target_y).astype(float)
    names = []
    out = pd.DataFrame(index=ids)
    for threshold in (0.30, 0.50, 0.70):
        tag = int(threshold * 100)
        names += [f"nn{tag}_log_n", f"nn{tag}_weighted_pos_ratio", f"nn{tag}_max_pos_score", f"nn{tag}_max_neg_score", f"nn{tag}_same_species_ratio", f"nn{tag}_other_species_ratio"]
        q = z[(z.identity >= threshold) & (z.coverage >= 0.30)].copy()
        q["w"] = np.maximum(q.score, 1e-4) ** 3
        q["wy"] = q.w * q.y
        q["same"] = q.target.map(target_sp).eq(q["query"].map(query_sp))
        agg = q.groupby("query").agg(n=("target", "size"), w=("w", "sum"), wy=("wy", "sum"))
        mx = q.groupby(["query", "y"])["score"].max().unstack(fill_value=0.0)
        same = q[q.same].groupby("query")[["w", "wy"]].sum()
        other = q[~q.same].groupby("query")[["w", "wy"]].sum()
        out[f"nn{tag}_log_n"] = np.log1p(agg.n).reindex(ids).fillna(0.0)
        out[f"nn{tag}_weighted_pos_ratio"] = (agg.wy / agg.w).reindex(ids).fillna(0.5)
        out[f"nn{tag}_max_pos_score"] = mx.get(1.0, pd.Series(dtype=float)).reindex(ids).fillna(0.0)
        out[f"nn{tag}_max_neg_score"] = mx.get(0.0, pd.Series(dtype=float)).reindex(ids).fillna(0.0)
        out[f"nn{tag}_same_species_ratio"] = ((same.wy + 0.5) / (same.w + 1.0)).reindex(ids).fillna(0.5)
        out[f"nn{tag}_other_species_ratio"] = ((other.wy + 0.5) / (other.w + 1.0)).reindex(ids).fillna(0.5)
    best = z.sort_values(["query", "score", "bits"], ascending=[True, False, False]).drop_duplicates("query").set_index("query")
    out["best_hit_identity"] = best.identity.reindex(ids).fillna(0.0)
    out["best_hit_coverage"] = best.coverage.reindex(ids).fillna(0.0)
    out["best_hit_bits_per_length"] = (best.bits / best[["qlen", "tlen"]].astype(float).max(axis=1)).reindex(ids).fillna(0.0)
    return out.to_numpy(np.float32)


def build_mapped_edges(frames):
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    all_ids = set(pd.concat([frames[s].record_id for s in frames]).astype(str))
    for species, tax in TAX.items():
        out = EDGE_DIR / f"{tax}.npz"
        if out.exists():
            continue
        records = set(frames["train"].loc[frames["train"].species == species, "record_id"].astype(str))
        records |= set(frames["val"].loc[frames["val"].species == species, "record_id"].astype(str))
        records |= set(frames["test"].loc[frames["test"].species == species, "record_id"].astype(str))
        mp = pd.read_csv(PPI_DIR / f"current_map_{tax}.tsv", sep="\t", header=None, names=["record", "sid", "pid", "aln", "e", "bits", "qlen", "tlen"])
        mp["record"] = mp.record.astype(str); mp["sid"] = mp.sid.astype(str)
        mp = mp[mp.record.isin(records)].drop_duplicates("record")
        sid_to_rec = dict(zip(mp.sid, mp.record))
        src, dst, score = [], [], []
        import gzip
        with gzip.open(PPI_DIR / f"{tax}.protein.links.v12.0.txt.gz", "rt", encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                a, b, sc = line.split()
                ra, rb = sid_to_rec.get(a), sid_to_rec.get(b)
                if ra is not None and rb is not None and ra != rb:
                    src.append(ra); dst.append(rb); score.append(float(sc) / 1000.0)
        np.savez_compressed(out, src=np.asarray(src, dtype=object), dst=np.asarray(dst, dtype=object), score=np.asarray(score, dtype=np.float32))
        print("edges", species, len(src), flush=True)


def load_edges(frames):
    build_mapped_edges(frames)
    edges = []
    for species, tax in TAX.items():
        z = np.load(EDGE_DIR / f"{tax}.npz", allow_pickle=True)
        edges.append((species, z["src"].astype(str), z["dst"].astype(str), z["score"].astype(np.float32)))
    return edges


def aggregate_ppi(query_ids: np.ndarray, target_y: dict[str, float], edges) -> np.ndarray:
    rows = {str(x): [0.0] * 15 for x in query_ids}
    qset = set(rows)
    for _, src, dst, score in edges:
        for a, b, sc in zip(src, dst, score):
            if a not in qset or b not in target_y:
                continue
            for j, th in enumerate((0.4, 0.7, 0.9)):
                if sc >= th:
                    rows[a][j * 5] += 1.0
                    rows[a][j * 5 + 1] += sc * target_y[b]
                    rows[a][j * 5 + 2] += sc
                    rows[a][j * 5 + 3] = max(rows[a][j * 5 + 3], sc * target_y[b])
                    rows[a][j * 5 + 4] = max(rows[a][j * 5 + 4], sc * (1.0 - target_y[b]))
    out = np.zeros((len(query_ids), 15), dtype=np.float32)
    for i, rid in enumerate(query_ids.astype(str)):
        for j, th in enumerate((0.4, 0.7, 0.9)):
            n, wy, w, mp, mn = rows[rid][j * 5:j * 5 + 5]
            out[i, j * 5] = np.log1p(n)
            out[i, j * 5 + 1] = wy / max(w, 1e-6) if n else 0.5
            out[i, j * 5 + 2] = mp
            out[i, j * 5 + 3] = mn
            out[i, j * 5 + 4] = n
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    static = load_static(frames)
    edges = load_edges(frames)
    train = frames["train"]; val = frames["val"]; test = frames["test"]
    train_ids = train.record_id.astype(str).to_numpy(); val_ids = val.record_id.astype(str).to_numpy(); test_ids = test.record_id.astype(str).to_numpy()
    y_train = train.label.eq("essential").astype(np.int8).to_numpy(); y_val = val.label.eq("essential").astype(np.int8).to_numpy(); y_test = test.label.eq("essential").astype(np.int8).to_numpy()
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(train_ids, y_train))
    mm_train = load_mm_hits("train", set(train_ids)); mm_val = load_mm_hits("val", set(train_ids)); mm_test = load_mm_hits("test", set(train_ids))
    train_sp = dict(zip(train_ids, train.species.astype(str)))
    val_sp = dict(zip(val_ids, val.species.astype(str)))
    test_sp = dict(zip(test_ids, test.species.astype(str)))
    dyn_oof = np.zeros((len(train), 36), dtype=np.float32); oof = np.zeros(len(train), dtype=np.float32)
    best_iters = []
    for fold, (inner, hold) in enumerate(folds):
        inner_ids = set(train_ids[inner]); labels = dict(zip(train_ids[inner], y_train[inner].astype(float)))
        dyn_oof[inner, :21] = aggregate_mm(mm_train[mm_train["query"].isin(inner_ids)], train_ids[inner], labels, {k: train_sp[k] for k in inner_ids}, {k: train_sp[k] for k in inner_ids})
        dyn_oof[inner, 21:] = aggregate_ppi(train_ids[inner], labels, edges)
        dyn_oof[hold, :21] = aggregate_mm(mm_train[mm_train["query"].isin(set(train_ids[hold]))], train_ids[hold], labels, train_sp, {k: train_sp[k] for k in train_ids[hold]})
        dyn_oof[hold, 21:] = aggregate_ppi(train_ids[hold], labels, edges)
        Xtr = np.concatenate([static["train"][inner], dyn_oof[inner]], axis=1)
        Xho = np.concatenate([static["train"][hold], dyn_oof[hold]], axis=1)
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=1600, learning_rate=.025, num_leaves=31, min_child_samples=30, colsample_bytree=.65, reg_alpha=.1, reg_lambda=5, class_weight="balanced", random_state=100 + fold, n_jobs=8, verbosity=-1)
        clf.fit(Xtr, y_train[inner], eval_set=[(Xho, y_train[hold])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[hold] = clf.predict_proba(Xho)[:, 1]
        best_iters.append(int(clf.best_iteration_ or 1600))
        print("fold", fold, "iter", clf.best_iteration_, "oof_fold_auc", roc_auc_score(y_train[hold], oof[hold]), flush=True)
    labels_full = dict(zip(train_ids, y_train.astype(float)))
    dyn_val_arr = np.concatenate([aggregate_mm(mm_val, val_ids, labels_full, train_sp, val_sp), aggregate_ppi(val_ids, labels_full, edges)], axis=1)
    dyn_test_arr = np.concatenate([aggregate_mm(mm_test, test_ids, labels_full, train_sp, test_sp), aggregate_ppi(test_ids, labels_full, edges)], axis=1)
    np.save(OUT / "dyn_oof.npy", dyn_oof); np.save(OUT / "dyn_val.npy", dyn_val_arr); np.save(OUT / "dyn_test.npy", dyn_test_arr); np.save(OUT / "oof_prob.npy", oof)
    Xfull = np.concatenate([static["train"], dyn_oof], axis=1)
    # For the final held-out prediction, fit the base learner on all labeled training records and use full-train dynamic features.
    dyn_full_train = np.concatenate([aggregate_mm(mm_train, train_ids, labels_full, train_sp, train_sp), aggregate_ppi(train_ids, labels_full, edges)], axis=1)
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=int(np.median(best_iters)), learning_rate=.025, num_leaves=31, min_child_samples=30, colsample_bytree=.65, reg_alpha=.1, reg_lambda=5, class_weight="balanced", random_state=42, n_jobs=8, verbosity=-1)
    clf.fit(np.concatenate([static["train"], dyn_full_train], axis=1), y_train)
    pv = clf.predict_proba(np.concatenate([static["val"], dyn_val_arr], axis=1))[:, 1]
    pt = clf.predict_proba(np.concatenate([static["test"], dyn_test_arr], axis=1))[:, 1]
    metrics = {"oof_auc": float(roc_auc_score(y_train, oof)), "oof_auprc": float(average_precision_score(y_train, oof)), "val_auc": float(roc_auc_score(y_val, pv)), "val_auprc": float(average_precision_score(y_val, pv)), "test_auc": float(roc_auc_score(y_test, pt)), "test_auprc": float(average_precision_score(y_test, pt))}
    np.save(OUT / "val_prob.npy", pv); np.save(OUT / "test_prob.npy", pt); (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
