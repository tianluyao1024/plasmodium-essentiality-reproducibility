from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
DATASET = "dedup95_strict_fixed"
FEATURE_ROOT = ROOT / "eukaryote_model_datasets" / "features" / DATASET
SPLIT_ROOT = ROOT / "eukaryote_model_datasets" / DATASET / "random_species_label_split_seed42"
MMSEQS = ROOT / "tools" / "mmseqs2" / "mmseqs" / "bin" / "mmseqs.exe"
OUT = FEATURE_ROOT / "mmseqs_neighbors"
OUT.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"Danio_rerio", "Candida_albicans__A21", "Bombyx_mori"}
THRESHOLDS = (0.30, 0.50, 0.70)


def search(split: str) -> Path:
    result = OUT / f"{split}_vs_train.tsv"
    if result.exists() and result.stat().st_size > 100:
        return result
    tmp = OUT / f"tmp_{split}"
    cmd = [str(MMSEQS), "easy-search", str(SPLIT_ROOT / f"{split}.protein.fasta"),
           str(SPLIT_ROOT / "train.protein.fasta"), str(result), str(tmp),
           "--format-output", "query,target,pident,alnlen,evalue,bits,qlen,tlen",
           "--max-seqs", "50", "-s", "7.5", "-e", "0.01", "--threads", "8"]
    subprocess.run(cmd, check=True)
    return result


def main():
    names_raw = json.loads((FEATURE_ROOT / "basic_seq" / "species_to_id.json").read_text(encoding="utf-8"))
    id_to_name = {int(v): k for k, v in names_raw.items()}
    train_meta = pd.read_csv(SPLIT_ROOT / "train_metadata.tsv", sep="\t")
    target = train_meta.set_index("record_id")[["label", "species"]]
    target_y = target["label"].map({"essential": 1, "nonessential": 0}).to_dict()
    target_sp = target["species"].to_dict()
    feature_names = []
    for t in THRESHOLDS:
        tag = int(t * 100)
        feature_names += [f"nn{tag}_log_n", f"nn{tag}_weighted_pos_ratio", f"nn{tag}_max_pos_score",
                          f"nn{tag}_max_neg_score", f"nn{tag}_same_species_ratio", f"nn{tag}_other_species_ratio"]
    feature_names += ["best_hit_identity", "best_hit_coverage", "best_hit_bits_per_length"]

    for split in ("train", "val", "test"):
        ids = np.load(FEATURE_ROOT / "basic_seq" / f"{split}_ids.npy", allow_pickle=True).astype(str)
        spids = np.load(FEATURE_ROOT / "basic_seq" / f"{split}_species.npy").astype(int)
        species = np.asarray([id_to_name[x] for x in spids], dtype=object)
        keep = ~np.isin(species, list(EXCLUDE))
        kept_ids, kept_sp = ids[keep], species[keep]
        hits = pd.read_csv(search(split), sep="\t", header=None,
                           names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"])
        hits = hits[hits["query"].isin(set(kept_ids)) & hits["target"].isin(target_y)].copy()
        hits = hits[hits["query"] != hits["target"]]
        hits["identity"] = hits["pident"].astype(float) / 100.0
        hits["coverage"] = hits["alnlen"] / hits[["qlen", "tlen"]].max(axis=1)
        hits["score"] = hits["identity"] * hits["coverage"]
        query_sp = dict(zip(kept_ids, kept_sp))
        hits["y"] = hits["target"].map(target_y).astype(float)
        hits["same"] = hits["target"].map(target_sp).eq(hits["query"].map(query_sp))
        matrix = pd.DataFrame(index=pd.Index(kept_ids, name="query"))
        for threshold in THRESHOLDS:
            z = hits[(hits["identity"] >= threshold) & (hits["coverage"] >= 0.30)].copy()
            z["w"] = np.maximum(z["score"], 1e-4) ** 3
            z["wy"] = z["w"] * z["y"]
            agg = z.groupby("query").agg(n=("target", "size"), w=("w", "sum"), wy=("wy", "sum"))
            max_by_label = z.groupby(["query", "y"])["score"].max().unstack(fill_value=0.0)
            same = z[z["same"]].groupby("query")[["w", "wy"]].sum()
            other = z[~z["same"]].groupby("query")[["w", "wy"]].sum()
            tag = int(threshold * 100)
            matrix[f"nn{tag}_log_n"] = np.log1p(agg["n"]).reindex(matrix.index).fillna(0.0)
            matrix[f"nn{tag}_weighted_pos_ratio"] = (agg["wy"] / agg["w"]).reindex(matrix.index).fillna(0.5)
            matrix[f"nn{tag}_max_pos_score"] = max_by_label.get(1.0, pd.Series(dtype=float)).reindex(matrix.index).fillna(0.0)
            matrix[f"nn{tag}_max_neg_score"] = max_by_label.get(0.0, pd.Series(dtype=float)).reindex(matrix.index).fillna(0.0)
            matrix[f"nn{tag}_same_species_ratio"] = ((same["wy"] + 0.5) / (same["w"] + 1.0)).reindex(matrix.index).fillna(0.5)
            matrix[f"nn{tag}_other_species_ratio"] = ((other["wy"] + 0.5) / (other["w"] + 1.0)).reindex(matrix.index).fillna(0.5)
        best = hits.sort_values(["query", "score", "bits"], ascending=[True, False, False]).drop_duplicates("query").set_index("query")
        matrix["best_hit_identity"] = best["identity"].reindex(matrix.index).fillna(0.0)
        matrix["best_hit_coverage"] = best["coverage"].reindex(matrix.index).fillna(0.0)
        matrix["best_hit_bits_per_length"] = (best["bits"] / best[["qlen", "tlen"]].max(axis=1)).reindex(matrix.index).fillna(0.0)
        arr = matrix[feature_names].to_numpy(dtype=np.float32)
        np.save(OUT / f"{split}_emb.npy", arr)
        np.save(OUT / f"{split}_ids.npy", kept_ids)
        print(split, arr.shape, "queries_with_hits", hits["query"].nunique(), flush=True)
    (OUT / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
