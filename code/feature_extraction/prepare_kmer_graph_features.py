from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
DATASET = os.environ.get("EUK_DATASET", "dedup95_strict")
SPLIT_NAME = os.environ.get("EUK_SPLIT_NAME", "random_species_label_split_seed42")
SPLIT_DIR = ROOT / "eukaryote_model_datasets" / DATASET / SPLIT_NAME
FEATURE_DIR = ROOT / "eukaryote_model_datasets" / "features" / DATASET
OUT_DIR = FEATURE_DIR / "kmer_graph"
SPLITS = ("train", "val", "test")


def parse_fasta(path: Path):
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def dna_digit_array(seq: str) -> np.ndarray:
    table = np.full(256, -1, dtype=np.int16)
    table[ord("A")] = 0
    table[ord("C")] = 1
    table[ord("G")] = 2
    table[ord("T")] = 3
    table[ord("a")] = 0
    table[ord("c")] = 1
    table[ord("g")] = 2
    table[ord("t")] = 3
    raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    vals = table[raw]
    return vals[vals >= 0].astype(np.int32, copy=False)


def kmer_locations(seq: str, k_size: int) -> np.ndarray:
    vals = dna_digit_array(seq)
    if vals.size < k_size:
        return np.asarray([], dtype=np.int32)
    if k_size == 2:
        return (vals[:-1] * 4 + vals[1:]).astype(np.int32, copy=False)
    if k_size == 3:
        return (vals[:-2] * 16 + vals[1:-1] * 4 + vals[2:]).astype(np.int32, copy=False)
    locs = np.zeros(vals.size - k_size + 1, dtype=np.int32)
    for pos in range(k_size):
        locs = locs * 4 + vals[pos : pos + locs.size]
    return locs


def transition_sparse_row(seq: str, k_size: int, distances: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    locs = kmer_locations(seq, k_size)
    n_kmer = 4**k_size
    col_blocks: list[np.ndarray] = []
    scale_blocks: list[np.ndarray] = []
    for block_idx in range(len(distances)):
        dis = distances[block_idx]
        valid_len = len(locs) - k_size - dis
        if valid_len <= 0:
            continue
        denom = max(1, len(seq) - 2 * k_size - dis + 1)
        block_offset = block_idx * n_kmer * n_kmer
        scale = 100.0 / denom
        left = locs[:valid_len]
        right = locs[k_size + dis : k_size + dis + valid_len]
        col_blocks.append((block_offset + left * n_kmer + right).astype(np.int32, copy=False))
        scale_blocks.append(np.full(valid_len, scale, dtype=np.float32))
    if not col_blocks:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)
    col_arr = np.concatenate(col_blocks).astype(np.int32, copy=False)
    scale_arr = np.concatenate(scale_blocks).astype(np.float32, copy=False)
    unique_cols, inverse = np.unique(col_arr, return_inverse=True)
    data = np.zeros(len(unique_cols), dtype=np.float32)
    np.add.at(data, inverse, scale_arr)
    return unique_cols.astype(np.int32, copy=False), data


def build_sparse_matrix(split: str, k_size: int, distances: tuple[int, ...], expected_ids: np.ndarray) -> sparse.csr_matrix:
    fasta = SPLIT_DIR / f"{split}.cds.fasta"
    seqs = {rid: seq for rid, seq in parse_fasta(fasta)}
    missing = [rid for rid in expected_ids if rid not in seqs]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} CDS records missing for kmer graph; first={missing[0]}")

    n_cols = len(distances) * (4**k_size) * (4**k_size)
    indptr = [0]
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for rid in expected_ids:
        try:
            cols, data = transition_sparse_row(seqs[str(rid)], k_size, distances)
        except Exception as exc:
            raise RuntimeError(
                f"{split}: failed kmer graph for record_id={rid} k={k_size} len={len(seqs[str(rid)])}"
            ) from exc
        indices.append(cols)
        values.append(data)
        indptr.append(indptr[-1] + len(cols))
    if indices:
        all_indices = np.concatenate(indices).astype(np.int32, copy=False)
        all_values = np.concatenate(values).astype(np.float32, copy=False)
    else:
        all_indices = np.asarray([], dtype=np.int32)
        all_values = np.asarray([], dtype=np.float32)
    return sparse.csr_matrix(
        (all_values, all_indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(expected_ids), n_cols),
        dtype=np.float32,
    )


def choose_components(requested: int, matrix: sparse.csr_matrix) -> int:
    return max(1, min(requested, matrix.shape[0] - 1, matrix.shape[1] - 1))


def fit_transform_group(
    matrices: dict[str, sparse.csr_matrix],
    group_name: str,
    n_components: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    train = matrices["train"]
    n_comp = choose_components(n_components, train)
    svd = TruncatedSVD(n_components=n_comp, algorithm="randomized", random_state=random_state)
    train_z = svd.fit_transform(train).astype(np.float32)
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_z).astype(np.float32)
    out = {"train": train_z}
    for split in ("val", "test"):
        out[split] = scaler.transform(svd.transform(matrices[split])).astype(np.float32)
    joblib.dump({"svd": svd, "scaler": scaler}, OUT_DIR / f"{group_name}_svd_scaler.joblib")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k2-components", type=int, default=128)
    parser.add_argument("--k3-components", type=int, default=384)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-raw-sparse", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = {
        split: np.load(FEATURE_DIR / "basic_seq" / f"{split}_ids.npy", allow_pickle=True).astype(str)
        for split in SPLITS
    }
    distances = (0, 1, 3)
    config = {
        "dataset": DATASET,
        "source": str(SPLIT_DIR),
        "distances": list(distances),
        "groups": {
            "kmer_k2": {"k": 2, "raw_dim": 768, "components": args.k2_components},
            "kmer_k3": {"k": 3, "raw_dim": 12288, "components": args.k3_components},
        },
    }
    (OUT_DIR / "kmer_graph_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    group_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for group_name, k_size, comps in (("kmer_k2", 2, args.k2_components), ("kmer_k3", 3, args.k3_components)):
        matrices = {}
        for split in SPLITS:
            mat = build_sparse_matrix(split, k_size, distances, ids[split])
            matrices[split] = mat
            print(f"{group_name} {split} sparse {mat.shape} nnz={mat.nnz}", flush=True)
            if args.save_raw_sparse:
                sparse.save_npz(OUT_DIR / f"{split}_{group_name}_raw.npz", mat)
        group_embeddings[group_name] = fit_transform_group(matrices, group_name, comps, args.random_state)
        del matrices
        gc.collect()

    for split in SPLITS:
        emb = np.concatenate([group_embeddings["kmer_k2"][split], group_embeddings["kmer_k3"][split]], axis=1)
        np.save(OUT_DIR / f"{split}_emb.npy", emb.astype(np.float32))
        np.save(OUT_DIR / f"{split}_ids.npy", ids[split])
        print(f"{split} kmer_graph emb {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
