from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit SVD on training rows only and transform the remaining rows.")
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transformer", type=Path, help="Previously fitted bundle; aligns this species' GO terms to its saved vocabulary before projection")
    args = ap.parse_args()
    idx = pd.read_csv(args.raw_dir / "record_index.tsv", sep="\t", dtype=str)
    basic = np.load(args.raw_dir / "basic_seq_raw.npy")
    ann = np.load(args.raw_dir / "go_terms_raw.npy")
    vocab = json.loads((args.raw_dir / "go_terms_vocab.json").read_text(encoding="utf-8"))
    if len(idx) != len(basic) or len(idx) != len(ann):
        raise ValueError("record_index and raw feature rows do not align")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "basic_seq.npy", basic)
    if args.transformer:
        bundle = joblib.load(args.transformer)
        if not {"mean", "std", "svd", "vocab"}.issubset(bundle):
            raise ValueError("Transformer lacks the raw-feature vocabulary/schema required for safe reuse")
        fit_vocab = bundle["vocab"]
        fit_ix = {term: j for j, term in enumerate(vocab)}
        aligned = np.zeros((len(idx), len(fit_vocab)), dtype=np.float32)
        for j, term in enumerate(fit_vocab):
            source_j = fit_ix.get(term)
            if source_j is not None:
                aligned[:, j] = ann[:, source_j]
        if aligned.shape[1] != len(bundle["mean"]) or len(bundle["mean"]) != bundle["svd"].n_features_in_:
            raise ValueError("Input feature width does not match the saved scaler/SVD schema")
        scaled = (aligned - bundle["mean"]) / bundle["std"]
        reduced = bundle["svd"].transform(scaled).astype(np.float32)
        n_comp = reduced.shape[1]
        fit_n = 0
        fit_split = "external fitted transformer"
        saved_bundle = bundle
    else:
        train = idx.split.eq(args.train_split).to_numpy()
        if train.sum() < 2:
            raise ValueError(f"Need >=2 rows in split {args.train_split!r} to fit SVD; for a held-out species, pass the training-species bundle with --transformer")
        n_comp = min(args.components, int(train.sum()) - 1, ann.shape[1] - 1)
        if n_comp < 1:
            raise ValueError("GO vocabulary too small for SVD")
        mean = ann[train].mean(axis=0).astype(np.float32)
        std = ann[train].std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        scaled = (ann - mean) / std
        svd = TruncatedSVD(n_components=n_comp, n_iter=4, random_state=args.seed)
        reduced = np.zeros((len(idx), n_comp), dtype=np.float32)
        reduced[train] = svd.fit_transform(scaled[train]).astype(np.float32)
        other = ~train
        if other.any(): reduced[other] = svd.transform(scaled[other]).astype(np.float32)
        fit_n = int(train.sum())
        fit_split = args.train_split
        saved_bundle = {"mean": mean, "std": std, "svd": svd, "vocab": vocab}
    np.save(args.out / "go_ipr_pfam_reduced.npy", reduced)
    joblib.dump(saved_bundle, args.out / "go_svd_scaler.joblib")
    (args.out / "reduction_manifest.json").write_text(json.dumps({"fit_split": fit_split, "fit_n": fit_n, "requested_components": args.components, "actual_go_components": n_comp, "seed": args.seed, "transformer_input": str(args.transformer) if args.transformer else None, "warning": "This local GAF-only reduction does not recreate the original GO/IPR/Pfam block."}, indent=2), encoding="utf-8")
    print(f"Reduced GO block to {n_comp} dimensions; basic sequence block remains 31 dimensions")


if __name__ == "__main__":
    main()
