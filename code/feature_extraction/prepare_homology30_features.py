from __future__ import annotations

import json
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
DATASET = os.environ.get("EUK_DATASET", "dedup95_strict")
SPLIT_NAME = os.environ.get("EUK_SPLIT_NAME", "random_species_label_split_seed42")
FEATURE_ROOT = ROOT / "eukaryote_model_datasets" / "features" / DATASET
SPLIT_DIR = ROOT / "eukaryote_model_datasets" / DATASET / SPLIT_NAME
DEFAULT_CLUSTER_TSV = ROOT / "eukaryote_model_datasets" / "mmseqs30" / "cluster30_cluster.tsv"
DEFAULT_OUT = FEATURE_ROOT / "homology30"
SPLITS = ("train", "val", "test")
EXCLUDE = {"Danio_rerio", "Candida_albicans__A21", "Bombyx_mori"}


FEATURE_NAMES = [
    "cluster_log_size_all",
    "cluster_log_species_all",
    "cluster_train_log_n",
    "cluster_train_pos_ratio",
    "cluster_train_balance",
    "cluster_train_has_pos",
    "cluster_train_has_neg",
    "cluster_train_same_species_log_n",
    "cluster_train_same_species_pos_ratio",
    "cluster_train_other_species_log_n",
    "cluster_train_other_species_pos_ratio",
    "cluster_train_same_clade_log_n",
    "cluster_train_same_clade_pos_ratio",
    "cluster_train_other_clade_log_n",
    "cluster_train_other_clade_pos_ratio",
]


def species_names() -> dict[int, str]:
    raw = json.loads((FEATURE_ROOT / "basic_seq" / "species_to_id.json").read_text(encoding="utf-8"))
    return {int(v): k for k, v in raw.items()}


def clade_of(species: str) -> str:
    if species in {"Arabidopsis_thaliana", "Oryza_sativa_japonica"}:
        return "plant"
    if species.startswith("Plasmodium_") or species == "Toxoplasma_gondii":
        return "apicomplexa"
    if "Candida" in species or "Saccharomyces" in species or "Schizosaccharomyces" in species or "Pichia" in species:
        return "fungi"
    return "animal"


def load_split(split: str, id_to_species: dict[int, str]) -> pd.DataFrame:
    basic = FEATURE_ROOT / "basic_seq"
    ids = np.load(basic / f"{split}_ids.npy", allow_pickle=True).astype(str)
    y = np.load(basic / f"{split}_y.npy").astype(int)
    spid = np.load(basic / f"{split}_species.npy").astype(int)
    species = np.asarray([id_to_species[int(s)] for s in spid], dtype=object)
    return pd.DataFrame(
        {
            "record_id": ids,
            "split": split,
            "y": y,
            "species": species,
            "clade": [clade_of(s) for s in species],
        }
    )


def safe_ratio(pos: float, total: float, prior: float) -> float:
    return float((pos + prior) / (total + 2.0 * prior)) if total > 0 else 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-tsv", type=Path, default=DEFAULT_CLUSTER_TSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    id_to_species = species_names()
    split_df = pd.concat([load_split(s, id_to_species) for s in SPLITS], ignore_index=True)
    split_df = split_df[~split_df["species"].isin(EXCLUDE)].copy()

    clusters = pd.read_csv(args.cluster_tsv, sep="\t", header=None, names=["cluster_id", "record_id"])
    clusters = clusters.drop_duplicates("record_id")
    df = split_df.merge(clusters, on="record_id", how="left")
    if df["cluster_id"].isna().any():
        raise RuntimeError(f"missing cluster ids: {df['cluster_id'].isna().sum()}")

    all_stats = (
        df.groupby("cluster_id")
        .agg(cluster_size_all=("record_id", "size"), cluster_species_all=("species", "nunique"))
        .reset_index()
    )
    train = df[df["split"] == "train"].copy()
    global_prior = float(train["y"].mean())

    cluster_total = {
        cid: (float(g.shape[0]), float(g["y"].sum()))
        for cid, g in train.groupby("cluster_id", sort=False)
    }
    csp_total = {
        key: (float(g.shape[0]), float(g["y"].sum()))
        for key, g in train.groupby(["cluster_id", "species"], sort=False)
    }
    ccl_total = {
        key: (float(g.shape[0]), float(g["y"].sum()))
        for key, g in train.groupby(["cluster_id", "clade"], sort=False)
    }

    def features_for_row(row: pd.Series) -> list[float]:
        cid = row["cluster_id"]
        species = row["species"]
        clade = row["clade"]
        own = 1 if row["split"] == "train" else 0
        own_y = int(row["y"]) if own else 0

        all_row = all_stats_map[cid]
        cn, cp = cluster_total.get(cid, (0.0, 0.0))
        cn_adj = cn - own
        cp_adj = cp - own_y
        cn_adj = max(cn_adj, 0.0)
        cp_adj = max(cp_adj, 0.0)

        csp_key = (cid, species)
        ssn, ssp = csp_total.get(csp_key, (0.0, 0.0))
        ssn -= own
        ssp -= own_y
        ssn, ssp = max(ssn, 0.0), max(ssp, 0.0)
        osn = max(cn_adj - ssn, 0.0)
        osp = max(cp_adj - ssp, 0.0)

        ccl_key = (cid, clade)
        scn, scp = ccl_total.get(ccl_key, (0.0, 0.0))
        scn -= own
        scp -= own_y
        scn, scp = max(scn, 0.0), max(scp, 0.0)
        ocn = max(cn_adj - scn, 0.0)
        ocp = max(cp_adj - scp, 0.0)

        ratio = safe_ratio(cp_adj, cn_adj, global_prior)
        return [
            np.log1p(float(all_row["cluster_size_all"])),
            np.log1p(float(all_row["cluster_species_all"])),
            np.log1p(cn_adj),
            ratio,
            2.0 * ratio - 1.0,
            float(cp_adj > 0),
            float(cn_adj - cp_adj > 0),
            np.log1p(ssn),
            safe_ratio(ssp, ssn, global_prior),
            np.log1p(osn),
            safe_ratio(osp, osn, global_prior),
            np.log1p(scn),
            safe_ratio(scp, scn, global_prior),
            np.log1p(ocn),
            safe_ratio(ocp, ocn, global_prior),
        ]

    all_stats_map = all_stats.set_index("cluster_id").to_dict(orient="index")
    for split in SPLITS:
        split_rows = df[df["split"] == split].copy()
        feats = np.asarray([features_for_row(row) for _, row in split_rows.iterrows()], dtype=np.float32)
        ids_expected = np.load(FEATURE_ROOT / "basic_seq" / f"{split}_ids.npy", allow_pickle=True).astype(str)
        species_expected = np.asarray([id_to_species[int(s)] for s in np.load(FEATURE_ROOT / "basic_seq" / f"{split}_species.npy")], dtype=object)
        keep = ~np.isin(species_expected, list(EXCLUDE))
        ids_expected = ids_expected[keep]
        if len(ids_expected) != len(split_rows) or np.any(ids_expected != split_rows["record_id"].to_numpy(str)):
            raise RuntimeError(f"{split}: output order mismatch")
        np.save(args.out / f"{split}_emb.npy", feats)
        np.save(args.out / f"{split}_ids.npy", ids_expected)
        print(split, feats.shape, flush=True)

    (args.out / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    summary = df.groupby(["split", "clade", "species"])["record_id"].size().reset_index(name="n")
    summary.to_csv(args.out / "summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
