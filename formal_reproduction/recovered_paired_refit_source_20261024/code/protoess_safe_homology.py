from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
DATA = ROOT / "eukaryote_model_datasets"
CLUSTER_FILES = {
    "homology30": DATA / "mmseqs30" / "cluster30_cluster.tsv",
    "homology50": DATA / "mmseqs50" / "cluster50_cluster.tsv",
}
FEATURE_NAMES = (
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
)


def clade_of(species: str) -> str:
    if species in {"Arabidopsis_thaliana", "Oryza_sativa_japonica"}:
        return "plant"
    if species.startswith("Plasmodium_") or species == "Toxoplasma_gondii":
        return "apicomplexa"
    if any(token in species for token in ("Candida", "Saccharomyces", "Schizosaccharomyces", "Pichia")):
        return "fungi"
    return "animal"


@lru_cache(maxsize=2)
def cluster_map(group: str) -> dict[str, str]:
    path = CLUSTER_FILES[group]
    table = pd.read_csv(path, sep="\t", header=None, names=["cluster_id", "record_id"], dtype=str)
    table = table.drop_duplicates("record_id", keep="first")
    return dict(zip(table.record_id, table.cluster_id))


@lru_cache(maxsize=2)
def all_cluster_stats(group: str) -> dict[str, tuple[int, int]]:
    table = pd.read_csv(
        CLUSTER_FILES[group], sep="\t", header=None,
        names=["cluster_id", "record_id"], dtype=str,
    ).drop_duplicates("record_id", keep="first")
    table["species"] = table.record_id.str.split("|", n=1).str[0]
    stats = table.groupby("cluster_id", sort=False).agg(size=("species", "size"), species_n=("species", "nunique"))
    return {
        str(cluster_id): (int(row["size"]), int(row["species_n"]))
        for cluster_id, row in stats.iterrows()
    }


def _safe_ratio(pos: float, total: float, prior: float) -> float:
    return float((pos + prior) / (total + 2.0 * prior)) if total > 0 else 0.5


def _group_stats(training: pd.DataFrame, key_cols: list[str]) -> dict[tuple[str, ...], tuple[float, float]]:
    if training.empty:
        return {}
    grouped = training.groupby(key_cols, sort=False).agg(n=("y", "size"), pos=("y", "sum"))
    if len(key_cols) == 1:
        return {(str(key),): (float(row.n), float(row.pos)) for key, row in grouped.iterrows()}
    return {(tuple(map(str, key)),): (float(row.n), float(row.pos)) for key, row in grouped.iterrows()}


def raw_homology_features(query: pd.DataFrame, training: pd.DataFrame,
                          query_cluster_overrides: dict[str, dict[str, str]] | None = None) -> np.ndarray:
    """Return homology30/50 features using only labels present in training."""
    if not {"record_id", "species"}.issubset(query.columns):
        raise ValueError("query must contain record_id and species")
    if not {"record_id", "species", "label"}.issubset(training.columns):
        raise ValueError("training must contain record_id, species, and label")
    if training.record_id.astype(str).duplicated().any():
        raise ValueError("duplicate IDs in allowed training labels")

    query_ids = query.record_id.astype(str).to_numpy()
    query_species = query.species.astype(str).to_numpy()
    train_ids = training.record_id.astype(str).to_numpy()
    train_species = training.species.astype(str).to_numpy()
    train_y = training.label.eq("essential").astype(np.int8).to_numpy()
    y_by_id = dict(zip(train_ids, train_y.astype(float)))
    prior_all = float(train_y.mean()) if len(train_y) else 0.5
    train_clade = np.asarray([clade_of(s) for s in train_species], dtype=object)
    query_clade = np.asarray([clade_of(s) for s in query_species], dtype=object)
    blocks: list[np.ndarray] = []

    query_cluster_overrides = query_cluster_overrides or {}
    for group in ("homology30", "homology50"):
        cmap = cluster_map(group)
        all_stats = all_cluster_stats(group)
        train_cluster = np.asarray([cmap.get(rid, rid) for rid in train_ids], dtype=object)
        overrides = query_cluster_overrides.get(group, {})
        query_cluster = np.asarray([overrides.get(rid, cmap.get(rid, rid)) for rid in query_ids], dtype=object)
        labels = pd.DataFrame({
            "cluster_id": train_cluster,
            "species": train_species,
            "clade": train_clade,
            "y": train_y,
        })
        cluster_stats = _group_stats(labels, ["cluster_id"])
        species_stats = _group_stats(labels, ["cluster_id", "species"])
        clade_stats = _group_stats(labels, ["cluster_id", "clade"])
        features = np.zeros((len(query_ids), len(FEATURE_NAMES)), dtype=np.float32)

        for i, (rid, cid, species, clade) in enumerate(zip(query_ids, query_cluster, query_species, query_clade)):
            cn, cp = cluster_stats.get((str(cid),), (0.0, 0.0))
            ssn, ssp = species_stats.get(((str(cid), str(species)),), (0.0, 0.0))
            scn, scp = clade_stats.get(((str(cid), str(clade)),), (0.0, 0.0))
            own_y = y_by_id.get(rid)
            own = 1.0 if own_y is not None else 0.0
            cn = max(cn - own, 0.0)
            cp = max(cp - (float(own_y) if own_y is not None else 0.0), 0.0)
            ssn = max(ssn - own, 0.0)
            ssp = max(ssp - (float(own_y) if own_y is not None else 0.0), 0.0)
            scn = max(scn - own, 0.0)
            scp = max(scp - (float(own_y) if own_y is not None else 0.0), 0.0)
            osn, osp = max(cn - ssn, 0.0), max(cp - ssp, 0.0)
            ocn, ocp = max(cn - scn, 0.0), max(cp - scp, 0.0)
            prior = (float(train_y.sum() - int(own_y)) / (len(train_y) - 1)) if own_y is not None and len(train_y) > 1 else prior_all
            ratio = _safe_ratio(cp, cn, prior)
            all_n, all_species_n = all_stats.get(str(cid), (1, 1))
            features[i] = [
                np.log1p(all_n), np.log1p(all_species_n), np.log1p(cn), ratio,
                2.0 * ratio - 1.0, float(cp > 0), float(cn - cp > 0),
                np.log1p(ssn), _safe_ratio(ssp, ssn, prior),
                np.log1p(osn), _safe_ratio(osp, osn, prior),
                np.log1p(scn), _safe_ratio(scp, scn, prior),
                np.log1p(ocn), _safe_ratio(ocp, ocn, prior),
            ]
        blocks.append(features)
    return np.concatenate(blocks, axis=1)


def scale_from_train(raw_train: np.ndarray, raw_query: np.ndarray) -> np.ndarray:
    mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw_train.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
    return ((raw_query - mean) / std).astype(np.float32)


def fold_safe_homology(frames: dict[str, pd.DataFrame], inner_ids: set[str], query_ids: np.ndarray) -> np.ndarray:
    train = frames["train"]
    inner_mask = train.record_id.astype(str).isin(inner_ids).to_numpy()
    inner = train.loc[inner_mask, ["record_id", "species", "label"]].copy()
    if inner.empty:
        raise ValueError("OOF inner training set is empty")
    query_index = pd.concat([frame[["record_id", "species"]] for frame in frames.values()], ignore_index=True)
    query_index = query_index.drop_duplicates("record_id").set_index("record_id")
    ids = np.asarray(query_ids, dtype=str)
    query = query_index.loc[ids].reset_index()
    raw_inner = raw_homology_features(inner[["record_id", "species"]], inner)
    raw_query = raw_homology_features(query, inner)
    return scale_from_train(raw_inner, raw_query)


def dynamic_safe_homology(query_ids: np.ndarray, target_ids: np.ndarray, target_y: np.ndarray,
                          id_to_species: dict[str, str]) -> np.ndarray:
    target_ids = np.asarray(target_ids, dtype=str)
    query_ids = np.asarray(query_ids, dtype=str)
    training = pd.DataFrame({
        "record_id": target_ids,
        "species": [str(id_to_species[x]) for x in target_ids],
        "label": np.where(np.asarray(target_y, dtype=int) == 1, "essential", "nonessential"),
    })
    query = pd.DataFrame({"record_id": query_ids, "species": [str(id_to_species[x]) for x in query_ids]})
    raw_train = raw_homology_features(training[["record_id", "species"]], training)
    raw_query = raw_homology_features(query, training)
    return scale_from_train(raw_train, raw_query)

