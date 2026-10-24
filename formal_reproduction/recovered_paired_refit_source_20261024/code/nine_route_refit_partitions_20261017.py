"""Paired, target-restricted partitions for nine-route resampling refits.

The four scopes may use permitted training labels from other species. All
target-species labels outside the current inner fit, including records omitted
by subsampling, must be absent. This module only creates/audits partitions;
it does not train models or produce refit-stability conclusions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCOPES = ("strict_within_species", "plasmodium_only_label_assisted",
          "cross_species_label_assisted", "cross_species_with_pk_oberstaller_labels")
COLS = ["record_id", "species", "label"]


def _validate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    result = frame.loc[:, COLS].copy()
    if result.isna().any().any():
        raise ValueError(f"{name}: missing IDs, species or labels")
    for col in COLS:
        result[col] = result[col].astype(str)
    if result.record_id.duplicated().any() or (result.record_id.str.len() == 0).any():
        raise ValueError(f"{name}: duplicate or empty record IDs")
    if not result.label.isin(["essential", "nonessential"]).all():
        raise ValueError(f"{name}: unexpected label values")
    return result


def id_digest(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids, dtype=str)).encode("utf-8")).hexdigest()


def stratified_subset(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return sorted unique indices, shared by full and masked comparisons."""
    y = np.asarray(labels)
    if y.ndim != 1 or not 0 < fraction <= 1 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Expected binary labels and 0 < fraction <= 1")
    rng = np.random.default_rng(seed)
    groups = []
    for value in (0, 1):
        members = np.flatnonzero(y == value)
        count = int(round(len(members) * fraction))
        if count < 5:
            raise ValueError("Each sampled class needs at least five records for five-fold OOF")
        groups.append(rng.choice(members, size=count, replace=False))
    return np.sort(np.concatenate(groups))


@dataclass(frozen=True)
class Partition:
    selected_indices: np.ndarray
    fold_assignment: np.ndarray
    seed: int
    fraction: float


def build_partition(labels: np.ndarray, fraction: float, seed: int, n_splits: int = 5) -> Partition:
    if n_splits != 5:
        raise ValueError("This protocol is fixed to five-fold OOF")
    selected = stratified_subset(labels, fraction, seed)
    y = np.asarray(labels)[selected]
    folds = np.full(len(selected), -1, dtype=np.int8)
    for fold, (_, hold) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(selected, y)):
        if (folds[hold] != -1).any():
            raise RuntimeError("A selected record appears in more than one OOF fold")
        folds[hold] = fold
    if (folds < 0).any():
        raise RuntimeError("A selected record is missing an OOF fold")
    return Partition(selected, folds, seed, fraction)


def make_scope_pools(target: str, fit: pd.DataFrame, all_training: pd.DataFrame,
                     ober_training: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Restrict target labels to fit IDs, retaining permitted external labels.

    For the fourth scope, Pk labels in ``ober_training`` are alternate screen
    calls only for matched Pk training IDs, not additional validation/test IDs.
    """
    fit = _validate(fit, "target fit")
    all_training = _validate(all_training, "all training")
    ober_training = _validate(ober_training, "alternate training")
    if fit.empty or not fit.species.eq(target).all():
        raise ValueError("fit must contain only nonempty target-species training records")
    reference = all_training.set_index("record_id")
    if not set(fit.record_id).issubset(set(reference.index)):
        raise ValueError("Target fit includes records outside permitted training metadata")
    selected = reference.loc[fit.record_id]
    if not np.array_equal(selected.species.to_numpy(), fit.species.to_numpy()) or not np.array_equal(selected.label.to_numpy(), fit.label.to_numpy()):
        raise ValueError("Target fit does not match frozen training species and labels")
    if not set(ober_training.record_id).issubset(set(all_training.record_id)):
        raise ValueError("Alternate pool includes IDs outside frozen training metadata")
    original_species = reference.loc[ober_training.record_id, "species"].to_numpy()
    if not np.array_equal(original_species, ober_training.species.to_numpy()):
        raise ValueError("Alternate pool changes record species")
    other = ober_training.species.ne("Plasmodium_knowlesi")
    if not np.array_equal(reference.loc[ober_training.loc[other, "record_id"], "label"].to_numpy(),
                          ober_training.loc[other, "label"].to_numpy()):
        raise ValueError("Alternate pool may only change Pk labels")

    fit_ids = set(fit.record_id)
    def restrict(pool: pd.DataFrame) -> pd.DataFrame:
        return pool.loc[pool.species.ne(target) | pool.record_id.isin(fit_ids)].copy().reset_index(drop=True)

    malaria = all_training.loc[all_training.species.str.startswith("Plasmodium_")]
    pools = {SCOPES[0]: fit.reset_index(drop=True), SCOPES[1]: restrict(malaria),
             SCOPES[2]: restrict(all_training), SCOPES[3]: restrict(ober_training)}
    for name, pool in pools.items():
        if not set(pool.loc[pool.species.eq(target), "record_id"]).issubset(fit_ids):
            raise RuntimeError(f"{name}: non-fit target labels in pool")
    return pools

