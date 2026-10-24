"""Fold-local feature-block selection and PCA transforms for ProtoEss."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BLOCKS = (
    "esm2",
    "prott5",
    "basic_seq",
    "go_ipr_pfam",
    "kmer_graph",
    "homology30",
    "homology50",
    "string_network",
    "string_structure",
    "species_onehot",
    "mmseqs_dynamic",
    "ppi_dynamic",
)


def block_columns(feature_names: list[str]) -> dict[str, np.ndarray]:
    names = np.asarray(feature_names, dtype=object)
    return {
        block: np.flatnonzero(np.char.startswith(names.astype(str), f"{block}_"))
        for block in BLOCKS
    }


def fit_profile_groups(
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
    profile: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Fit each selected block transform on X_fit only and apply it to X_eval."""
    profile = profile or {}
    include = set(profile.get("include_blocks", BLOCKS))
    pca_components = {str(k): int(v) for k, v in profile.get("pca_components", {}).items()}
    if not include.issubset(BLOCKS):
        raise ValueError(f"Unknown feature blocks in profile: {sorted(include - set(BLOCKS))}")
    if X_fit.ndim != 2 or X_eval.ndim != 2 or X_fit.shape[1] != len(feature_names) or X_eval.shape[1] != len(feature_names):
        raise ValueError("Feature matrix dimensions do not match feature_names")

    columns = block_columns(feature_names)
    fit_groups: list[np.ndarray] = []
    eval_groups: list[np.ndarray] = []
    group_names: list[str] = []
    for block in BLOCKS:
        idx = columns[block]
        if block not in include or not len(idx):
            continue
        a = np.asarray(X_fit[:, idx], dtype=np.float32)
        b = np.asarray(X_eval[:, idx], dtype=np.float32)
        n_components = pca_components.get(block)
        if n_components is not None:
            if n_components < 1:
                raise ValueError(f"PCA components must be positive for {block}")
            n_components = min(n_components, a.shape[1], a.shape[0] - 1)
            if n_components < 1:
                raise ValueError(f"Not enough training rows for PCA in {block}")
            scaler = StandardScaler()
            a = scaler.fit_transform(a).astype(np.float32)
            b = scaler.transform(b).astype(np.float32)
            reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
            a = reducer.fit_transform(a).astype(np.float32)
            b = reducer.transform(b).astype(np.float32)
        fit_groups.append(a)
        eval_groups.append(b)
        group_names.append(block)
    if not fit_groups:
        raise ValueError("Feature profile selected no feature blocks")
    return fit_groups, eval_groups, group_names


def concatenate_groups(groups: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(groups, axis=1).astype(np.float32, copy=False)


def transformer_modalities(
    groups: list[np.ndarray],
    group_names: list[str],
) -> list[np.ndarray]:
    by_name = dict(zip(group_names, groups))
    ordered = [by_name[name] for name in (
        "esm2", "prott5", "basic_seq", "go_ipr_pfam", "kmer_graph",
        "homology30", "homology50", "string_network",
    ) if name in by_name]
    context = [by_name[name] for name in ("string_structure", "species_onehot", "ppi_dynamic") if name in by_name]
    if context:
        ordered.append(np.concatenate(context, axis=1).astype(np.float32, copy=False))
    if "mmseqs_dynamic" in by_name:
        ordered.append(by_name["mmseqs_dynamic"])
    return ordered


def fit_profile_matrix(
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
    profile: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    fit_groups, eval_groups, group_names = fit_profile_groups(
        X_fit, X_eval, feature_names, profile, seed
    )
    widths = [int(group.shape[1]) for group in fit_groups]
    return concatenate_groups(fit_groups), concatenate_groups(eval_groups), group_names, widths

