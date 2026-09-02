"""Repeated training-refit stability for retained directional orthogroups.

This is deliberately *not* labelled a bootstrap analysis.  Each repeat fits
the complete five-branch OOF stack on a fresh, class-stratified 80% subset of
the frozen training partition, reconstructing the label-derived MMseqs2 and
PPI channels from that subset.  The formal validation set is used only to set
the operating threshold; formal test labels are never read by this runner.

Both the full and functional-annotation-excluded stacks are refit.  A retained
directional pattern is counted only when the same pattern is recovered by both
stacks in a repeat.  Per-repeat checkpoints make the long GPU job restartable.
"""
from __future__ import annotations

import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from run_ncomms_annotation_excluded import (
    ALIASES, ANNOTATION_SLICE, APP_META, MM_HITS, SPECIES, app_arrays,
    app_ppi_dynamic, app_static, all_vs_all, catboost, choose_threshold,
    dynamic_x, edge_bundle, lgbm, read_all,
)
from train_scope_transformer_oof import split_groups, train_predict


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get(
    "NCOMMS_DIRECTIONAL_REFIT_OUT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "directional_resampling_refit_stability_216"),
))
STATE = OUT / "state.json"
REPEATS = int(os.environ.get("DIRECTIONAL_REFIT_REPEATS", "100"))
FRACTION = float(os.environ.get("DIRECTIONAL_REFIT_FRACTION", "0.80"))
SEED = int(os.environ.get("DIRECTIONAL_REFIT_SEED", "20260821"))
REPEAT_START = int(os.environ.get("DIRECTIONAL_REFIT_START", "0"))
REPEAT_STRIDE = int(os.environ.get("DIRECTIONAL_REFIT_STRIDE", "1"))
SUMMARISE_AFTER = os.environ.get("DIRECTIONAL_REFIT_SUMMARISE_AFTER", "1") != "0"

BASE = Path(os.environ.get(
    "NCOMMS_DIRECTIONAL_INPUT_ROOT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_annotation_excluded" / "directional_orthogroup_audits"),
))
GENE_AUDIT = BASE / "gene_level_full_vs_annotation_excluded.tsv"
PATTERNS = BASE / "single_copy_1to1to1_directional_patterns.tsv"


def write_state(**items: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    state.update(items)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Independent shard workers may update progress simultaneously.  A
    # process-specific temporary name avoids cross-worker truncation; the
    # final rename is retried because Windows can briefly hold state.json.
    temp = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            temp.replace(STATE)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    temp.replace(STATE)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def annotation_mask(x: np.ndarray, excluded: bool) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if excluded:
        y[:, ANNOTATION_SLICE] = 0.0
    return y


def sample_train_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected: list[np.ndarray] = []
    for klass in (0, 1):
        members = np.flatnonzero(y == klass)
        take = max(3, int(round(len(members) * FRACTION)))
        selected.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def app_matrix(species: str, app_parts: dict[str, np.ndarray], train_ids: np.ndarray,
               y_train: np.ndarray, mm_hits: pd.DataFrame, ppi_dynamic: np.ndarray,
               excluded: bool) -> np.ndarray:
    frames = {
        "train": pd.DataFrame({"record_id": train_ids, "label": np.where(y_train == 1, "essential", "non-essential"),
                               "species": species}),
        "val": pd.DataFrame(), "test": pd.DataFrame(),
    }
    # app_static only uses the train frame for this within-species scope.
    static = app_static("within_species", species, frames, app_parts)
    chosen = mm_hits[mm_hits.target.isin(set(train_ids))].copy()
    chosen["identity"] = chosen.pident.astype(float) / 100.0
    chosen["coverage"] = chosen.alnlen.astype(float) / chosen[["qlen", "tlen"]].astype(float).max(axis=1)
    chosen["score"] = chosen.identity * chosen.coverage
    from build_strict_oof_dynamic_stack import aggregate_mm
    labels = dict(zip(train_ids.astype(str), y_train.astype(float)))
    target_species = dict.fromkeys(train_ids.astype(str), species)
    reverse_alias = {value: key for key, value in ALIASES.items()}
    query_species = dict(zip(APP_META.record_id.astype(str), APP_META.species.astype(str).map(lambda x: reverse_alias.get(x, x))))
    app_ids = APP_META.record_id.astype(str).to_numpy()
    mm = aggregate_mm(chosen[chosen["query"].isin(set(app_ids))], app_ids, labels, target_species, query_species)
    result = np.concatenate([static, mm, ppi_dynamic], axis=1).astype(np.float32)
    return annotation_mask(result, excluded)


def oof_base(raw: np.ndarray, ids: np.ndarray, y: np.ndarray, meta: pd.DataFrame, hits: pd.DataFrame,
             edges: object, excluded: bool, seed: int) -> np.ndarray:
    id_to_species = dict(zip(ids.astype(str), meta.species.astype(str)))
    result = np.zeros((len(y), 5), dtype=np.float32)
    folds = StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    for fold, (inner, held) in enumerate(folds):
        x_inner = annotation_mask(dynamic_x(raw[inner], ids[inner], ids[inner], y[inner], id_to_species, hits, edges), excluded)
        x_held = annotation_mask(dynamic_x(raw[held], ids[held], ids[inner], y[inner], id_to_species, hits, edges), excluded)
        a = lgbm(x_inner, y[inner], seed + 101 + fold)
        b = catboost(x_inner, y[inner], seed + 201 + fold)
        c = lgbm(x_inner, y[inner], seed + 301 + fold)
        transformer, _, _ = train_predict(
            split_groups(x_inner, 1), y[inner], split_groups(x_held, 1), y[held], split_groups(x_held, 1), seed + 401 + fold,
        )
        result[held] = np.column_stack([
            a.predict_proba(x_held)[:, 1], b.predict_proba(x_held)[:, 1], c.predict_proba(x_held)[:, 1],
            np.repeat(float(y[inner].mean()), len(held)), transformer,
        ])
    return result


def score_species(species: str, repeat: int, excluded: bool, candidate_ids: set[str],
                  app_parts: dict[str, np.ndarray], mm_hits: pd.DataFrame, ppi_dynamic: np.ndarray) -> pd.DataFrame:
    meta, raw_all = read_all(species)
    y_all = meta.label.eq("essential").astype(np.int8).to_numpy()
    ids_all = meta.record_id.astype(str).to_numpy()
    train_global = np.flatnonzero(meta.formal_split.eq("train"))
    val_global = np.flatnonzero(meta.formal_split.eq("val"))
    rng = np.random.default_rng(SEED + repeat * 1009 + (17 if excluded else 0) + SPECIES.index(species))
    local = sample_train_indices(y_all[train_global], rng)
    train = train_global[local]
    y_train, ids_train, raw_train = y_all[train], ids_all[train], raw_all[train]
    similarity_cache = OUT / "cached_similarity" / species
    similarity_cache.mkdir(parents=True, exist_ok=True)
    hits = all_vs_all(species, similarity_cache)
    edges = edge_bundle(species)
    id_to_species = dict(zip(ids_all, meta.species.astype(str)))
    oof = oof_base(raw_train, ids_train, y_train, meta.iloc[train].reset_index(drop=True), hits, edges, excluded, SEED + repeat * 37)
    x_train = annotation_mask(dynamic_x(raw_train, ids_train, ids_train, y_train, id_to_species, hits, edges), excluded)
    x_val = annotation_mask(dynamic_x(raw_all[val_global], ids_all[val_global], ids_train, y_train, id_to_species, hits, edges), excluded)
    x_app = app_matrix(species, app_parts, ids_train, y_train, mm_hits, ppi_dynamic, excluded)
    if x_train.shape[1] != x_app.shape[1]:
        raise RuntimeError(f"Feature mismatch for {species}: {x_train.shape} vs {x_app.shape}")
    a = lgbm(x_train, y_train, SEED + repeat * 53 + 1)
    b = catboost(x_train, y_train, SEED + repeat * 53 + 2)
    c = lgbm(x_train, y_train, SEED + repeat * 53 + 3)
    val_t, app_t, _ = train_predict(
        split_groups(x_train, 1), y_train, split_groups(x_val, 1), y_all[val_global], split_groups(x_app, 1), SEED + repeat * 53 + 4,
    )
    def branches(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.column_stack([a.predict_proba(x)[:, 1], b.predict_proba(x)[:, 1], c.predict_proba(x)[:, 1], np.repeat(float(y_train.mean()), len(x)), t])
    meta_model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED + repeat)
    meta_model.fit(oof, y_train)
    threshold = choose_threshold(y_all[val_global], meta_model.predict_proba(branches(x_val, val_t))[:, 1])
    app_score = meta_model.predict_proba(branches(x_app, app_t))[:, 1]
    app = APP_META.loc[:, ["record_id", "species", "gene_id"]].copy()
    app["score"] = app_score
    app["threshold"] = threshold
    app["call"] = app_score >= threshold
    app = app[app.record_id.isin(candidate_ids)].copy()
    app["repeat"] = repeat
    app["model"] = "annotation_excluded" if excluded else "full"
    app["training_records"] = len(train)
    return app


def original_patterns() -> tuple[dict[str, tuple[str, str, str]], dict[str, set[str]]]:
    patterns = pd.read_csv(PATTERNS, sep="\t", dtype=str).fillna("")
    retained = patterns[
        patterns.pattern_retained.astype(str).str.lower().eq("true")
        & patterns.full_divergent.astype(str).str.lower().eq("true")
        & patterns.excluded_divergent.astype(str).str.lower().eq("true")
        & patterns.strong_full.astype(str).str.lower().eq("true")
        & patterns.strong_excluded.astype(str).str.lower().eq("true")
    ].copy()
    mapping = {
        row.homology_group_id: tuple("N" if value == "NE" else value for value in row.full_pattern.split("/"))
        for _, row in retained.iterrows()
    }
    genes = pd.read_csv(GENE_AUDIT, sep="\t", dtype=str)
    ids: dict[str, set[str]] = {}
    for group, sub in genes[genes.homology_group_id.isin(mapping)].groupby("species"):
        ids[group] = set(sub.record_id.astype(str))
    return mapping, ids


def summarise(patterns: dict[str, tuple[str, str, str]]) -> None:
    files = sorted((OUT / "repeats").glob("repeat_*_*.tsv"))
    rows = pd.concat([pd.read_csv(path, sep="\t", dtype={"record_id": str}) for path in files], ignore_index=True)
    gene = pd.read_csv(GENE_AUDIT, sep="\t", dtype=str)
    gene = gene[gene.homology_group_id.isin(patterns)].copy()
    merged = gene[["record_id", "species", "homology_group_id"]].merge(rows, on=["record_id", "species"], how="left")
    group_rows = []
    order = [ALIASES[s] for s in SPECIES]
    for (repeat, group), sub in merged.groupby(["repeat", "homology_group_id"], sort=True):
        if sub.model.nunique() != 2 or sub.species.nunique() != 3:
            continue
        calls = []
        agreement = True
        for sp in order:
            z = sub[sub.species.eq(sp)]
            if len(z) != 2 or z.call.nunique() != 1:
                agreement = False
                break
            calls.append("E" if bool(z.call.iloc[0]) else "N")
        group_rows.append({"repeat": repeat, "homology_group_id": group, "pattern": "/".join(calls) if agreement else "discordant", "both_model_agree": agreement,
                           "matches_reference": agreement and tuple(calls) == patterns[group]})
    per_repeat = pd.DataFrame(group_rows)
    per_repeat.to_csv(OUT / "repeat_level_directional_patterns.tsv", sep="\t", index=False)
    stability = per_repeat.groupby("homology_group_id", as_index=False).agg(
        refit_repeats=("repeat", "nunique"), both_model_agreement_frequency=("both_model_agree", "mean"),
        reference_pattern_recovery_frequency=("matches_reference", "mean"),
    )
    stability["resampling_scheme"] = f"class-stratified {FRACTION:.0%} unique-record training subsets; complete OOF stack refit"
    stability.to_csv(OUT / "directional_refit_stability.tsv", sep="\t", index=False)
    overall = {"requested_repeats": REPEATS, "completed_repeats": int(per_repeat.repeat.nunique()), "retained_directional_orthogroups": len(patterns),
               "median_reference_pattern_recovery": float(stability.reference_pattern_recovery_frequency.median()),
               "groups_recovered_at_least_80pct": int((stability.reference_pattern_recovery_frequency >= .8).sum())}
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")


def main() -> None:
    if not GENE_AUDIT.exists() or not PATTERNS.exists():
        raise FileNotFoundError("Missing retained directional audit inputs")
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repeats").mkdir(exist_ok=True)
    patterns, candidate_ids = original_patterns()
    if os.environ.get("DIRECTIONAL_REFIT_SUMMARISE_ONLY", "0") == "1":
        summarise(patterns)
        write_state(status="complete", requested_repeats=REPEATS, completed_files=len(list((OUT / "repeats").glob("*.tsv"))))
        return
    app_parts = app_arrays()
    mm_hits = pd.read_csv(
        MM_HITS, sep="\t", header=None,
        names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"],
        dtype={"query": str, "target": str},
    )
    ppi_dynamic = app_ppi_dynamic({}, app_parts)
    for repeat in range(REPEAT_START, REPEATS, REPEAT_STRIDE):
        for excluded in (False, True):
            label = "annotation_excluded" if excluded else "full"
            path = OUT / "repeats" / f"repeat_{repeat:03d}_{label}.tsv"
            if path.exists():
                continue
            write_state(status="running", repeat=repeat, model=label, completed_files=len(list((OUT / "repeats").glob("*.tsv"))), requested_repeats=REPEATS)
            output = []
            for species in SPECIES:
                output.append(score_species(species, repeat, excluded, candidate_ids[ALIASES[species]], app_parts, mm_hits, ppi_dynamic))
            pd.concat(output, ignore_index=True).to_csv(path, sep="\t", index=False)
            write_state(status="running", repeat=repeat, model=label, completed_files=len(list((OUT / "repeats").glob("*.tsv"))), requested_repeats=REPEATS)
    if SUMMARISE_AFTER:
        summarise(patterns)
        write_state(status="complete", requested_repeats=REPEATS, completed_files=len(list((OUT / "repeats").glob("*.tsv"))))
    else:
        write_state(status="shard_complete", requested_repeats=REPEATS, completed_files=len(list((OUT / "repeats").glob("*.tsv"))),
                    repeat_start=REPEAT_START, repeat_stride=REPEAT_STRIDE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "last_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_state(status="failed")
        raise
