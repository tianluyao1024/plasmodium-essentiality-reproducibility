"""Score the three complete malaria proteomes with the frozen full OOF stack.

This is the application-side counterpart to the functional-annotation-excluded
run.  It uses the same species-specific formal splits, the same three-fold OOF
meta-training arrays and the same five branch architecture, but retains the
GO/Pfam/InterPro block.  Its sole purpose is to create an apples-to-apples
whole-proteome comparator for the controlled feature-exclusion analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from run_ncomms_annotation_excluded import (
    ALIASES,
    APP_META,
    CHECKPOINTS,
    MM_HITS,
    MODEL_OUT,
    OUT,
    PRED_OUT,
    RUN,
    SPECIES,
    app_arrays,
    app_ppi_dynamic,
    application_matrix,
    edge_bundle,
    lgbm,
    catboost,
    read_all,
    read_frames,
    seed_all,
    train_predict,
)
from run_plasmodium_cluster_blocked_full_oof import all_vs_all, dynamic_x
from train_scope_transformer_oof import split_groups


ROOT = Path(__file__).resolve().parent
# Optional overrides make it possible to freeze a new downstream lineage
# without overwriting a previously submitted atlas.
FULL_OUT = Path(os.environ.get("NCOMMS_FULL_APPLICATION_OUT", str(OUT / "full_stack_application_baseline")))
FULL_PRED = FULL_OUT / "genomewide_predictions"
FULL_CHECKPOINTS = FULL_OUT / "checkpoints"
STATE = FULL_OUT / "state.json"
CANONICAL_ROOT = os.environ.get("NCOMMS_CANONICAL_FIVE_BRANCH_ROOT", "")
if CANONICAL_ROOT:
    _canonical = Path(CANONICAL_ROOT)
    TREE = _canonical / "four_branch_oof" / "within_species"
    TRANSFORMER = _canonical / "transformer_oof" / "within_species"
    FULL = _canonical / "within_species"
else:
    TREE = ROOT / "eukaryote_model_results" / "scope_strict_oof_stack" / "within_species"
    TRANSFORMER = ROOT / "eukaryote_model_results" / "scope_transformer_oof" / "within_species"
    FULL = ROOT / "eukaryote_model_results" / "scope_full_oof_stack" / "within_species"


def write_state(**updates: object) -> None:
    current: dict[str, object] = {}
    if STATE.exists():
        try:
            current = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    # A restarted run must not keep an exception from a prior attempt visible
    # as if it were a current failure.
    if updates.get("status") == "running":
        current.pop("error", None)
        current.pop("traceback", None)
    current.update(updates)
    current["updated_at_epoch"] = time.time()
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def train_and_score(species: str, app_parts: dict[str, np.ndarray], mm_hits: pd.DataFrame,
                    ppi_dynamic: np.ndarray) -> dict[str, object]:
    frames = read_frames(species)
    metadata, raw = read_all(species)
    labels = metadata.label.eq("essential").astype(np.int8).to_numpy()
    ids = metadata.record_id.astype(str).to_numpy()
    train = np.flatnonzero(metadata.formal_split.eq("train"))
    val = np.flatnonzero(metadata.formal_split.eq("val"))
    test = np.flatnonzero(metadata.formal_split.eq("test"))
    id_to_species = dict(zip(ids, metadata.species.astype(str)))
    cache = FULL_CHECKPOINTS / species
    cache.mkdir(parents=True, exist_ok=True)
    hits = all_vs_all(species, cache)
    edges = edge_bundle(species)
    x_train = dynamic_x(raw[train], ids[train], ids[train], labels[train], id_to_species, hits, edges)
    x_val = dynamic_x(raw[val], ids[val], ids[train], labels[train], id_to_species, hits, edges)
    x_test = dynamic_x(raw[test], ids[test], ids[train], labels[train], id_to_species, hits, edges)
    # application_matrix returns the annotation-masked matrix by design.  Add
    # its original counterpart locally by restoring the 128 transformed columns
    # from the same train-only app construction path.
    from score_final_application_models import app_static
    static = app_static("within_species", species, frames, app_parts)
    selected = mm_hits.loc[np.isin(mm_hits["target"].astype(str).to_numpy(), frames["train"].record_id.astype(str).to_numpy())].copy()
    selected["identity"] = selected.pident.astype(float) / 100.0
    selected["coverage"] = selected.alnlen.astype(float) / selected[["qlen", "tlen"]].astype(float).max(axis=1)
    selected["score"] = selected.identity * selected.coverage
    from train_scope_strict_oof_stack import aggregate_mm
    train_ids = frames["train"].record_id.astype(str).to_numpy()
    app_ids = APP_META.record_id.astype(str).to_numpy()
    labels_map = dict(zip(train_ids, frames["train"].label.eq("essential").astype(float)))
    target_species = dict(zip(train_ids, frames["train"].species.astype(str)))
    reverse_alias = {value: key for key, value in ALIASES.items()}
    query_species = dict(zip(APP_META.record_id.astype(str), APP_META.species.astype(str).map(lambda value: reverse_alias.get(value, value))))
    dynamic_mm = aggregate_mm(selected.loc[np.isin(selected["query"].astype(str).to_numpy(), app_ids)], app_ids, labels_map, target_species, query_species)
    x_app = np.concatenate([static, dynamic_mm, ppi_dynamic], axis=1).astype(np.float32)
    if x_app.shape[1] != x_train.shape[1]:
        raise RuntimeError(f"{species}: app {x_app.shape}, train {x_train.shape}")
    model_lgbm = lgbm(x_train, labels[train], 42)
    model_cat = catboost(x_train, labels[train], 42)
    model_expert = lgbm(x_train, labels[train], 43)
    _, app_transformer, _ = train_predict(
        split_groups(x_train, 1), labels[train], split_groups(x_val, 1), labels[val], split_groups(x_app, 1), 900,
    )
    app_branches = np.column_stack([
        model_lgbm.predict_proba(x_app)[:, 1],
        model_cat.predict_proba(x_app)[:, 1],
        model_expert.predict_proba(x_app)[:, 1],
        np.repeat(labels[train].mean(), len(x_app)),
        app_transformer,
    ])
    oof = np.column_stack([np.load(TREE / species / "oof_base.npy"), np.load(TRANSFORMER / species / "oof_prob.npy")])
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42).fit(oof, labels[train])
    score = meta.predict_proba(app_branches)[:, 1]
    threshold = float(json.loads((FULL / species / "metrics.json").read_text(encoding="utf-8"))["threshold"])
    mask = APP_META.species.astype(str).eq(ALIASES[species]).to_numpy()
    table = APP_META.loc[mask, ["record_id", "species", "gene_id", "protein_id", "protein_length"]].copy()
    table["full_formal_stack_score"] = score[mask]
    table["full_formal_stack_threshold"] = threshold
    table["full_formal_stack_margin"] = table.full_formal_stack_score - threshold
    table["full_formal_stack_predicted_essential"] = table.full_formal_stack_score >= threshold
    table.to_csv(FULL_PRED / f"{species}_genomewide.tsv", sep="\t", index=False)
    return {
        "species": species,
        "genomewide_n": int(mask.sum()),
        "threshold": threshold,
        "predicted_essential_n": int(table.full_formal_stack_predicted_essential.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score complete malaria proteomes with the full formal five-branch stack.")
    parser.add_argument("--species", choices=SPECIES, help="Score one species only; omitted scores all three.")
    args = parser.parse_args()
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    FULL_PRED.mkdir(parents=True, exist_ok=True)
    FULL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    seed_all()
    write_state(status="running", stage="load_application_features", pid=os.getpid())
    parts = app_arrays()
    hits = pd.read_csv(MM_HITS, sep="\t", header=None, names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"], dtype={"query": str, "target": str})
    ppi = app_ppi_dynamic({}, parts)
    rows = []
    try:
        selected_species = (args.species,) if args.species else SPECIES
        for species in selected_species:
            target = FULL_PRED / f"{species}_genomewide.tsv"
            if target.exists():
                rows.append({"species": species, "status": "reused"})
                continue
            write_state(status="running", stage="score_species", species=species, pid=os.getpid())
            rows.append(train_and_score(species, parts, hits, ppi))
        pd.DataFrame(rows).to_csv(FULL_OUT / "full_stack_application_summary.tsv", sep="\t", index=False)
        all_paths = [FULL_PRED / f"{species}_genomewide.tsv" for species in SPECIES]
        if all(path.exists() for path in all_paths):
            combined = pd.concat([pd.read_csv(path, sep="\t") for path in all_paths], ignore_index=True)
            combined.to_csv(FULL_OUT / "full_formal_stack_genomewide_all_three.tsv", sep="\t", index=False)
            (FULL_OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            write_state(status="complete", stage="complete", pid=os.getpid())
    except Exception as error:
        trace = traceback.format_exc()
        (FULL_OUT / "last_error.log").write_text(trace, encoding="utf-8")
        write_state(status="failed", stage="exception", error=str(error), traceback=trace, pid=os.getpid())
        raise


if __name__ == "__main__":
    main()
