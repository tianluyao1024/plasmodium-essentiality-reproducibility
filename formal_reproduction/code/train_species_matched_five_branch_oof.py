"""Rebuild the canonical species-matched five-branch formal OOF stack.

This is the single source-to-result entry point for the manuscript's formal
Pf/Pb/Pk model. It isolates every newly generated array under one output root,
so historical outputs remain untouched. The workflow is:
four fold-local tree/expert/prior branches + fold-local Transformer ->
five-column OOF matrix -> L2 logistic meta-learner -> validation MCC threshold
-> held-out test metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_scope_strict_oof_stack import read_frames


ROOT = Path.cwd()
DEFAULT_OUT = ROOT / "eukaryote_model_results" / "scope_full_oof_stack_reproduced_v32"
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
SEED = 42
TRANSFORMER_BATCH = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y, probability >= threshold) for threshold in grid]
    return float(grid[int(np.argmax(scores))])


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & ((probability < right) if right < 1 else (probability <= right))
        if mask.any():
            value += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def evaluate(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    negative = y == 0
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "F1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float((~predicted[negative]).mean()) if negative.any() else float("nan"),
        "MCC": float(matthews_corrcoef(y, predicted)),
        "Brier": float(brier_score_loss(y, probability)),
        "ECE_10bin": expected_calibration_error(y, probability),
        "threshold": float(threshold),
    }


def run_base_branches(species: str, output: Path, force: bool) -> None:
    tree_result = output / "four_branch_oof" / "within_species" / species / "metrics.json"
    transformer_result = output / "transformer_oof" / "within_species" / species / "metrics.json"
    if tree_result.exists() and transformer_result.exists() and not force:
        return
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(SEED),
        "STACK_ONLY": f"within_species:{species}",
        "STACK_FORCE": "1" if force else "0",
        "STRICT_OOF_OUT": str(output / "four_branch_oof"),
        "TRANSFORMER_ONLY": f"within_species:{species}",
        "TRANSFORMER_FORCE": "1" if force else "0",
        "TRANSFORMER_OOF_OUT": str(output / "transformer_oof"),
        "TRANSFORMER_BATCH": str(TRANSFORMER_BATCH),
    })
    subprocess.run([sys.executable, "train_scope_strict_oof_stack.py"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "train_scope_transformer_oof.py"], cwd=ROOT, env=env, check=True)


def branch_arrays(output: Path, species: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tree = output / "four_branch_oof" / "within_species" / species
    transformer = output / "transformer_oof" / "within_species" / species
    four = {split: np.load(tree / f"{split}_base.npy") for split in ("oof", "val", "test")}
    neural = {split: np.load(transformer / f"{split}_prob.npy") for split in ("oof", "val", "test")}
    for split in four:
        if len(four[split]) != len(neural[split]):
            raise RuntimeError(f"{species} {split}: tree/Transformer row count mismatch")
        if four[split].shape[1] != 4:
            raise RuntimeError(f"{species} {split}: expected four non-neural branches, found {four[split].shape}")
    return four, neural


def train_meta(species: str, output: Path) -> dict[str, object]:
    frames = read_frames("within_species", species)
    labels = {split: frames[split].label.eq("essential").astype(np.int8).to_numpy() for split in frames}
    four, neural = branch_arrays(output, species)
    branches = {split: np.column_stack([four[split], neural[split]]) for split in four}
    branch_label_split = {"oof": "train", "val": "val", "test": "test"}
    for split, matrix in branches.items():
        if len(matrix) != len(labels[branch_label_split[split]]) or matrix.shape[1] != 5:
            raise RuntimeError(f"{species} {split}: invalid five-branch matrix {matrix.shape}")
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    meta.fit(branches["oof"], labels["train"])
    probability = {
        "oof": meta.predict_proba(branches["oof"])[:, 1],
        "val": meta.predict_proba(branches["val"])[:, 1],
        "test": meta.predict_proba(branches["test"])[:, 1],
    }
    threshold = mcc_threshold(labels["val"], probability["val"])

    four_meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    four_meta.fit(four["oof"], labels["train"])
    probability_four = {split: four_meta.predict_proba(four[split])[:, 1] for split in four}
    threshold_four = mcc_threshold(labels["val"], probability_four["val"])

    out = output / "within_species" / species
    out.mkdir(parents=True, exist_ok=True)
    for split in branches:
        np.save(out / f"{split}_branches.npy", branches[split])
        np.save(out / f"{split}_prob.npy", probability[split])
        np.save(out / f"{split}_four_branch_prob.npy", probability_four[split])
    pd.DataFrame({
        "branch": ["LightGBM", "CatBoost", "species_expert", "species_prevalence", "Transformer"],
        "coefficient": meta.coef_.ravel(),
    }).to_csv(out / "meta_coefficients.tsv", sep="\t", index=False)
    result = {
        "scope": "within_species",
        "name": species,
        "formal_model": "five_branch_oof_l2_logistic_meta",
        "branches": ["LightGBM", "CatBoost", "species_expert", "species_prevalence", "Transformer"],
        "seed": SEED,
        "transformer_batch_size": TRANSFORMER_BATCH,
        "meta_learner": {"type": "L2 logistic regression", "C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
        "threshold_selection": "validation MCC maximisation on 0.05-0.95 grid in 0.005 increments",
        "threshold": threshold,
        "oof": evaluate(labels["train"], probability["oof"], threshold),
        "validation": evaluate(labels["val"], probability["val"], threshold),
        "test": evaluate(labels["test"], probability["test"], threshold),
        "four_branch_without_transformer": {
            "threshold": threshold_four,
            "test": evaluate(labels["test"], probability_four["test"], threshold_four),
        },
        "input_manifest_sha256": {split: sha256(ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species" / species / f"{split}_metadata.tsv") for split in ("train", "val", "test")},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output: Path, results: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        full = result["test"]
        reduced = result["four_branch_without_transformer"]["test"]
        rows.append({
            "species": result["name"],
            "five_branch_AUROC": full["AUROC"],
            "four_branch_AUROC": reduced["AUROC"],
            "delta_AUROC_five_minus_four": full["AUROC"] - reduced["AUROC"],
            "five_branch_AUPRC": full["AUPRC"],
            "four_branch_AUPRC": reduced["AUPRC"],
            "delta_AUPRC_five_minus_four": full["AUPRC"] - reduced["AUPRC"],
            "five_branch_threshold": result["threshold"],
            "four_branch_threshold": result["four_branch_without_transformer"]["threshold"],
        })
    pd.DataFrame(rows).to_csv(output / "five_vs_four_branch_test_metrics.tsv", sep="\t", index=False)
    manifest = {
        "purpose": "isolated clean-reproduction output for the canonical species-matched five-branch stack",
        "species": list(SPECIES),
        "seed": SEED,
        "transformer_batch_size": TRANSFORMER_BATCH,
        "input_scope": "eukaryote_model_datasets/scope_feature_sets_v2/within_species",
        "split_scope": "eukaryote_model_datasets/unified_v3_experiment_manifests_pre95human/within_species",
        "outputs": ["four_branch_oof", "transformer_oof", "within_species", "five_vs_four_branch_test_metrics.tsv"],
    }
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--species", choices=SPECIES)
    parser.add_argument("--force", action="store_true", help="Retrain isolated branch checkpoints even when present.")
    args = parser.parse_args()
    seed_everything(SEED)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = (args.species,) if args.species else SPECIES
    results = []
    for species in selected:
        run_base_branches(species, output, args.force)
        results.append(train_meta(species, output))
    if len(selected) == len(SPECIES):
        write_summary(output, results)
    print(json.dumps({"output_root": str(output), "species": list(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
