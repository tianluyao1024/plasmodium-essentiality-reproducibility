"""Create source-data tables for the leakage-safe four-species LOSO rerun.

The held-species probability vectors are produced by ``train_clade_loso_full_stack``.
This helper only recomputes descriptive metrics and class-stratified bootstrap
intervals from those frozen vectors; it never fits a model or selects a threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eukaryote_model_results" / "leakage_safe_clade_loso_v46" / "protozoa"
MANIFEST = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "clade" / "protozoa"
OUT = Path(r"D:\ProtoEss_loso_inner_validation_v46")
SPECIES = (
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
    "Toxoplasma_gondii",
)
BOOTSTRAPS = 1_000


def held_labels(species: str) -> np.ndarray:
    frames = [
        pd.read_csv(MANIFEST / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str})
        for split in ("train", "val", "test")
    ]
    held = pd.concat(
        [frame.loc[frame.species.astype(str).eq(species)] for frame in frames],
        ignore_index=True,
    )
    return held.label.eq("essential").astype(np.int8).to_numpy()


def calculate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    call = score >= threshold
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "Accuracy": float(accuracy_score(y, call)),
        "Precision": float(precision_score(y, call, zero_division=0)),
        "SN": float(recall_score(y, call, zero_division=0)),
        "SP": float(recall_score(1 - y, 1 - call, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, call)),
        "F1": float(f1_score(y, call, zero_division=0)),
    }


def interval(y: np.ndarray, score: np.ndarray, threshold: float, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = {name: np.empty(BOOTSTRAPS, dtype=float) for name in calculate(y, score, threshold)}
    for index in range(BOOTSTRAPS):
        sample = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        summary = calculate(y[sample], score[sample], threshold)
        for name, value in summary.items():
            values[name][index] = value
    return {name: (float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))) for name, value in values.items()}


def main() -> None:
    rows: list[dict[str, object]] = []
    for number, species in enumerate(SPECIES, start=1):
        folder = RESULTS / species
        payload = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        y = held_labels(species)
        score = np.load(folder / "test_prob.npy")
        threshold = float(payload["threshold"])
        if len(y) != len(score):
            raise RuntimeError(f"Held-label length mismatch for {species}: {len(y)} versus {len(score)}")
        observed = calculate(y, score, threshold)
        ci = interval(y, score, threshold, 20260901 + number)
        row: dict[str, object] = {
            "species": species,
            "model_provenance": "leakage-safe five-branch LOSO rerun; Transformer epoch selection used training-internal validation only",
            "scope": "clade_LOSO",
            "clade": "protozoa",
            "n": int(len(y)),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "threshold": threshold,
            "bootstrap_repeats": BOOTSTRAPS,
        }
        for name, value in observed.items():
            row[name] = value
            row[f"{name}_low"], row[f"{name}_high"] = ci[name]
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "four_species_loso_bootstrap.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
