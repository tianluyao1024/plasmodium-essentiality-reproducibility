"""Add deterministic stratified-bootstrap uncertainty intervals to Fig. 2a."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get(
    "PROTOESS_SUBMISSION_OUT",
    ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v46_reproducible",
))
FROZEN_RESULTS = ROOT / "eukaryote_model_results" / "scope_full_oof_stack_reproduced_v32" / "within_species"
LEAKAGE_SAFE_ROOT = Path(os.environ.get(
    "PROTOESS_LEAKAGE_SAFE_RESULTS",
    r"D:\ProtoEss_formal_inner_validation_v46",
))
MANIFESTS = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species"
SPECIES = {
    "P. falciparum": "Plasmodium_falciparum",
    "P. berghei": "Plasmodium_berghei",
    "P. knowlesi": "Plasmodium_knowlesi",
}


def bootstrap(y: np.ndarray, p: np.ndarray, seed: int, draws: int = 2_000) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    aurocs, auprcs = [], []
    for _ in range(draws):
        idx = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        aurocs.append(roc_auc_score(y[idx], p[idx]))
        auprcs.append(average_precision_score(y[idx], p[idx]))
    return (*np.quantile(aurocs, [0.025, 0.975]), *np.quantile(auprcs, [0.025, 0.975]))


def main() -> None:
    path = OUT / "Source_Data" / "Fig2a_species_matched_performance.tsv"
    table = pd.read_csv(path, sep="\t")
    table = table.loc[:, ~table.columns.duplicated()].copy()
    table = table.drop(columns=[col for col in table.columns if col.startswith((
        "AUROC_ci95_", "AUPRC_ci95_", "bootstrap_method", "bootstrap_seed"
    ))], errors="ignore")
    leakage_safe_results = LEAKAGE_SAFE_ROOT / "within_species"
    use_leakage_safe = all((leakage_safe_results / species / "metrics.json").exists() for species in SPECIES.values())
    results = leakage_safe_results if use_leakage_safe else FROZEN_RESULTS
    rows = []
    for i, row in table.iterrows():
        name = SPECIES[row["species"]]
        p = np.load(results / name / "test_prob.npy")
        raw_labels = pd.read_csv(MANIFESTS / name / "test_metadata.tsv", sep="\t")["label"]
        labels = raw_labels.map({"essential": 1, "nonessential": 0, "non_essential": 0, "non-essential": 0}).to_numpy(dtype=int)
        if len(p) != len(labels):
            raise ValueError(f"Test probability/label mismatch for {name}: {len(p)} vs {len(labels)}")
        lo_auc, hi_auc, lo_prc, hi_prc = bootstrap(labels, p, 4602 + i)
        metrics = json.loads((results / name / "metrics.json").read_text(encoding="utf-8"))["test"]
        table.loc[table.index[i], "AUROC"] = metrics["AUROC"]
        table.loc[table.index[i], "AUPRC"] = metrics["AUPRC"]
        # v46 source tables inherited two historical column conventions.
        # Keep either convention synchronized with the validation-fixed value
        # recorded by the leakage-safe model rather than retaining a stale
        # threshold from the pre-v46 package.
        for threshold_column in ("threshold", "validation_MCC_threshold"):
            if threshold_column in table.columns:
                table.loc[table.index[i], threshold_column] = metrics["threshold"]
        rows.append({"AUROC_ci95_low": lo_auc, "AUROC_ci95_high": hi_auc,
                     "AUPRC_ci95_low": lo_prc, "AUPRC_ci95_high": hi_prc,
                     "bootstrap_method": "2,000 class-stratified resamples of held-out test rows",
                     "bootstrap_seed": 4602 + i,
                     "result_lineage": "inner-validation leakage-safe five-branch rerun" if use_leakage_safe else "frozen historical five-branch result"})
    enhanced = pd.concat([table, pd.DataFrame(rows)], axis=1)
    enhanced.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    main()
