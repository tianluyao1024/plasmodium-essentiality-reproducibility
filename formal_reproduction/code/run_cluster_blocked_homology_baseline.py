"""Compare the locked full stack with training-only homology baselines.

The five existing MMseqs2 30% cluster allocations already isolate complete
families across train/validation/test.  This analysis reuses those frozen
allocations and scores each validation/test record only from MMseqs2 hits to
the corresponding training partition.  It therefore asks whether the full
model retains discrimination beyond direct, training-only homology transfer; it
does not alter the formal atlas model. Three prespecified baselines are used:
nearest-essential, signed nearest-neighbour and similarity-weighted top-5 kNN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
BASE = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "cluster_blocked_full_oof" / "repeated_allocations"
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "cluster_blocked_homology_baseline"
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
COLS = ["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"]


def homology_scores(hits: pd.DataFrame, query: set[str], train_labels: pd.Series) -> pd.DataFrame:
    """Return three label-aware homology-only scores using training hits only."""
    h = hits[hits["query"].isin(query) & hits["target"].isin(train_labels.index)].copy()
    h = h[h["query"].ne(h["target"])]
    h["score"] = (h.pident.astype(float) / 100.0) * (
        h.alnlen.astype(float) / h[["qlen", "tlen"]].astype(float).max(axis=1)
    )
    h["essential_train"] = h["target"].map(train_labels).astype(float)
    positive = h[h.essential_train.eq(1)].groupby("query").score.max().rename("nearest_essential_score")
    negative = h[h.essential_train.eq(0)].groupby("query").score.max().rename("nearest_nonessential_score")
    # A label-aware nearest-neighbour classifier: strongest positive minus strongest negative hit.
    scores = pd.concat([positive, negative], axis=1).fillna(0.0)
    scores["signed_nearest_neighbour_score"] = scores.nearest_essential_score - scores.nearest_nonessential_score
    # Prespecified k=5 weighted voting over the closest labelled training neighbours.
    top5 = h.sort_values(["query", "score"], ascending=[True, False]).groupby("query", as_index=False).head(5)
    weighted = top5.groupby("query").apply(
        lambda q: float((q.score * q.essential_train).sum() / q.score.sum()) if q.score.sum() > 0 else 0.5,
        include_groups=False,
    ).rename("weighted_top5_knn_score")
    return scores.join(weighted, how="outer").fillna(0.0)


def bootstrap_delta(
    y: np.ndarray, full: np.ndarray, baseline: np.ndarray, metric, repeats: int = 1000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260812)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(metric(y[ix], full[ix]) - metric(y[ix], baseline[ix]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed_dir in sorted(BASE.glob("seed_*")):
        for species in SPECIES:
            folder = seed_dir / "mmseqs30" / species
            manifest = pd.read_csv(folder / "cluster_blocked_manifest.tsv", sep="\t", dtype={"record_id": str})
            hits = pd.read_csv(folder / "all_vs_all.tsv", sep="\t", names=COLS, dtype={"query": str, "target": str})
            train = manifest[manifest.cluster_split.eq("training")].copy()
            train_labels = train.set_index("record_id").label.eq("essential")
            for split, pred_name in (("validation", "val_prob.npy"), ("test", "test_prob.npy")):
                frame = manifest[manifest.cluster_split.eq(split)].copy()
                baseline = homology_scores(hits, set(frame.record_id), train_labels).reindex(frame.record_id, fill_value=0.0)
                full = np.load(folder / pred_name)
                y = frame.label.eq("essential").astype(int).to_numpy()
                if len(full) != len(y):
                    raise RuntimeError(f"{seed_dir.name} {species} {split}: prediction length mismatch")
                row = {
                    "allocation": seed_dir.name,
                    "species": species,
                    "split": split,
                    "n": len(y),
                    "essential_fraction": float(y.mean()),
                    "full_stack_AUROC": float(roc_auc_score(y, full)),
                    "full_stack_AUPRC": float(average_precision_score(y, full)),
                    "zero_nearest_essential_score_fraction": float((baseline.nearest_essential_score == 0).mean()),
                }
                for name in ("nearest_essential_score", "signed_nearest_neighbour_score", "weighted_top5_knn_score"):
                    score = baseline[name].to_numpy()
                    prefix = name.replace("_score", "")
                    row[f"{prefix}_AUROC"] = float(roc_auc_score(y, score))
                    row[f"{prefix}_AUPRC"] = float(average_precision_score(y, score))
                    if split == "test":
                        lo, hi = bootstrap_delta(y, full, score, roc_auc_score)
                        row[f"AUROC_delta_full_minus_{prefix}"] = row["full_stack_AUROC"] - row[f"{prefix}_AUROC"]
                        row[f"AUROC_delta_full_minus_{prefix}_ci_low"] = lo
                        row[f"AUROC_delta_full_minus_{prefix}_ci_high"] = hi
                        lo, hi = bootstrap_delta(y, full, score, average_precision_score)
                        row[f"AUPRC_delta_full_minus_{prefix}"] = row["full_stack_AUPRC"] - row[f"{prefix}_AUPRC"]
                        row[f"AUPRC_delta_full_minus_{prefix}_ci_low"] = lo
                        row[f"AUPRC_delta_full_minus_{prefix}_ci_high"] = hi
                rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "cluster_blocked_homology_baseline_by_allocation.tsv", sep="\t", index=False)
    test = detail[detail.split.eq("test")]
    summary = test.groupby("species", as_index=False).agg(
        allocations=("allocation", "nunique"),
        full_stack_AUROC_mean=("full_stack_AUROC", "mean"),
        full_stack_AUROC_sd=("full_stack_AUROC", "std"),
        nearest_essential_AUROC_mean=("nearest_essential_AUROC", "mean"),
        nearest_essential_AUROC_sd=("nearest_essential_AUROC", "std"),
        signed_nearest_neighbour_AUROC_mean=("signed_nearest_neighbour_AUROC", "mean"),
        signed_nearest_neighbour_AUROC_sd=("signed_nearest_neighbour_AUROC", "std"),
        weighted_top5_knn_AUROC_mean=("weighted_top5_knn_AUROC", "mean"),
        weighted_top5_knn_AUROC_sd=("weighted_top5_knn_AUROC", "std"),
        full_stack_AUPRC_mean=("full_stack_AUPRC", "mean"),
        nearest_essential_AUPRC_mean=("nearest_essential_AUPRC", "mean"),
        signed_nearest_neighbour_AUPRC_mean=("signed_nearest_neighbour_AUPRC", "mean"),
        weighted_top5_knn_AUPRC_mean=("weighted_top5_knn_AUPRC", "mean"),
        nearest_essential_delta_mean=("AUROC_delta_full_minus_nearest_essential", "mean"),
        signed_nearest_neighbour_delta_mean=("AUROC_delta_full_minus_signed_nearest_neighbour", "mean"),
        weighted_top5_knn_delta_mean=("AUROC_delta_full_minus_weighted_top5_knn", "mean"),
        nearest_essential_AUPRC_delta_mean=("AUPRC_delta_full_minus_nearest_essential", "mean"),
        signed_nearest_neighbour_AUPRC_delta_mean=("AUPRC_delta_full_minus_signed_nearest_neighbour", "mean"),
        weighted_top5_knn_AUPRC_delta_mean=("AUPRC_delta_full_minus_weighted_top5_knn", "mean"),
        zero_nearest_essential_score_fraction_mean=("zero_nearest_essential_score_fraction", "mean"),
    )
    summary.to_csv(OUT / "cluster_blocked_homology_baseline_summary.tsv", sep="\t", index=False)
    (OUT / "analysis_manifest.json").write_text(json.dumps({
        "design": "Five frozen 30% MMseqs2 cluster-blocked allocations; validation/test genes scored only from labelled training neighbours.",
        "baselines": {"nearest_essential": "max(identity x reciprocal-length coverage) over essential training hits", "signed_nearest_neighbour": "max similarity to essential training hit minus max similarity to non-essential training hit", "weighted_top5_knn": "similarity-weighted essential-label fraction over the five closest labelled training hits"},
        "comparison": "locked full OOF stack prediction on the identical held-out records",
        "bootstrap": "1000 paired class-stratified test-set resamples for each per-allocation AUROC and AUPRC delta",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
