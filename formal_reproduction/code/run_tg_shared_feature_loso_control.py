"""Re-run the Tg source-composition test without species-specific shortcuts.

For each held malaria species, this runner compares the identical full held
species record set under (i) the other two malaria sources and (ii) those
sources plus Toxoplasma gondii. Both arms remove clade-level species one-hot
features, the species expert and the species-prevalence branch. Every task
checkpoints through train_clade_loso_full_stack.py and can be resumed safely.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MANIFEST = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "clade" / "protozoa"
BASE_SUBDIR = os.environ.get(
    "TG_SHARED_CONTROL_BASE_SUBDIR",
    "clade_loso_tg_shared_molecular_control_malaria_only",
)
AUG_SUBDIR = os.environ.get(
    "TG_SHARED_CONTROL_AUG_SUBDIR",
    "clade_loso_tg_shared_molecular_control_plus_tg",
)
BASE = ROOT / "eukaryote_model_results" / BASE_SUBDIR / "protozoa"
AUG = ROOT / "eukaryote_model_results" / AUG_SUBDIR / "protozoa"
OUT = Path(os.environ.get(
    "TG_SHARED_CONTROL_OUT",
    str(ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "tg_augmentation_shared_molecular_control"),
))
STATE = OUT / "state.json"
MALARIA = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
RUN_FORCE = os.environ.get("TG_SHARED_CONTROL_FORCE", "0") == "1"


def write_state(**updates: object) -> None:
    current: dict[str, object] = {}
    if STATE.exists():
        try:
            current = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    current.update(updates)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def held_records(held: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(MANIFEST / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str})
        parts.append(frame.loc[frame.species.astype(str).eq(held)].copy())
    return pd.concat(parts, ignore_index=True)


def run_one(held: str, source_mode: str, subdir: str) -> None:
    marker = ROOT / "eukaryote_model_results" / subdir / "protozoa" / held / "metrics.json"
    if marker.exists() and not RUN_FORCE:
        return
    env = os.environ.copy()
    env.update({
        "CLOSO_ONLY": f"protozoa:{held}",
        "CLOSO_FORCE": "1" if RUN_FORCE else "0",
        "CLOSO_SOURCE_MODE": source_mode,
        "CLOSO_SHARED_FEATURES_ONLY": "1",
        "CLOSO_OUT_SUBDIR": subdir,
        "TRANSFORMER_BATCH": "64",
    })
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{held}__{source_mode}.log"
    write_state(status="running", stage="fit", held_out_species=held, source_mode=source_mode, log=str(log))
    with log.open("a", encoding="utf-8") as fh:
        process = subprocess.run([PYTHON, "-u", "train_clade_loso_full_stack.py"], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not marker.exists():
        raise RuntimeError(f"{held} {source_mode} failed, see {log}")


def bootstrap_delta(y: np.ndarray, augmented: np.ndarray, baseline: np.ndarray, metric: str, seed: int) -> tuple[float, float, float]:
    scorer = roc_auc_score if metric == "AUROC" else average_precision_score
    observed = float(scorer(y, augmented) - scorer(y, baseline))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = np.empty(2000, dtype=float)
    for i in range(len(boot)):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        boot[i] = scorer(y[idx], augmented[idx]) - scorer(y[idx], baseline[idx])
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarise() -> None:
    rows: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for idx, held in enumerate(MALARIA, start=1):
        records = held_records(held)
        y = records.label.eq("essential").astype(np.int8).to_numpy()
        base = np.load(BASE / held / "test_prob.npy")
        aug = np.load(AUG / held / "test_prob.npy")
        if not (len(records) == len(base) == len(aug)):
            raise RuntimeError(f"Held-record mismatch for {held}: {len(records)}, {len(base)}, {len(aug)}")
        row_hash = hashlib.sha256("\n".join(records.record_id.astype(str)).encode()).hexdigest()
        for model, probability, folder, sources in (
            ("two_malaria_sources_shared_features", base, BASE, ";".join(x for x in MALARIA if x != held)),
            ("two_malaria_plus_Tg_shared_features", aug, AUG, ";".join([*(x for x in MALARIA if x != held), "Toxoplasma_gondii"])),
        ):
            meta = json.loads((folder / held / "metrics.json").read_text(encoding="utf-8"))
            threshold = float(meta["threshold"])
            rows.append({
                "held_out_species": held, "training_sources": sources, "model": model,
                "n_test": len(records), "n_essential": int(y.sum()), "n_nonessential": int((1 - y).sum()),
                "test_record_order_sha256": row_hash, "threshold": threshold,
                "AUROC": float(roc_auc_score(y, probability)),
                "AUPRC": float(average_precision_score(y, probability)),
                "F1": float(f1_score(y, probability >= threshold, zero_division=0)),
                "species_onehot_removed": True, "species_expert_removed": True, "species_prevalence_removed": True,
            })
        for metric in ("AUROC", "AUPRC"):
            delta, lo, hi = bootstrap_delta(y, aug, base, metric, 20260827 + idx * 100 + len(metric))
            deltas.append({
                "held_out_species": held, "metric": metric,
                "contrast": "two_malaria_plus_Tg_shared_features minus two_malaria_sources_shared_features",
                "delta": delta, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "n_bootstrap": 2000,
                "test_record_order_sha256": row_hash,
            })
        pd.DataFrame({
            "record_id": records.record_id.astype(str), "species": held, "label": y,
            "two_malaria_sources_shared_features_probability": base,
            "two_malaria_plus_Tg_shared_features_probability": aug,
        }).to_csv(OUT / f"{held}_paired_test_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(OUT / "shared_molecular_tg_augmentation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "shared_molecular_tg_augmentation_paired_bootstrap.tsv", sep="\t", index=False)
    (OUT / "README.md").write_text(
        "# Shared-molecular-feature Tg source-composition control\n\n"
        "The two-malaria and Tg-augmented LOSO arms use identical held-malaria record sets. Both arms remove all clade-level species one-hot columns, the species-specific expert branch and the species-prevalence branch. Label-derived MMseqs2/PPI features remain fold-local.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for held in MALARIA:
            run_one(held, "malaria_only", BASE_SUBDIR)
            run_one(held, "all_other_species", AUG_SUBDIR)
        summarise()
        write_state(status="complete", stage="complete", completed_species=list(MALARIA))
    except Exception as error:
        write_state(status="failed", stage="exception", error=str(error))
        raise


if __name__ == "__main__":
    main()
