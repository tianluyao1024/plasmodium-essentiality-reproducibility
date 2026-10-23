from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".tsv", ".txt"}
ABSOLUTE_PATH = re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
EXPECTED_METRICS = {
    "Pf": (5347, 4294, 526, 527, 0.8205940594059405, 0.8728263805155562),
    "Pb": (2096, 1687, 204, 205, 0.8740096618357488, 0.8964012622941426),
    "Pk": (4099, 3284, 408, 407, 0.9151080179788316, 0.9209502849754174),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return center - radius, center + radius


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def auc_and_average_precision(rows: list[dict[str, str]]) -> tuple[float, float, int, int]:
    parsed = [(int(row["label"]), float(row["probability"])) for row in rows]
    if any(label not in (0, 1) or not 0 <= score <= 1 for label, score in parsed):
        raise ValueError("Test prediction rows contain invalid binary labels or probabilities")
    positives = sum(label for label, _ in parsed)
    negatives = len(parsed) - positives
    if not positives or not negatives:
        raise ValueError("Both classes are required to compute AUROC/AUPRC")

    ordered_ascending = sorted(parsed, key=lambda item: item[1])
    rank_sum = 0.0
    i = 0
    while i < len(ordered_ascending):
        j = i + 1
        while j < len(ordered_ascending) and ordered_ascending[j][1] == ordered_ascending[i][1]:
            j += 1
        mean_rank = ((i + 1) + j) / 2
        rank_sum += mean_rank * sum(label for label, _ in ordered_ascending[i:j])
        i = j
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    ordered_descending = sorted(parsed, key=lambda item: item[1], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    i = 0
    while i < len(ordered_descending):
        j = i + 1
        while j < len(ordered_descending) and ordered_descending[j][1] == ordered_descending[i][1]:
            j += 1
        group_tp = sum(label for label, _ in ordered_descending[i:j])
        tp += group_tp
        fp += (j - i) - group_tp
        recall = tp / positives
        precision = tp / (tp + fp)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return auc, average_precision, positives, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen 2026-10-24 Nature Microbiology release candidate.")
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON verification report")
    args = parser.parse_args()
    root = args.release_dir.resolve()
    manifest_path = root / "release_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for relative, metadata in manifest["payload_files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing payload: {relative}")
            continue
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            errors.append(f"payload hash/size mismatch: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and ABSOLUTE_PATH.search(path.read_bytes()):
            errors.append(f"machine-specific absolute path: {relative}")

    main_rows = read_tsv(root / "source_data" / "Fig2" / "F2a_performance.tsv")
    test_dir = root / "source_data" / "Frozen_Test_Predictions"
    test_path = test_dir / "frozen_test_predictions.tsv"
    test_manifest = json.loads((test_dir / "manifest.json").read_text(encoding="utf-8"))
    test_rows = read_tsv(test_path)
    if sha256(test_path) != test_manifest["output_sha256"]:
        errors.append("test-only prediction table does not match its source manifest")
    if test_manifest.get("filter") != "split == test; no training or validation predictions exported":
        errors.append("test-only prediction manifest has an unexpected split policy")
    main_metrics = {}
    for row in main_rows:
        species = row["short_species"]
        if species not in EXPECTED_METRICS:
            continue
        expected = EXPECTED_METRICS[species]
        species_test_rows = [item for item in test_rows if item["species"] == row["species"]]
        auc, auprc, positives, negatives = auc_and_average_precision(species_test_rows)
        observed = (
            int(row["labelled_records"]), int(row["training_records"]),
            int(row["validation_records"]), int(row["test_records"]),
            auc, auprc,
        )
        if (observed[:4] != expected[:4] or observed[3] != len(species_test_rows)
                or positives + negatives != observed[3]
                or not close(observed[4], expected[4]) or not close(observed[5], expected[5])
                or not close(float(row["AUROC"]), auc) or not close(float(row["AUPRC"]), auprc)):
            errors.append(f"frozen main metrics differ from expected values: {species}")
        main_metrics[species] = {
            "labelled": observed[0], "train": observed[1], "validation": observed[2], "test": observed[3],
            "test_auroc": observed[4], "test_auprc": observed[5],
            "test_essential": positives, "test_nonessential": negatives,
        }
    if set(main_metrics) != set(EXPECTED_METRICS):
        errors.append("main performance table does not contain exactly Pf, Pb and Pk")

    loso_rows = read_tsv(root / "source_data" / "Fig3" / "F3a_LOSO.tsv")
    species_short = {
        "Plasmodium_falciparum": "Pf", "Plasmodium_berghei": "Pb",
        "Plasmodium_knowlesi": "Pk", "Toxoplasma_gondii": "Tg",
    }
    loso_species = sorted(species_short.get(row["species"], row["species"]) for row in loso_rows)
    if loso_species != ["Pb", "Pf", "Pk", "Tg"]:
        errors.append("four-species LOSO table is incomplete or contains unexpected species")

    allocations = read_tsv(root / "source_data" / "Fig2d_sensitivity" / "F2d_per_allocation.tsv")
    allocation_count = len({(row["species"], row["allocation"]) for row in allocations})
    if allocation_count != 15:
        errors.append(f"expected 15 Fig. 2d allocations, observed {allocation_count}")

    pair_rows = read_tsv(root / "source_data" / "Pk_dual_screen" / "source_gene_pairs.tsv")
    discordant = sum(row["screen_disagreement"].lower() == "true" for row in pair_rows)
    pair_count = len(pair_rows)
    low, high = wilson_interval(discordant, pair_count)
    summary_rows = read_tsv(root / "atlas" / "pk_dual_screen_summary.tsv")
    paired_summary = summary_rows[0]
    if pair_count != int(paired_summary["paired_genes"]) or discordant != int(paired_summary["discordant_calls"]):
        errors.append("Pk pair-level table does not reconcile with its published summary")
    if not close(low, float(paired_summary["wilson_95_ci_low"])) or not close(high, float(paired_summary["wilson_95_ci_high"])):
        errors.append("Pk Wilson interval does not reproduce")

    refits = read_tsv(root / "source_data" / "Fig4_final_frozen_20261024" / "Fig4d_refit_recovery.tsv")
    stable_groups = sum(row["final_refit_stable"].lower() == "true" for row in refits)
    if stable_groups != 184:
        errors.append(f"expected 184 refit-stable groups, observed {stable_groups}")

    s4_rows = read_tsv(root / "atlas" / "s4_label_exchange_runs.tsv")
    if len(s4_rows) != 60:
        errors.append(f"expected 60 S4 label-exchange runs, observed {len(s4_rows)}")

    figure_manifest = json.loads((root / "figures" / "main" / "figure_export_manifest.json").read_text(encoding="utf-8"))
    figure_checks = {}
    for item in figure_manifest["figures"]:
        figure = item["figure"]
        exported = root / "figures" / "main" / f"{figure}.png"
        expected_hash = item["embedded_png_sha256"]
        actual_hash = sha256(exported) if exported.is_file() else None
        figure_checks[figure] = actual_hash
        if actual_hash != expected_hash:
            errors.append(f"embedded figure hash mismatch: {figure}")

    report = {
        "release_id": manifest["release_id"],
        "payload_count": len(manifest["payload_files"]),
        "payload_hashes_and_portability": "passed" if not errors else "failed",
        "main_test_metrics": main_metrics,
        "metrics_recomputed_from_test_only_predictions": True,
        "four_species_loso": loso_species,
        "fig2d_allocation_count": allocation_count,
        "pk_dual_screen": {
            "paired_genes": pair_count,
            "discordant": discordant,
            "fraction": discordant / pair_count,
            "wilson_95_ci": [low, high],
            "continuous_spearman_rho": float(paired_summary["continuous_score_spearman_rho"]),
        },
        "refit_stable_groups": stable_groups,
        "s4_label_exchange_runs": len(s4_rows),
        "embedded_main_figure_hashes": figure_checks,
        "errors": errors,
        "scope_note": "This verifies frozen release outputs and source tables; it does not refit the nine-route model from the full 628-feature matrices.",
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
