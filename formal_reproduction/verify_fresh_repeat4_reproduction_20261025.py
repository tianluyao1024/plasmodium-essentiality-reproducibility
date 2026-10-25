from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FRESH = Path(r"D:\P100")
ARCHIVE = Path(r"D:\ProtoEss_original_structure_revision_20261017")
SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
VARIANTS = ("full", "annotation_excluded")
REPEAT = "fraction_0.800_repeat_004_seed_20260821"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    errors: list[str] = []
    source_path = FRESH / "source_manifest_v2.json"
    source = load(source_path)
    for raw_path, expected in source["files"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            errors.append(f"source snapshot mismatch: {path}")

    variant_results: list[dict] = []
    for species in SPECIES:
        root = FRESH / "pilot" / REPEAT / species
        old_root = ARCHIVE / "nine_route_refit_pilot" / REPEAT / species
        pair = load(root / "paired_refit_audit.json")
        if pair.get("repeat") != 4 or not pair.get("same_selected_training_ids") or not pair.get("same_fivefold_assignment"):
            errors.append(f"paired partition mismatch: {species}")

        for variant in VARIANTS:
            app_dir = root / variant / "application"
            old_app_dir = old_root / variant / "application"
            prediction = app_dir / "application_predictions.tsv"
            app_manifest = load(app_dir / "manifest.json")
            score_audit = load(app_dir / "independent_score_audit.json")
            pred_hash = sha256(prediction)
            if pred_hash != app_manifest["application_predictions_sha256"]:
                errors.append(f"application manifest hash mismatch: {species}/{variant}")
            archived_prediction_match = pred_hash == sha256(old_app_dir / "application_predictions.tsv")
            if not archived_prediction_match:
                errors.append(f"application predictions differ from archive: {species}/{variant}")
            if score_audit["prediction_sha256"] != pred_hash or score_audit["max_saved_model_score_error"] > 1e-12:
                errors.append(f"saved-model score replay failed: {species}/{variant}")
            if app_manifest["test_or_validation_labels_used_for_fitting_or_threshold"]:
                errors.append(f"held-out labels used in fit/threshold: {species}/{variant}")

            oof_dir = root / variant / "frozen_unlabelled"
            old_oof_dir = old_root / variant / "frozen_unlabelled"
            oof_manifest = load(oof_dir / "oof_manifest.json")
            oof_path = oof_dir / "oof_probabilities.npy"
            oof_hash = sha256(oof_path)
            archived_oof_match = oof_hash == sha256(old_oof_dir / "oof_probabilities.npy")
            if oof_hash != oof_manifest["oof_sha256"] or not archived_oof_match:
                errors.append(f"OOF hash/archive mismatch: {species}/{variant}")

            frame = pd.read_csv(prediction, sep="\t")
            model = joblib.load(app_dir / "models" / "nine_route_meta_lightgbm.joblib")
            order = app_manifest["input_order"]
            inputs = frame[[f"route_{name}_probability" for name in order]].to_numpy(dtype=np.float32)
            replay_error = float(np.max(np.abs(model.predict_proba(inputs)[:, 1] - frame.score.to_numpy())))
            if replay_error > 1e-12:
                errors.append(f"application model replay mismatch: {species}/{variant}")

            variant_results.append({
                "species": species,
                "variant": variant,
                "oof_sha256": oof_hash,
                "application_prediction_sha256": pred_hash,
                "matches_archived_repeat4": archived_oof_match and archived_prediction_match,
                "max_saved_model_score_error": replay_error,
                "training_validation_test_labels_used_for_fit_or_threshold": False,
            })

    fresh_repeat = FRESH / "nine_route_resampling_v2" / "repeats" / "repeat_004_seed_20260821"
    old_repeat = ARCHIVE / "nine_route_resampling_v1" / "repeats" / "repeat_004_seed_20260821"
    fresh_manifest, old_manifest = load(fresh_repeat / "manifest.json"), load(old_repeat / "manifest.json")
    fresh_table = fresh_repeat / "paired_directional_recovery.tsv"
    old_table = old_repeat / "paired_directional_recovery.tsv"
    table_hash = sha256(fresh_table)
    if table_hash != fresh_manifest["output_sha256"] or table_hash != sha256(old_table) or fresh_manifest["output_sha256"] != old_manifest["output_sha256"]:
        errors.append("fresh paired directional summary differs from archive or manifest")

    aggregate_path = FRESH / "nine_route_resampling_v2" / "aggregate_005_of_100_seed_20260821_nullable" / "manifest.json"
    aggregate = load(aggregate_path)
    if aggregate["completed_repeats"] != 5 or aggregate["expected_repeats"] != 100 or aggregate["final_stable_groups"] is not None:
        errors.append("partial aggregate is not correctly marked as non-final")

    result = {
        "status": "passed_single_repeat_only" if not errors else "failed",
        "scope": "fresh end-to-end reproduction of paired repeat 4; partial 5/100 aggregation is not a final stability result",
        "repeat": 4,
        "species_count": len(SPECIES),
        "variants_per_species": list(VARIANTS),
        "source_snapshot_file_count": len(source["files"]),
        "source_snapshot_sha256": sha256(source_path),
        "variant_results": variant_results,
        "candidate_groups": fresh_manifest["candidate_groups"],
        "recovered_by_both": fresh_manifest["reference_pattern_recovered_by_both"],
        "repeat_summary_sha256": table_hash,
        "matches_archived_repeat4_summary": table_hash == sha256(old_table),
        "partial_aggregate_manifest_sha256": sha256(aggregate_path),
        "partial_recovery_median": aggregate["median_partial_recovery_frequency"],
        "archived_100_repeat_refit_status": "historical saved outputs integrity-audited separately; repeats 0, 1, 2, 3 and 4 freshly rerun",
        "errors": errors,
    }
    output = Path(__file__).with_name("fresh_repeat4_reproduction_audit_20261025.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

