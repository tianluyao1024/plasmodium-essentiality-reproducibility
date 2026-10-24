"""Check a completed application refit and compare it to the frozen scores."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")
BASELINE = Path(r"D:\ProtoEss_feature_matrix_repair_20261015\nine_route_v3_5fold_fixed_params_20261016\full_application\genomewide_predictions")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--variant", choices=("full", "annotation_excluded"), default="annotation_excluded")
    args = parser.parse_args()
    directory = (ROOT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}"
                 / args.species / args.variant / "application")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["species"] != args.species or manifest["fraction"] != args.fraction
            or manifest["repeat"] != args.repeat or manifest["variant"] != args.variant):
        raise ValueError("Application manifest does not match requested experiment")
    path = directory / "application_predictions.tsv"
    if sha256(path) != manifest["application_predictions_sha256"]:
        raise ValueError("Application predictions do not match manifest")
    data = pd.read_csv(path, sep="\t", dtype={"record_id": str})
    if data.record_id.duplicated().any() or len(data) != manifest["n_application_genes"]:
        raise ValueError("Invalid application IDs or row count")
    model = joblib.load(directory / "models" / "nine_route_meta_lightgbm.joblib")
    X = data[[f"route_{name}_probability" for name in manifest["input_order"]]].to_numpy(dtype=np.float32)
    score = model.predict_proba(X)[:, 1]
    discrepancy = np.abs(score - data.score.to_numpy())
    if discrepancy.max() > 1e-6:
        raise ValueError("Stored model does not reproduce scores")
    threshold = float(manifest["threshold_from_training_cv"])
    if not np.array_equal(score >= threshold, data.predicted_essential.to_numpy(dtype=bool)):
        raise ValueError("Score/threshold classification mismatch")
    original = pd.read_csv(BASELINE / f"{args.species}_nine_route.tsv", sep="\t", dtype={"record_id": str})
    paired = data.merge(original, on="record_id", how="inner", validate="one_to_one", suffixes=("_excluded", "_full"))
    if len(paired) != len(data) or len(paired) != len(original):
        raise ValueError("Application sets differ")
    old = paired.predicted_essential_at_training_cv_threshold.to_numpy(dtype=bool)
    new = paired.predicted_essential.to_numpy(dtype=bool)
    delta = np.abs(paired.score - paired.nine_route_lightgbm_score)
    result = {"status": "single-species application check, not an independent performance or stability estimate",
              "species": args.species, "variant": args.variant, "fraction": args.fraction, "repeat": args.repeat,
              "application_records": len(data),
              "prediction_sha256": sha256(path), "max_saved_model_score_error": float(discrepancy.max()),
              "refit_threshold": threshold,
              "refit_predicted_essential": int(new.sum()),
              "frozen_predicted_essential": int(old.sum()),
              "calls_changed": int((new != old).sum()),
              "mean_absolute_score_difference": float(delta.mean()),
              "caveats": ["Different training-only thresholds are applied in the two models.",
                          "Application records include training records; these are descriptive counts."]
                         + (["Masking this annotation block does not remove correlated information in other features."]
                            if args.variant == "annotation_excluded" else []),
              "validation_or_test_labels_read": False}
    audit_path = directory / "independent_score_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {audit_path}")
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

