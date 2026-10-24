"""Check that a resampling pilot used matched records and five-fold partitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_nine_route_application_refit_20261017 import sha256


ROOT = Path(r"D:\ProtoEss_original_structure_revision_20261017\nine_route_refit_pilot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--fraction", type=float, default=.8)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = ROOT / f"fraction_{args.fraction:.3f}_repeat_{args.repeat:03d}_seed_{args.seed}" / args.species
    oof, scores, checks = {}, {}, {}
    for variant in ("full", "annotation_excluded"):
        variant_root = root / variant
        oof_dir = variant_root / "frozen_unlabelled"
        meta = json.loads((oof_dir / "oof_manifest.json").read_text(encoding="utf-8"))
        ids_path, oof_path = oof_dir / "selected_ids.npy", oof_dir / "oof_probabilities.npy"
        if sha256(ids_path) != meta["selected_ids_sha256"] or sha256(oof_path) != meta["oof_sha256"]:
            raise ValueError(f"Invalid OOF hashes for {variant}")
        oof[variant] = {"ids": np.load(ids_path, allow_pickle=False),
                        "folds": np.load(oof_dir / "fold_assignments.npy", allow_pickle=False), "meta": meta}
        app_dir = variant_root / "application"
        app = app_dir / "application_predictions.tsv"
        audit = json.loads((app_dir / "independent_score_audit.json").read_text(encoding="utf-8"))
        if audit["prediction_sha256"] != sha256(app):
            raise ValueError(f"Invalid application hash for {variant}")
        scores[variant] = pd.read_csv(app, sep="\t", dtype={"record_id": str})
        if scores[variant].record_id.duplicated().any():
            raise ValueError(f"Duplicate application IDs for {variant}")
        checks[variant] = {"oof_sha256": meta["oof_sha256"], "prediction_sha256": audit["prediction_sha256"]}
    left, right = oof["full"], oof["annotation_excluded"]
    if (not np.array_equal(left["ids"], right["ids"]) or not np.array_equal(left["folds"], right["folds"])
            or left["meta"]["training_labels_source_sha256"] != right["meta"]["training_labels_source_sha256"]):
        raise ValueError("The two variants do not use the same training subset/folds/label source")
    if len(np.unique(left["folds"])) != 5:
        raise ValueError("Expected five folds")
    paired = scores["full"].merge(scores["annotation_excluded"], on="record_id", validate="one_to_one",
                                  suffixes=("_full", "_excluded"))
    if len(paired) != len(scores["full"]) or len(paired) != len(scores["annotation_excluded"]):
        raise ValueError("Application record sets differ")
    result = {"status": "one paired species refit; not a 100-repeat stability result", "species": args.species,
              "fraction": args.fraction, "repeat": args.repeat, "seed": args.seed,
              "training_records": len(left["ids"]), "same_selected_training_ids": True, "same_fivefold_assignment": True,
              "application_records": len(paired), "application_calls_disagree": int(
                  paired.predicted_essential_full.ne(paired.predicted_essential_excluded).sum()),
              "artifacts": checks,
              "caveat": "Application classification differences use each variant's training-CV threshold."}
    path = root / "paired_refit_audit.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

