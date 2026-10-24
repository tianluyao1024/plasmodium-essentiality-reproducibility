"""Package training-only nine-route OOF inputs for reproducible meta fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
SCOPES = (
    "strict_within_species",
    "plasmodium_only_label_assisted",
    "cross_species_label_assisted",
    "cross_species_with_pk_oberstaller_labels",
)
BRANCHES = ("lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scope-predictions", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = args.release_dir.resolve() / "source_data" / "Meta_Training_OOF"
    out.mkdir(parents=True, exist_ok=True)
    scope_frame = pd.read_csv(args.scope_predictions, sep="\t", dtype={"record_id": str})
    rows: list[pd.DataFrame] = []
    provenance: dict[str, dict[str, object]] = {}

    for species in SPECIES:
        train = pd.read_csv(args.manifest_root / species / "train_metadata.tsv", sep="\t", dtype={"record_id": str})
        ids = train.record_id.astype(str).to_numpy()
        labels = train.label.eq("essential").astype(np.int8).to_numpy()
        scoped = scope_frame.loc[(scope_frame.species == species) & (scope_frame.split == "train_oof")]
        columns: list[np.ndarray] = []
        for scope in SCOPES:
            part = scoped.loc[scoped.scope == scope, ["record_id", "label", "probability"]].copy()
            if part.empty or part.record_id.duplicated().any():
                raise ValueError(f"{species}/{scope}: missing or duplicate OOF rows")
            part = part.set_index("record_id").reindex(ids)
            if part.probability.isna().any() or not np.array_equal(part.label.astype(int).to_numpy(), labels):
                raise ValueError(f"{species}/{scope}: train ID/label mismatch")
            columns.append(part.probability.to_numpy(dtype=np.float32))

        branch_path = root / "five_branch" / species / "oof_branches.npy"
        branches = np.asarray(np.load(branch_path), dtype=np.float32)
        if branches.shape != (len(ids), len(BRANCHES)):
            raise ValueError(f"{species}: unexpected branch OOF shape {branches.shape}")
        matrix = np.column_stack([*columns, branches]).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{species}: non-finite OOF probabilities")
        frame = pd.DataFrame(matrix, columns=[*SCOPES, *BRANCHES])
        frame.insert(0, "label", labels)
        frame.insert(0, "record_id", ids)
        frame.insert(0, "species", species)
        rows.append(frame)
        metadata_path = args.manifest_root / species / "train_metadata.tsv"
        provenance[species] = {
            "n_train": len(ids),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "source_train_metadata_sha256": sha256(metadata_path),
            "source_branch_oof_sha256": sha256(branch_path),
        }

    output = out / "training_meta_inputs.tsv"
    pd.concat(rows, ignore_index=True).to_csv(output, sep="\t", index=False, float_format="%.9g")
    manifest = {
        "status": "training_only_meta_inputs_packaged",
        "model_version": "fivefold_nine_route_lightgbm_frozen_2026-10-17_internal_v10",
        "rows": sum(item["n_train"] for item in provenance.values()),
        "input_count": 9,
        "input_order": [*SCOPES, *BRANCHES],
        "feature_values": "base-model training OOF probabilities; float32 round-trip",
        "validation_or_test_labels_included": False,
        "validation_or_test_rows_included": False,
        "source_scope_oof_sha256": sha256(args.scope_predictions),
        "species": provenance,
        "output_sha256": sha256(output),
        "limitation": "These compact OOF inputs support refitting the nine-route meta learner only. They do not regenerate the base routes or their fold-local features from the 628-dimensional matrices.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": manifest["rows"], "sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
