from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SPECIES = (
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
)
EXPECTED_INPUTS = (
    "strict_within_species",
    "plasmodium_only_label_assisted",
    "cross_species_label_assisted",
    "cross_species_with_pk_oberstaller_labels",
    "lgbm",
    "catboost",
    "species_expert_lgbm",
    "species_prior",
    "transformer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(release: Path) -> dict[str, object]:
    freeze = json.loads((release / "atlas" / "final_freeze_manifest.json").read_text(encoding="utf-8"))
    replay = release / "source_data" / "Frozen_Test_Predictions" / "meta_model_replay"
    inputs = pd.read_csv(replay / "model_inputs_validation_test.tsv", sep="\t", dtype={"record_id": str})
    observed = pd.read_csv(
        release / "source_data" / "Frozen_Test_Predictions" / "frozen_test_predictions.tsv",
        sep="\t",
        dtype={"record_id": str},
    )
    output: dict[str, object] = {}

    required = {"species", "split", "record_id", *EXPECTED_INPUTS}
    if not required.issubset(inputs.columns):
        raise ValueError(f"Route-input table is missing columns: {sorted(required - set(inputs.columns))}")
    if inputs.duplicated(["species", "split", "record_id"]).any():
        raise ValueError("Route-input table contains duplicate gene rows")
    if set(inputs["split"]) != {"validation", "test"}:
        raise ValueError("Replay inputs must contain validation and test rows only")

    for species in SPECIES:
        species_inputs = inputs.loc[inputs.species.eq(species)]
        species_dir = replay / "models"
        model_path = species_dir / f"{species}.joblib"
        config_path = species_dir / f"{species}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_hash = freeze["species"][species]["evaluation_meta_model_sha256"]
        if sha256(model_path) != expected_hash:
            raise ValueError(f"{species}: saved model hash differs from the freeze manifest")
        if config["input_order"] != list(EXPECTED_INPUTS):
            raise ValueError(f"{species}: model input order differs from the frozen route order")

        model = joblib.load(model_path)
        species_results: dict[str, object] = {}
        for split in ("validation", "test"):
            rows = species_inputs.loc[species_inputs.split.eq(split)].copy()
            expected_n = int(freeze["species"][species]["partitions"]["val" if split == "validation" else "test"]["n"])
            if len(rows) != expected_n or rows.record_id.duplicated().any():
                raise ValueError(f"{species}/{split}: row count or ID uniqueness mismatch")
            matrix = rows.loc[:, EXPECTED_INPUTS].astype(np.float32)
            if not np.isfinite(matrix.to_numpy()).all():
                raise ValueError(f"{species}/{split}: non-finite route probabilities")
            probabilities = model.predict_proba(matrix)[:, 1]
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError(f"{species}/{split}: invalid meta-model probability")
            species_results[split] = {"n": len(rows), "predicted": True}

            if split == "test":
                reference = observed.loc[observed.species.eq(species)].set_index("record_id")
                if set(reference.index) != set(rows.record_id):
                    raise ValueError(f"{species}: test IDs differ from frozen predictions")
                aligned = reference.loc[rows.record_id]
                max_abs_diff = float(np.max(np.abs(probabilities - aligned.probability.to_numpy())))
                if max_abs_diff > 1e-12:
                    raise ValueError(f"{species}: replayed predictions differ by {max_abs_diff}")
                y = aligned.label.to_numpy(dtype=np.int8)
                auroc = float(roc_auc_score(y, probabilities))
                auprc = float(average_precision_score(y, probabilities))
                frozen = freeze["species"][species]
                if abs(auroc - float(frozen["test_auroc"])) > 1e-12:
                    raise ValueError(f"{species}: test AUROC mismatch")
                if abs(auprc - float(frozen["test_auprc"])) > 1e-12:
                    raise ValueError(f"{species}: test AUPRC mismatch")
                species_results["test"].update(
                    {"max_abs_probability_difference": max_abs_diff, "auroc": auroc, "auprc": auprc}
                )
        output[species] = species_results

    return {
        "status": "passed",
        "scope": "meta-model inference replay from supplied nine-route probabilities; not model refitting",
        "input_rows": int(len(inputs)),
        "species": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the frozen nine-route LightGBM meta-model scores.")
    parser.add_argument("--release-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.release_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
