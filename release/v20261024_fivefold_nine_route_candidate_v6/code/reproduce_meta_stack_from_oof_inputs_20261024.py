"""Refit the frozen nine-route LightGBM meta models from training-only OOF inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SPECIES = ("Plasmodium_falciparum", "Plasmodium_berghei", "Plasmodium_knowlesi")
INPUTS = (
    "strict_within_species", "plasmodium_only_label_assisted",
    "cross_species_label_assisted", "cross_species_with_pk_oberstaller_labels",
    "lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer",
)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5601)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    release, output = args.release_dir.resolve(), args.output_dir.resolve()
    train_path = release / "source_data" / "Meta_Training_OOF" / "training_meta_inputs.tsv"
    test_path = release / "source_data" / "Frozen_Test_Predictions" / "frozen_test_predictions.tsv"
    model_inputs_path = release / "source_data" / "Frozen_Test_Predictions" / "meta_model_replay" / "model_inputs_validation_test.tsv"
    train = pd.read_csv(train_path, sep="\t", dtype={"record_id": str})
    test_labels = pd.read_csv(test_path, sep="\t", dtype={"record_id": str})
    inference = pd.read_csv(model_inputs_path, sep="\t", dtype={"record_id": str})
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    configurations: dict[str, dict] = {}

    for species in SPECIES:
        cfg_path = release / "source_data" / "Frozen_Test_Predictions" / "meta_model_replay" / "models" / f"{species}.json"
        cfg = read_json(cfg_path)
        params = cfg["candidate_parameters"].copy()
        params.pop("name", None)
        frame = train.loc[train.species == species]
        X = frame.loc[:, INPUTS].astype(np.float32)
        y = frame.label.to_numpy(dtype=np.int8)
        if len(frame) != cfg["n_train"] or len(np.unique(y)) != 2:
            raise ValueError(f"{species}: train rows/labels mismatch")

        repeat_prob = np.zeros((3, len(y)), dtype=np.float32)
        for fold, (fit_idx, hold_idx) in enumerate(CV.split(X, y)):
            model = lgb.LGBMClassifier(
                objective="binary", class_weight="balanced", random_state=42 + fold,
                n_jobs=4, verbosity=-1, subsample=0.9, subsample_freq=1,
                colsample_bytree=1.0, reg_alpha=0.1, **params,
            )
            model.fit(X.iloc[fit_idx], y[fit_idx])
            repeat_prob[fold // 5, hold_idx] = model.predict_proba(X.iloc[hold_idx])[:, 1].astype(np.float32)
        grid = np.linspace(0.05, 0.95, 181)
        mean_oof = repeat_prob.mean(axis=0)
        threshold = float(grid[int(np.argmax([matthews_corrcoef(y, mean_oof >= t) for t in grid]))])
        if abs(threshold - float(cfg["threshold_from_training_only_repeated_cv"])) > 1e-12:
            raise ValueError(f"{species}: recomputed threshold differs from frozen config")

        final_model = lgb.LGBMClassifier(
            objective="binary", class_weight="balanced", random_state=42,
            n_jobs=4, verbosity=-1, subsample=0.9, subsample_freq=1,
            colsample_bytree=1.0, reg_alpha=0.1, **params,
        ).fit(X, y)
        pred = inference.loc[inference.species == species].copy()
        pred["probability"] = final_model.predict_proba(pred.loc[:, INPUTS].astype(np.float32))[:, 1]
        prediction_rows.append(pred[["species", "split", "record_id", "probability"]])
        test_pred = pred.loc[pred.split == "test"].merge(
            test_labels.loc[test_labels.species == species, ["species", "record_id", "label"]],
            on=["species", "record_id"], validate="one_to_one",
        )
        metric_rows.append({
            "species": species, "n_test": len(test_pred),
            "AUROC": roc_auc_score(test_pred.label, test_pred.probability),
            "AUPRC": average_precision_score(test_pred.label, test_pred.probability),
            "threshold": threshold, "candidate": cfg["selected_candidate"],
        })
        configurations[species] = {"candidate": cfg["selected_candidate"], "threshold": threshold, "n_train": len(frame)}

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(output / "validation_test_predictions.tsv", sep="\t", index=False, float_format="%.17g")
    metrics.to_csv(output / "test_metrics.tsv", sep="\t", index=False, float_format="%.17g")
    frozen = test_labels.merge(predictions.loc[predictions.split == "test"], on=["species", "record_id"], validate="one_to_one", suffixes=("_frozen", "_reproduced"))
    max_diff = float((frozen.probability_frozen - frozen.probability_reproduced).abs().max())
    if max_diff > 1e-12:
        raise ValueError(f"reproduced test probabilities differ from frozen values (max abs diff={max_diff})")
    report = {
        "status": "passed",
        "scope": "meta-model refit from packaged training-only base OOF probabilities; base routes are not refit",
        "training_input_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_prediction_rows": len(frozen),
        "max_abs_test_probability_difference_from_frozen": max_diff,
        "test_metrics_recomputed": metrics.to_dict(orient="records"),
        "models": configurations,
        "validation_test_labels_used_for_fit_or_threshold": False,
        "historical_test_exposure": "Metrics remain retrospective; prior project iterations inspected held-out Pk outcomes.",
    }
    (output / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

