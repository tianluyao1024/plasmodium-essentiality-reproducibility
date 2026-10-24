"""Apply the frozen nine-route LightGBM stack to complete malaria proteomes.

The script keeps the original full-application outputs untouched. It validates
the repaired application feature matrices, refits the four homology-scope and
five branch base models on each species' frozen training split, then fits the
LightGBM meta-model on the matching training OOF probabilities.
"""
from __future__ import annotations

import hashlib
import json
import joblib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
CODE = ROOT / "plasmodium-essentiality-reproducibility" / "formal_reproduction" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

DATA_ROOT = Path(r"D:\ProtoEss_feature_matrix_repair_20261015")
FEATURE_ROOT = Path(os.environ.get("PROTOESS_NINE_FEATURE_ROOT", str(DATA_ROOT / "scope_feature_sets_v2")))
APP_ROOT = Path(os.environ.get("PROTOESS_NINE_APP_ROOT", str(DATA_ROOT / "application")))
OOF_ROOT = Path(os.environ.get("PROTOESS_NINE_OOF_ROOT", str(DATA_ROOT / "corrected_raw_foldmatched_meta_20261016")))
SCOPE_PREDICTIONS = Path(os.environ.get(
    "PROTOESS_NINE_SCOPE_PREDICTIONS",
    str(DATA_ROOT / "homology_scope_with_oberstaller_20261016" / "scope_comparison_predictions.tsv"),
))
PROFILE_REPRESENTATIVES = Path(os.environ.get(
    "PROTOESS_NINE_PROFILE_REPRESENTATIVES",
    str(OOF_ROOT / "meta" / "training_cv_profile_ensemble" / "profile_representatives.tsv"),
))
FINAL_METRICS = Path(os.environ.get(
    "PROTOESS_NINE_FINAL_METRICS",
    str(DATA_ROOT / "all_species_nine_route_final_evaluation_20261016" / "all_nine_metrics.tsv"),
))
APP_METADATA = ROOT / "eukaryote_model_datasets" / "genomewide_target_application" / "random_species_label_split_seed42" / "app_metadata.tsv"
MANIFEST_ROOT = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "within_species"
OUTPUT = Path(os.environ.get(
    "PROTOESS_NINE_APPLICATION_OUTPUT",
    str(DATA_ROOT / "all_species_nine_route_full_application_20261016"),
))
APP_SPLIT = ROOT / "eukaryote_model_datasets" / "genomewide_target_application" / "random_species_label_split_seed42"
SPECIES = (
    "Plasmodium_falciparum",
    "Plasmodium_berghei",
    "Plasmodium_knowlesi",
)
SPECIES_ALIASES = {
    "Plasmodium_falciparum_3D7": "Plasmodium_falciparum",
    "Plasmodium_berghei_ANKA": "Plasmodium_berghei",
    "Plasmodium_knowlesi_H": "Plasmodium_knowlesi",
}
SCOPE_NAMES = (
    "strict_within_species",
    "plasmodium_only_label_assisted",
    "cross_species_label_assisted",
    "cross_species_with_pk_oberstaller_labels",
)
BRANCH_NAMES = ("lgbm", "catboost", "species_expert_lgbm", "species_prior", "transformer")
ROUTE_NAMES = SCOPE_NAMES + BRANCH_NAMES
PROFILE_NAME = "combined_nine_base_probabilities"

os.environ["PROTOESS_SCOPE_FEATURE_ROOT"] = str(FEATURE_ROOT)
os.environ["PROTOESS_SOURCE_SCOPE_FEATURE_ROOT"] = str(ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2")

import build_repaired_application_matrices as app_builder  # noqa: E402
import compare_homology_label_scopes as scopes  # noqa: E402
import train_scope_strict_oof_stack as strict  # noqa: E402
import train_scope_transformer_oof as transformer  # noqa: E402
import tune_oof_probability_lightgbm_meta_20261016 as meta  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training(species: str):
    frames = {
        split: pd.read_csv(MANIFEST_ROOT / species / f"{split}_metadata.tsv", sep="\t", dtype={"record_id": str})
        for split in ("train", "val", "test")
    }
    ids = {split: frame.record_id.astype(str).to_numpy() for split, frame in frames.items()}
    y_train = frames["train"].label.eq("essential").astype(np.int8).to_numpy()
    matrix_dir = FEATURE_ROOT / "within_species" / species
    arrays: dict[str, np.ndarray] = {}
    for split in frames:
        matrix_ids = np.load(matrix_dir / f"ids_{split}.npy", allow_pickle=True).astype(str)
        if not np.array_equal(matrix_ids, ids[split]):
            raise ValueError(f"{species}/{split}: matrix IDs differ from the frozen manifest")
        values = np.asarray(np.load(matrix_dir / f"X_{split}.npy", mmap_mode="r"), dtype=np.float32)
        if values.shape != (len(ids[split]), 628) or not np.isfinite(values).all():
            raise ValueError(f"{species}/{split}: expected finite raw 628-column matrix")
        arrays[split] = values
    feature_names = json.loads((matrix_dir / "feature_names.json").read_text(encoding="utf-8"))
    if len(feature_names) != 628:
        raise ValueError(f"{species}: unexpected training feature schema")
    return frames, ids, y_train, matrix_dir, arrays, feature_names


def load_application(species: str, feature_names: list[str], app_metadata: pd.DataFrame):
    canonical = app_metadata.species.astype(str).map(lambda value: SPECIES_ALIASES.get(value, value))
    query = app_metadata.loc[canonical.eq(species)].copy().reset_index(drop=True)
    query["species"] = species
    app_dir = APP_ROOT / species
    ids = np.load(app_dir / "ids_app.npy", allow_pickle=True).astype(str)
    values = np.asarray(np.load(app_dir / "X_app.npy", mmap_mode="r"), dtype=np.float32)
    app_feature_names = json.loads((app_dir / "feature_names.json").read_text(encoding="utf-8"))
    expected = query.record_id.astype(str).to_numpy()
    if not np.array_equal(ids, expected):
        raise ValueError(f"{species}: application matrix IDs do not align to app_metadata.tsv")
    if app_feature_names != feature_names or values.shape != (len(ids), 628):
        raise ValueError(f"{species}: application feature schema differs from training")
    if not np.isfinite(values).all():
        raise ValueError(f"{species}: non-finite application feature value")
    return query, ids, values.copy(), app_dir


def dynamic_columns(feature_names: list[str]):
    hom_ix = np.asarray([i for i, name in enumerate(feature_names) if name.startswith(("homology30_", "homology50_"))], dtype=np.int64)
    hom_local = np.asarray([j for j, index in enumerate(hom_ix) if int(feature_names[index].rsplit("_", 1)[1]) >= 2], dtype=np.int64)
    mm_ix = np.asarray([i for i, name in enumerate(feature_names) if name.startswith("mmseqs_dynamic_")], dtype=np.int64)
    ppi_ix = np.asarray([i for i, name in enumerate(feature_names) if name.startswith("ppi_dynamic_")], dtype=np.int64)
    if (len(hom_ix), len(hom_local), len(mm_ix), len(ppi_ix)) != (30, 26, 21, 15):
        raise ValueError("Unexpected homology/MMseqs/PPI feature layout")
    return hom_ix, hom_local, hom_ix[hom_local], mm_ix, ppi_ix


def read_application_hits(species: str, app_ids: np.ndarray) -> pd.DataFrame:
    path = APP_ROOT / "search_cache" / f"{species}_app_vs_train.tsv"
    hits = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"],
        dtype={"query": str, "target": str},
    )
    if not set(hits["query"].astype(str)).issubset(set(app_ids.astype(str))):
        raise ValueError(f"{species}: application MMseqs cache contains unexpected query IDs")
    hits["identity"] = hits.pident.astype(float) / 100.0
    hits["coverage"] = hits.alnlen.astype(float) / hits[["qlen", "tlen"]].astype(float).max(axis=1)
    hits["score"] = hits.identity * hits.coverage
    return hits


def verify_application_dynamic_features(species: str, app_ids: np.ndarray, app_values: np.ndarray,
                                        feature_names: list[str], train: pd.DataFrame,
                                        y_train: np.ndarray, app_dir: Path):
    hom_ix, _, hom_label_ix, mm_ix, ppi_ix = dynamic_columns(feature_names)
    query_meta = pd.DataFrame({"record_id": app_ids, "species": species})
    overrides = app_builder.load_app_cluster_assignments(app_ids, species, APP_ROOT / "search_cache")
    raw_train = scopes.raw_homology_features(train[["record_id", "species"]], train)
    raw_app = scopes.raw_homology_features(query_meta, train, overrides)
    means = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = raw_train.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
    expected_hom = ((raw_app - means) / stds).astype(np.float32)
    hom_error = float(np.max(np.abs(app_values[:, hom_ix] - expected_hom)))
    if hom_error > 2e-5:
        raise ValueError(f"{species}: application homology values mismatch training-only reconstruction ({hom_error})")

    train_ids = train.record_id.astype(str).to_numpy()
    labels = dict(zip(train_ids, y_train.astype(float)))
    train_species = dict(zip(train_ids, train.species.astype(str)))
    query_species = dict(zip(app_ids.astype(str), [species] * len(app_ids)))
    app_hits = read_application_hits(species, app_ids)
    expected_mm = scopes.aggregate_mm(app_hits, app_ids, labels, train_species, query_species)
    mm_error = float(np.max(np.abs(app_values[:, mm_ix] - expected_mm)))
    if mm_error > 2e-5:
        raise ValueError(f"{species}: application MMseqs values mismatch training-only reconstruction ({mm_error})")

    tax = app_builder.SPECIES_TAX[species][1]
    app_map = app_builder.map_rows(app_builder.STRING_APP_RUN / f"app_map_{tax}.tsv")
    expected_ppi = app_builder.app_ppi_dynamic(app_ids, app_map, train_ids, y_train, species)
    ppi_error = float(np.max(np.abs(app_values[:, ppi_ix] - expected_ppi)))
    if ppi_error > 2e-5:
        raise ValueError(f"{species}: application PPI values mismatch training-only reconstruction ({ppi_error})")
    return {"max_abs_error_homology_label_columns": hom_error,
            "max_abs_error_mmseqs_dynamic": mm_error,
            "max_abs_error_ppi_dynamic": ppi_error}


def load_allowed_pools(all_train: pd.DataFrame, all_train_ober: pd.DataFrame, train: pd.DataFrame):
    return (
        train,
        all_train.loc[all_train.species.astype(str).str.startswith("Plasmodium_")].copy(),
        all_train,
        all_train_ober,
    )


def predict_scope_routes(species: str, frames: dict[str, pd.DataFrame], ids: dict[str, np.ndarray],
                         y_train: np.ndarray, arrays: dict[str, np.ndarray], app_values: np.ndarray,
                         app_ids: np.ndarray, feature_names: list[str], all_train: pd.DataFrame,
                         all_train_ober: pd.DataFrame, app_overrides: dict[str, dict[str, str]],
                         artifact_dir: Path):
    train = frames["train"].copy()
    train_ids = ids["train"]
    train_y_map = dict(zip(train_ids, y_train.astype(float)))
    train_sp = dict(zip(train_ids, train.species.astype(str)))
    query_train = train[["record_id", "species"]]
    query_app = pd.DataFrame({"record_id": app_ids, "species": species})
    hom_ix, hom_local, hom_label_ix, mm_ix, ppi_ix = dynamic_columns(feature_names)
    train_hits = scopes.read_hits(species, "train")
    train_edges = scopes.read_target_species_edges(species, set(train_ids))
    mm_train = scopes.aggregate_mm(train_hits, train_ids, train_y_map, train_sp, train_sp)
    ppi_train = scopes.aggregate_ppi(train_ids, train_y_map, train_edges)

    route_values = []
    route_dir = artifact_dir / "scope_routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    for route_name, allowed in zip(SCOPE_NAMES, load_allowed_pools(all_train, all_train_ober, train)):
        raw_train = scopes.raw_homology_features(query_train, allowed)
        raw_app = scopes.raw_homology_features(query_app, allowed, app_overrides)
        mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = raw_train.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
        h_train = ((raw_train - mean) / std).astype(np.float32)
        h_app = ((raw_app - mean) / std).astype(np.float32)
        X_train = arrays["train"].copy()
        X_app = app_values.copy()
        X_train[:, hom_label_ix] = h_train[:, hom_local]
        X_app[:, hom_label_ix] = h_app[:, hom_local]
        X_train[:, mm_ix] = mm_train
        X_train[:, ppi_ix] = ppi_train
        model = scopes.make_model(42)
        model.fit(X_train, y_train)
        joblib.dump(model, route_dir / f"{route_name}.joblib")
        model.booster_.save_model(str(route_dir / f"{route_name}.txt"))
        np.savez_compressed(route_dir / f"{route_name}_homology_scaler.npz", mean=mean, std=std)
        route_values.append(model.predict_proba(X_app)[:, 1].astype(np.float32))
        del X_train, X_app, model
    return np.column_stack(route_values).astype(np.float32)


def predict_five_branches(species: str, frames: dict[str, pd.DataFrame], ids: dict[str, np.ndarray],
                          y_train: np.ndarray, arrays: dict[str, np.ndarray], app_values: np.ndarray,
                          app_ids: np.ndarray, feature_names: list[str], artifact_dir: Path):
    train_ids = ids["train"]
    train_species = frames["train"].species.astype(str).to_numpy()
    app_species = np.repeat(species, len(app_ids))
    train_hits = scopes.read_hits(species, "train")
    train_edges = scopes.read_target_species_edges(species, set(train_ids))
    k_species = int(arrays["train"].shape[1] - 582 - 9 - 21 - 15)
    static_cols = 582 + 9 + k_species
    X_train = strict.fold_features(
        arrays["train"].copy(), frames, train_hits, train_edges,
        set(train_ids), train_ids, static_cols,
    )
    X_fit, X_app, _, _ = strict.fit_profile_matrix(
        X_train, app_values, feature_names, profile=None, seed=42,
    )
    model_lgbm = strict.lgbm(X_fit, y_train, 42)
    model_cat = strict.cat(X_fit, y_train, 200)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_lgbm, artifact_dir / "lgbm.joblib")
    model_lgbm.booster_.save_model(str(artifact_dir / "lgbm.txt"))
    model_cat.save_model(str(artifact_dir / "catboost.cbm"))
    p_lgbm = model_lgbm.predict_proba(X_app)[:, 1].astype(np.float32)
    p_cat = model_cat.predict_proba(X_app)[:, 1].astype(np.float32)

    p_expert = p_lgbm.copy()
    p_prior = np.zeros(len(app_ids), dtype=np.float32)
    for group in sorted(np.unique(train_species)):
        train_local = np.flatnonzero(train_species == group)
        app_local = np.flatnonzero(app_species == group)
        if not len(app_local):
            continue
        if len(train_local) >= 30 and np.unique(y_train[train_local]).size == 2:
            expert = strict.lgbm(X_fit[train_local], y_train[train_local], 342)
            joblib.dump(expert, artifact_dir / f"species_expert_{group}.joblib")
            expert.booster_.save_model(str(artifact_dir / f"species_expert_{group}.txt"))
            p_expert[app_local] = expert.predict_proba(X_app[app_local])[:, 1]
        p_prior[app_local] = float(y_train[train_local].mean())
    (artifact_dir / "species_prevalence_prior.json").write_text(json.dumps({
        str(group): float(y_train[train_species == group].mean())
        for group in sorted(np.unique(train_species))
    }, indent=2), encoding="utf-8")

    fit_groups, app_groups, group_names = transformer.fit_profile_groups(
        X_train, app_values, feature_names, profile=None, seed=42,
    )
    train_modalities = transformer.transformer_modalities(fit_groups, group_names)
    app_modalities = transformer.transformer_modalities(app_groups, group_names)
    _, p_transformer, best_epoch = transformer.train_predict(
        train_modalities, y_train, app_modalities, None, app_modalities, 42,
        artifact_path=artifact_dir / "transformer.pt",
    )
    branches = np.column_stack([p_lgbm, p_cat, p_expert, p_prior, p_transformer]).astype(np.float32)
    return branches, int(best_epoch)


def load_oof_meta_inputs(species: str, ids: dict[str, np.ndarray], y_train: np.ndarray,
                         scope_frame: pd.DataFrame, candidate_name: str):
    oof_ids, oof_y, scopes_oof, scope_names = meta.align_scope(scope_frame, species, "train")
    if not np.array_equal(oof_ids, ids["train"]) or not np.array_equal(oof_y, y_train):
        raise ValueError(f"{species}: scope OOF rows do not align to the frozen train split")
    if tuple(scope_names) != SCOPE_NAMES:
        raise ValueError(f"{species}: expected the frozen four homology scopes, found {scope_names}")
    branch_path = OOF_ROOT / "five_branch" / species / "oof_branches.npy"
    branches_oof = np.asarray(np.load(branch_path, mmap_mode="r"), dtype=np.float32)
    if branches_oof.shape != (len(oof_ids), len(BRANCH_NAMES)) or not np.isfinite(branches_oof).all():
        raise ValueError(f"{species}: invalid five-branch training OOF matrix")
    candidate = next((item for item in meta.CANDIDATES if item["name"] == candidate_name), None)
    if candidate is None:
        raise ValueError(f"{species}: candidate {candidate_name} is not in the frozen LightGBM candidate set")
    return np.column_stack([scopes_oof, branches_oof]).astype(np.float32), branch_path, candidate


def score_species(species: str, app_metadata: pd.DataFrame, all_train: pd.DataFrame,
                  all_train_ober: pd.DataFrame, scope_frame: pd.DataFrame,
                  profile_representatives: pd.DataFrame, metric_rows: pd.DataFrame):
    frames, ids, y_train, matrix_dir, arrays, feature_names = load_training(species)
    query, app_ids, app_values, app_dir = load_application(species, feature_names, app_metadata)
    validation = verify_application_dynamic_features(
        species, app_ids, app_values, feature_names, frames["train"], y_train, app_dir,
    )
    overrides = app_builder.load_app_cluster_assignments(app_ids, species, APP_ROOT / "search_cache")
    model_dir = OUTPUT / "models" / species

    scope_probabilities = predict_scope_routes(
        species, frames, ids, y_train, arrays, app_values, app_ids, feature_names,
        all_train, all_train_ober, overrides, model_dir / "base_routes",
    )
    branch_probabilities, transformer_best_epoch = predict_five_branches(
        species, frames, ids, y_train, arrays, app_values, app_ids, feature_names,
        model_dir / "base_routes" / "five_branches",
    )
    if scope_probabilities.shape != (len(app_ids), 4) or branch_probabilities.shape != (len(app_ids), 5):
        raise ValueError(f"{species}: nine-route probability shape mismatch")
    base_probabilities = np.column_stack([scope_probabilities, branch_probabilities]).astype(np.float32)
    if not np.isfinite(base_probabilities).all() or (base_probabilities < 0).any() or (base_probabilities > 1).any():
        raise ValueError(f"{species}: invalid base probability values")

    oof_matrix, branch_oof_path, candidate = load_oof_meta_inputs(
        species, ids, y_train, scope_frame,
        str(profile_representatives.loc[
            (profile_representatives.species == species)
            & (profile_representatives.profile == PROFILE_NAME), "candidate"
        ].iloc[0]),
    )
    model = meta.make_model(candidate, 42)
    model.fit(oof_matrix, y_train)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "nine_route_meta_lightgbm.joblib")
    model.booster_.save_model(str(model_dir / "nine_route_meta_lightgbm.txt"))
    (model_dir / "model_config.json").write_text(json.dumps({
        "species": species,
        "profile": PROFILE_NAME,
        "input_order": list(ROUTE_NAMES),
        "candidate": candidate,
        "threshold_from_training_cv": float(metric_rows.loc[
            (metric_rows.species == species) & (metric_rows.split == "test"),
            "threshold_from_nine_route_training_CV",
        ].iloc[0]),
    }, indent=2), encoding="utf-8")
    score = model.predict_proba(base_probabilities)[:, 1].astype(np.float32)

    threshold_rows = metric_rows.loc[
        (metric_rows.species == species) & (metric_rows.split == "test"),
        "threshold_from_nine_route_training_CV",
    ]
    if len(threshold_rows) != 1:
        raise ValueError(f"{species}: expected one frozen nine-route training-CV threshold")
    threshold = float(threshold_rows.iloc[0])
    result = query[["record_id", "species", "gene_id", "protein_id", "protein_length"]].copy()
    result["species"] = species
    for index, name in enumerate(ROUTE_NAMES):
        result[f"route_{name}_probability"] = base_probabilities[:, index]
    result["nine_route_lightgbm_score"] = score
    result["training_cv_threshold"] = threshold
    result["score_margin"] = score - threshold
    result["predicted_essential_at_training_cv_threshold"] = score >= threshold
    if result.record_id.astype(str).duplicated().any() or not np.isfinite(score).all():
        raise ValueError(f"{species}: duplicate application IDs or invalid final scores")

    species_dir = OUTPUT / "genomewide_predictions"
    species_dir.mkdir(parents=True, exist_ok=True)
    path = species_dir / f"{species}_nine_route.tsv"
    temporary = path.with_suffix(".tsv.tmp")
    result.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)
    summary = {
        "species": species,
        "n_application_genes": int(len(result)),
        "n_predicted_essential_at_training_cv_threshold": int(result.predicted_essential_at_training_cv_threshold.sum()),
        "threshold_from_training_cv": threshold,
        "meta_candidate_from_training_cv_profile_representatives": candidate["name"],
        "meta_model_artifacts": str(model_dir),
        "base_model_artifacts": str(model_dir / "base_routes"),
        "transformer_epoch_selected_with_internal_training_holdout": transformer_best_epoch,
        "feature_validation": validation,
        "application_matrix_sha256": sha256(app_dir / "X_app.npy"),
        "application_ids_sha256": sha256(app_dir / "ids_app.npy"),
        "training_matrix_sha256": sha256(matrix_dir / "X_train.npy"),
        "scope_oof_sha256": sha256(SCOPE_PREDICTIONS),
        "branch_oof_sha256": sha256(branch_oof_path),
    }
    print(f"{species}: scored {len(result)} genes; threshold={threshold:.3f}; predicted essential={summary['n_predicted_essential_at_training_cv_threshold']}", flush=True)
    return result, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app_metadata = pd.read_csv(APP_METADATA, sep="\t", dtype={"record_id": str, "species": str}).fillna("")
    scope_frame = pd.read_csv(SCOPE_PREDICTIONS, sep="\t", dtype={"record_id": str})
    representatives = pd.read_csv(PROFILE_REPRESENTATIVES, sep="\t")
    representatives = representatives.loc[representatives.profile.eq(PROFILE_NAME)].copy()
    metrics = pd.read_csv(FINAL_METRICS, sep="\t")
    if "threshold_from_nine_route_training_CV" not in metrics.columns and "threshold_from_training_CV" in metrics.columns:
        metrics = metrics.rename(columns={"threshold_from_training_CV": "threshold_from_nine_route_training_CV"})
    if set(representatives.species.astype(str)) != set(SPECIES):
        raise ValueError("Training-CV representative table does not contain a nine-route choice for all species")
    all_train = scopes.read_all_training_metadata()
    all_train_ober, ober_stats = scopes.build_oberstaller_training_pool(all_train)

    all_results = []
    species_summaries = []
    for species in SPECIES:
        result, summary = score_species(
            species, app_metadata, all_train, all_train_ober, scope_frame,
            representatives, metrics,
        )
        all_results.append(result)
        species_summaries.append(summary)

    combined = pd.concat(all_results, ignore_index=True)
    expected_counts = {"Plasmodium_falciparum": 5318, "Plasmodium_berghei": 4945, "Plasmodium_knowlesi": 5328}
    observed = combined.groupby("species").size().to_dict()
    if observed != expected_counts or combined.record_id.astype(str).duplicated().any():
        raise ValueError(f"Unexpected whole-proteome application coverage: {observed}")
    combined_path = OUTPUT / "all_three_species_nine_route_genomewide.tsv"
    tmp_path = combined_path.with_suffix(".tsv.tmp")
    combined.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(combined_path)
    pd.DataFrame(species_summaries).to_csv(OUTPUT / "application_summary.tsv", sep="\t", index=False)
    manifest = {
        "purpose": "Full-proteome application of the selected nine-route LightGBM model for all three Plasmodium species.",
        "routes_order": list(ROUTE_NAMES),
        "feature_count": 628,
        "feature_values": "original repaired raw 628-column matrices; no feature deletion, PCA, or feature selection",
        "model_selection": "per-species combined_nine_base_probabilities candidate from training-only profile representatives",
        "training_data": "frozen species-specific 95%-deduplicated, label-stratified random train split; meta-model fit on matching 4-scope plus 5-branch OOF probabilities",
        "application_dynamic_features": "validated against training-label-only homology, MMseqs and STRING/PPI reconstruction; homology application queries use genomewide cluster overrides",
        "threshold": "training-only repeated-CV threshold carried from the all-species nine-route final evaluation; used for flags and margins, not score fitting",
        "test_labels_used": False,
        "Oberstaller_source": "definite P. knowlesi calls from the independent human-RBC screen replace only matched P. knowlesi training-source labels in route 4; no application labels are used",
        "historic_test_caveat": "Earlier test access and nine-route selection after inspection mean reported test scores are post-hoc, not an untouched blind estimate.",
        "meta_cv_caveat": "The existing meta-model candidate selection uses precomputed base OOF probabilities and is not fully nested.",
        "input_hashes": {
            "application_metadata": sha256(APP_METADATA),
            "scope_training_oof": sha256(SCOPE_PREDICTIONS),
            "training_cv_profile_representatives": sha256(PROFILE_REPRESENTATIVES),
            "nine_route_final_metrics_thresholds": sha256(FINAL_METRICS),
            "Oberstaller_aligned_calls": sha256(scopes.OBERST_CALLS),
        },
        "species": species_summaries,
        "total_application_genes": int(len(combined)),
        "output_file": str(combined_path),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "DONE").write_text("complete\n", encoding="utf-8")
    print(f"Saved {len(combined)} nine-route genomewide predictions to {combined_path}", flush=True)


if __name__ == "__main__":
    main()

