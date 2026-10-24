from __future__ import annotations

import gzip
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from build_strict_oof_dynamic_stack import aggregate_mm
from protoess_safe_homology import raw_homology_features


ROOT = Path.cwd()
DATA = ROOT / "eukaryote_model_datasets"
ALIGN = DATA / "features" / "unified_v3_aligned_features"
APP_SPLIT = DATA / "genomewide_target_application" / "random_species_label_split_seed42"
APP_FEATURES = DATA / "features" / "genomewide_target_application"
STRING = ROOT / "downloads" / "string_v12"
STRING_APP_RUN = Path(os.environ.get(
    "PROTOESS_STRING_APPLICATION_MAP_ROOT",
    r"D:\eukaryote_model_runs_go_ipr_pfam\genomewide_target_application\final_application_features",
))
MATRIX_ROOT = Path(os.environ.get(
    "PROTOESS_REPAIRED_SCOPE_FEATURE_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\scope_feature_sets_v2",
))
OUT = Path(os.environ.get(
    "PROTOESS_REPAIRED_APP_ROOT",
    r"D:\ProtoEss_feature_matrix_repair_20261015\application",
))
MMSEQS = ROOT / "tools" / "mmseqs2" / "mmseqs" / "bin" / "mmseqs.exe"
SPECIES_TAX = {
    "Plasmodium_berghei": ("Plasmodium_berghei_ANKA", 5823),
    "Plasmodium_falciparum": ("Plasmodium_falciparum_3D7", 36329),
    "Plasmodium_knowlesi": ("Plasmodium_knowlesi_H", 5851),
}
GROUPS = ("esm2", "prott5", "basic_seq", "go_ipr_pfam", "kmer_graph", "homology30", "homology50", "string_network")
STRING_NAMES = (
    "string_mapped", "string_map_identity", "string_map_coverage", "degree_score400",
    "degree_score700", "degree_score900", "weighted_degree", "mean_edge_score", "max_edge_score",
)


def read_app_metadata() -> pd.DataFrame:
    return pd.read_csv(APP_SPLIT / "app_metadata.tsv", sep="\t", dtype={"record_id": str, "species": str}).fillna("")


def load_raw_app(group: str, app_ids: np.ndarray) -> np.ndarray:
    directory = APP_FEATURES / group
    filename = "app_basic.npy" if group == "basic_seq" else "app_emb.npy"
    values = np.asarray(np.load(directory / filename, mmap_mode="r"), dtype=np.float32)
    ids = np.load(directory / "app_ids.npy", allow_pickle=True).astype(str)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid application ID order for {group}")
    position = {rid: i for i, rid in enumerate(ids)}
    missing = [rid for rid in app_ids if rid not in position]
    if missing:
        raise ValueError(f"{group} lacks {len(missing)} application IDs")
    result = values[np.asarray([position[rid] for rid in app_ids], dtype=np.int64)].copy()
    if group == "kmer_graph":
        result = result[:, :512]
    return result


def aligned_group(group: str, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.load(ALIGN / group / "train_emb_aligned.npy", mmap_mode="r"), dtype=np.float32)
    present = np.asarray(np.load(ALIGN / group / "train_present.npy", mmap_mode="r"), dtype=bool)
    master = np.load(ALIGN / "master_train_ids.npy", allow_pickle=True).astype(str)
    position = {rid: i for i, rid in enumerate(master)}
    idx = np.asarray([position[rid] for rid in train_ids], dtype=np.int64)
    return values[idx].copy(), present[idx].copy()


def map_rows(path: Path) -> pd.DataFrame:
    cols = ["record_id", "node", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"]
    table = pd.read_csv(path, sep="\t", header=None, names=cols, dtype={"record_id": str, "node": str})
    table["coverage"] = table.alnlen.astype(float) / table[["qlen", "tlen"]].astype(float).max(axis=1)
    table = table[(table.pident.astype(float) >= 70.0) & (table.coverage >= 0.6)]
    return table.sort_values(["record_id", "bits"], ascending=[True, False]).drop_duplicates("record_id")


def load_app_cluster_assignments(app_ids: np.ndarray, species: str, output: Path) -> dict[str, dict[str, str]]:
    cluster30_table = pd.read_csv(
        APP_FEATURES / "homology30" / "app_cluster_assignments.tsv",
        sep="\t", dtype={"record_id": str, "cluster_id": str},
    )
    cluster30_map = dict(zip(cluster30_table.record_id, cluster30_table.cluster_id))
    missing30 = [rid for rid in app_ids if rid not in cluster30_map]
    if missing30:
        raise ValueError(f"homology30 is missing {len(missing30)} application cluster assignments")

    app_meta = read_app_metadata().set_index("record_id")
    app_fasta = APP_SPLIT / "app.protein.fasta"
    query_fasta = output / f"{species}_queries.fasta"
    selected = set(app_ids.astype(str))
    if not query_fasta.exists():
        with app_fasta.open(encoding="utf-8", errors="ignore") as source, query_fasta.open("w", encoding="utf-8", newline="\n") as target:
            writing = False
            for line in source:
                if line.startswith(">"):
                    rid = line[1:].split()[0]
                    writing = rid in selected
                if writing:
                    target.write(line)
    rep = DATA / "mmseqs50" / "cluster50_rep_seq.fasta"
    hits_path = output / f"{species}_app_vs_cluster50.tsv"
    tmp = output / f"tmp_{species}_cluster50"
    if not hits_path.exists():
        subprocess.run([
            str(MMSEQS), "easy-search", str(query_fasta), str(rep), str(hits_path), str(tmp),
            "--format-output", "query,target,pident,alnlen,evalue,bits,qlen,tlen",
            "--max-seqs", "3", "-s", "7.5", "-e", "0.01", "--min-seq-id", "0.5",
            "-c", "0.5", "--cov-mode", "0", "--threads", "8",
        ], check=True)
    cols = ["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"]
    hits = pd.read_csv(hits_path, sep="\t", header=None, names=cols, dtype={"query": str, "target": str})
    if hits.empty:
        cluster50_map: dict[str, str] = {}
    else:
        hits["qcov"] = hits.alnlen.astype(float) / hits.qlen.astype(float).clip(lower=1)
        hits["tcov"] = hits.alnlen.astype(float) / hits.tlen.astype(float).clip(lower=1)
        hits["score"] = hits.pident.astype(float) / 100.0 * hits[["qcov", "tcov"]].min(axis=1)
        hits = hits.sort_values(["query", "score", "bits"], ascending=[True, False, False]).drop_duplicates("query")
        cluster50_map = dict(zip(hits["query"].astype(str), hits["target"].astype(str)))
    return {
        "homology30": {rid: cluster30_map.get(rid, rid) for rid in app_ids.astype(str)},
        "homology50": {rid: cluster50_map.get(rid, rid) for rid in app_ids.astype(str)},
    }


def app_string_blocks(app_meta: pd.DataFrame, app_ids: np.ndarray, species: str) -> tuple[np.ndarray, np.ndarray]:
    app_taxon, tax = SPECIES_TAX[species]
    app_map = map_rows(STRING_APP_RUN / f"app_map_{tax}.tsv")
    app_map = app_map.set_index("record_id")
    links = STRING / f"{tax}.protein.links.v12.0.txt.gz"
    nodes = set(app_map.loc[app_map.index.isin(app_ids), "node"].astype(str))
    stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with gzip.open(links, "rt", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            left, right, raw_score = line.split()
            if left not in nodes and right not in nodes:
                continue
            score = float(raw_score) / 1000.0
            for node in (left, right):
                if node not in nodes:
                    continue
                row = stats[node]
                row[0] += float(raw_score.isdigit() and int(raw_score) >= 400)
                row[1] += float(raw_score.isdigit() and int(raw_score) >= 700)
                row[2] += float(raw_score.isdigit() and int(raw_score) >= 900)
                row[3] += score
                row[4] += 1.0
                row[5] = max(row[5], score)

    original = np.zeros((len(app_ids), 9), dtype=np.float32)
    structure = np.zeros_like(original)
    aliases = {"Plasmodium_berghei_ANKA": "Plasmodium_berghei", "Plasmodium_falciparum_3D7": "Plasmodium_falciparum", "Plasmodium_knowlesi_H": "Plasmodium_knowlesi"}
    meta_species = dict(zip(app_meta.record_id.astype(str), app_meta.species.astype(str).map(lambda value: aliases.get(value, value))))
    for i, rid in enumerate(app_ids.astype(str)):
        if meta_species.get(rid) != species or rid not in app_map.index:
            continue
        m = app_map.loc[rid]
        node = str(m.node)
        degree = stats.get(node, [0.0] * 6)
        identity = float(m.pident) / 100.0
        coverage = float(m.coverage)
        original[i] = [1.0, identity, coverage, degree[0], degree[1], degree[2], degree[3], degree[3] / max(degree[4], 1.0), degree[5]]
        structure[i] = [1.0, 0.0, 0.0, degree[0], degree[1], degree[2], degree[3], degree[3] / max(degree[4], 1.0), degree[5]]
    return original, structure


def app_ppi_dynamic(app_ids: np.ndarray, app_string_nodes: pd.DataFrame, train_ids: np.ndarray,
                    train_y: np.ndarray, species: str) -> np.ndarray:
    _, tax = SPECIES_TAX[species]
    train_ids = np.asarray(train_ids, dtype=str)
    train_labels = dict(zip(train_ids, np.asarray(train_y, dtype=float)))
    train_map = map_rows(STRING / f"current_map_{tax}.tsv")
    train_map = train_map[train_map.record_id.isin(set(train_ids))]
    node_to_record = dict(zip(train_map.node.astype(str), train_map.record_id.astype(str)))
    node_y = {node: train_labels[rid] for node, rid in node_to_record.items() if rid in train_labels}
    query_map = app_string_nodes[app_string_nodes.record_id.isin(set(app_ids.astype(str)))]
    node_queries: dict[str, list[int]] = defaultdict(list)
    app_position = {rid: i for i, rid in enumerate(app_ids.astype(str))}
    for row in query_map.itertuples(index=False):
        if row.record_id in app_position:
            node_queries[str(row.node)].append(app_position[row.record_id])
    accumulator = np.zeros((len(app_ids), 15), dtype=np.float64)
    with gzip.open(STRING / f"{tax}.protein.links.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            left, right, raw_score = line.split()
            query_rows = node_queries.get(left)
            target_y = node_y.get(right)
            if not query_rows or target_y is None or left == right:
                continue
            score = float(raw_score) / 1000.0
            for j, threshold in enumerate((0.4, 0.7, 0.9)):
                if score < threshold:
                    continue
                for i in query_rows:
                    offset = 5 * j
                    accumulator[i, offset] += 1
                    accumulator[i, offset + 1] += score * target_y
                    accumulator[i, offset + 2] += score
                    accumulator[i, offset + 3] = max(accumulator[i, offset + 3], score * target_y)
                    accumulator[i, offset + 4] = max(accumulator[i, offset + 4], score * (1.0 - target_y))
    output = np.zeros((len(app_ids), 15), dtype=np.float32)
    for i in range(len(app_ids)):
        for j in range(3):
            offset = 5 * j
            n, wy, weight, max_pos, max_neg = accumulator[i, offset:offset + 5]
            output[i, offset:offset + 5] = [np.log1p(n), wy / weight if n else 0.5, max_pos, max_neg, n]
    return output


def fit_transform(group: str, raw_train: np.ndarray, valid: np.ndarray, raw_app: np.ndarray) -> np.ndarray:
    train = np.nan_to_num(raw_train.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.asarray(valid, dtype=bool)
    mean = np.where(valid[:, None], train, 0.0).sum(axis=0) / np.maximum(valid.sum(), 1)
    centered = np.where(valid[:, None], train - mean, 0.0).astype(np.float64)
    std = np.sqrt((centered ** 2).sum(axis=0) / np.maximum(valid.sum(), 1)).astype(np.float32).clip(min=1e-6)
    app = np.nan_to_num(raw_app.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    app = ((app - mean) / std).astype(np.float32)
    if train.shape[1] > 128:
        svd = TruncatedSVD(n_components=128, n_iter=4, random_state=42)
        svd.fit(train)
        app = svd.transform(app).astype(np.float32)
    return app


def search_app_vs_train(species: str, app_ids: np.ndarray, out: Path) -> pd.DataFrame:
    target = ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2" / "within_species" / species / "scope_train.fasta"
    all_fasta = APP_SPLIT / "app.protein.fasta"
    query = out / f"{species}_queries.fasta"
    if not query.exists():
        selected = set(app_ids.astype(str))
        with all_fasta.open(encoding="utf-8", errors="ignore") as source, query.open("w", encoding="utf-8", newline="\n") as target_fa:
            writing = False
            for line in source:
                if line.startswith(">"):
                    writing = line[1:].split()[0] in selected
                if writing:
                    target_fa.write(line)
    result = out / f"{species}_app_vs_train.tsv"
    if not result.exists():
        subprocess.run([
            str(MMSEQS), "easy-search", str(query), str(target), str(result), str(out / f"tmp_{species}_train_hits"),
            "--format-output", "query,target,pident,alnlen,evalue,bits,qlen,tlen",
            "--max-seqs", "100", "-s", "7.5", "-e", "0.01", "--threads", "8",
        ], check=True)
    cols = ["query", "target", "pident", "alnlen", "evalue", "bits", "qlen", "tlen"]
    return pd.read_csv(result, sep="\t", header=None, names=cols, dtype={"query": str, "target": str})


def build_species(species: str, app_meta: pd.DataFrame) -> dict[str, object]:
    output = OUT / "search_cache"
    output.mkdir(parents=True, exist_ok=True)
    split_name, tax = SPECIES_TAX[species]
    alias = app_meta.species.astype(str).map({
        "Plasmodium_berghei_ANKA": "Plasmodium_berghei",
        "Plasmodium_falciparum_3D7": "Plasmodium_falciparum",
        "Plasmodium_knowlesi_H": "Plasmodium_knowlesi",
    }).fillna(app_meta.species.astype(str))
    query_meta = app_meta.loc[alias.eq(species)].copy().reset_index(drop=True)
    query_meta["species"] = species
    app_ids = query_meta.record_id.astype(str).to_numpy()
    frames = {
        part: pd.read_csv(DATA / "unified_v3_experiment_manifests_pre95human" / "within_species" / species / f"{part}_metadata.tsv", sep="\t", dtype={"record_id": str})
        for part in ("train", "val", "test")
    }
    train_meta = frames["train"][["record_id", "species", "label"]].copy()
    train_ids = train_meta.record_id.astype(str).to_numpy()
    matrix_dir = MATRIX_ROOT / "within_species" / species
    feature_names = json.loads((matrix_dir / "feature_names.json").read_text(encoding="utf-8"))
    train_matrix = np.load(matrix_dir / "X_train.npy", mmap_mode="r")
    if train_matrix.shape[1] != len(feature_names):
        raise ValueError(f"Training schema mismatch for {species}")
    positions = {name: i for i, name in enumerate(feature_names)}
    app_parts: dict[str, np.ndarray] = {}

    cluster_overrides = load_app_cluster_assignments(app_ids, species, output)
    homology_train = raw_homology_features(train_meta[["record_id", "species"]], train_meta)
    homology_app = raw_homology_features(query_meta[["record_id", "species"]], train_meta, cluster_overrides)
    for group, start in (("homology30", 0), ("homology50", 15)):
        columns = [positions[f"{group}_{i:03d}"] for i in range(15)]
        raw_train = homology_train[:, start:start + 15]
        mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = raw_train.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
        app_parts[group] = ((homology_app[:, start:start + 15] - mean) / std).astype(np.float32)
        if len(columns) != app_parts[group].shape[1]:
            raise RuntimeError(f"{group} feature width mismatch")

    app_network, app_structure = app_string_blocks(query_meta, app_ids, species)
    for group in GROUPS:
        if group == "homology30":
            continue
        if group == "homology50":
            continue
        if group == "string_network":
            raw_app = app_network
        else:
            raw_app = load_raw_app(group, app_ids)
        raw_train, valid = aligned_group(group, train_ids)
        expected_width = sum(name.startswith(group + "_") for name in feature_names)
        if raw_train.shape[1] != raw_app.shape[1]:
            raise ValueError(f"{species}/{group}: train {raw_train.shape} versus app {raw_app.shape}")
        transformed = fit_transform(group, raw_train, valid, raw_app)
        if transformed.shape[1] != expected_width:
            raise ValueError(f"{species}/{group}: transformed width {transformed.shape[1]} != {expected_width}")
        app_parts[group] = transformed

    static_groups = [app_parts[group] for group in GROUPS]
    static_groups.append(app_structure)
    train_species = sorted(train_meta.species.astype(str).unique())
    onehot = np.zeros((len(app_ids), len(train_species)), dtype=np.float32)
    query_species = query_meta.species.astype(str).map({
        "Plasmodium_berghei_ANKA": "Plasmodium_berghei",
        "Plasmodium_falciparum_3D7": "Plasmodium_falciparum",
        "Plasmodium_knowlesi_H": "Plasmodium_knowlesi",
    }).fillna(query_meta.species.astype(str)).to_numpy()
    species_pos = {sp: i for i, sp in enumerate(train_species)}
    for i, value in enumerate(query_species):
        if value in species_pos:
            onehot[i, species_pos[value]] = 1.0
    static_groups.append(onehot)
    static = np.concatenate(static_groups, axis=1).astype(np.float32)

    hit_table = search_app_vs_train(species, app_ids, output)
    hit_table["identity"] = hit_table.pident.astype(float) / 100.0
    hit_table["coverage"] = hit_table.alnlen.astype(float) / hit_table[["qlen", "tlen"]].astype(float).max(axis=1)
    hit_table["score"] = hit_table.identity * hit_table.coverage
    train_y = dict(zip(train_meta.record_id.astype(str), train_meta.label.eq("essential").astype(float)))
    train_sp = dict(zip(train_meta.record_id.astype(str), train_meta.species.astype(str)))
    query_sp = dict(zip(app_ids, query_species))
    dynamic_mm = aggregate_mm(hit_table, app_ids, train_y, train_sp, query_sp)

    _, app_tax = SPECIES_TAX[species]
    app_map = map_rows(STRING_APP_RUN / f"app_map_{app_tax}.tsv")
    ppi_dynamic = app_ppi_dynamic(
        app_ids, app_map, train_ids, train_meta.label.eq("essential").astype(float).to_numpy(), species,
    )
    matrix = np.concatenate([static, dynamic_mm, ppi_dynamic], axis=1).astype(np.float32)
    if matrix.shape[1] != train_matrix.shape[1]:
        raise RuntimeError(f"{species}: repaired app matrix {matrix.shape} != train matrix {train_matrix.shape}")
    target = OUT / species
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "X_app.npy", matrix)
    np.save(target / "X_app_static.npy", static)
    np.save(target / "ids_app.npy", app_ids, allow_pickle=True)
    (target / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    manifest = {
        "species": species,
        "application_records": len(app_ids),
        "n_features": int(matrix.shape[1]),
        "feature_schema_matches_train": True,
        "homology_labels_source": "current scope train only",
        "ppi_labels_source": "current scope train only",
        "mmseqs_targets": "current scope train only",
        "application_gene_ids_sha256": __import__("hashlib").sha256("\n".join(app_ids).encode()).hexdigest(),
        "training_feature_schema": "within_species feature_names.json",
        "string_application_mapping_taxon": tax,
        "cluster50_search": "min_seq_id=0.5, coverage=0.5, cov_mode=0",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    app_meta = read_app_metadata()
    only = os.environ.get("PROTOESS_APP_ONLY", "")
    selected = [only] if only else list(SPECIES_TAX)
    results = [build_species(species, app_meta) for species in selected]
    (OUT / "application_matrix_repair_summary.json").write_text(json.dumps({
        "repaired_root": str(OUT),
        "scope_feature_root": str(MATRIX_ROOT),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"repaired application matrices for {len(results)} species -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

