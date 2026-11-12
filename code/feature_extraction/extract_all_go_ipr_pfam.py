from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prepare_go_features import (
    add_external_tgondii_domains,
    build_matrix,
    parse_go_files,
    terms_for_record,
)


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input_manifest.json"
OUT = ROOT / "all_sequences"


def main() -> None:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    annotations = ROOT / config["annotation_dir"]
    metadata_path = ROOT / config["metadata"]
    vocab_path = ROOT / config["raw_vocab"]
    reducer_path = ROOT / config["reducer"]
    external_path = config.get("external_domain_file")

    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str).fillna("")
    required = {"record_id", "species", "primary_id", "gene_id", "transcript_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks required columns: {sorted(missing)}")
    if metadata.record_id.duplicated().any():
        raise ValueError("record_id must be unique")

    aliases = parse_go_files(annotations)
    if external_path:
        add_external_tgondii_domains(aliases, ROOT / external_path)

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    records = []
    for _, row in metadata.iterrows():
        terms, aspect, evidence, interpro, pfam = terms_for_record(row, aliases)
        records.append({
            "terms": terms,
            "aspect": aspect,
            "evidence": evidence,
            "interpro": interpro,
            "pfam": pfam,
        })

    raw = build_matrix(
        records,
        vocab["terms"],
        vocab["evidence"],
        vocab["interpro"],
        vocab["pfam"],
    )
    if raw.shape != (len(metadata), 1573):
        raise ValueError(f"Expected a 1573-column raw matrix, got {raw.shape}")

    transform = joblib.load(reducer_path)
    reduced = transform["svd"].transform(
        ((raw - transform["mean"]) / transform["std"]).astype(np.float32)
    ).astype(np.float32)
    if reduced.shape != (len(metadata), 128):
        raise ValueError(f"Expected a 128-column reduced matrix, got {reduced.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((ROOT / config["reduced_names"]).read_text(encoding="utf-8"))
    output_path = OUT / "go_ipr_pfam_all_sequences.npz"
    np.savez_compressed(
        output_path,
        record_id=metadata.record_id.to_numpy(str),
        gene_id=metadata.gene_id.to_numpy(str),
        split=metadata["split"].to_numpy(str) if "split" in metadata else np.repeat("A1H1", len(metadata)),
        raw_1573=raw,
        reduced_128=reduced,
        raw_feature_names=np.asarray(json.loads((ROOT / config["raw_feature_names"]).read_text(encoding="utf-8")), dtype=str),
        reduced_feature_names=np.asarray(names, dtype=str),
    )
    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report = {
        "species_folder": config["species_folder"],
        "source_species_key": config["source_species_key"],
        "rows": len(metadata),
        "raw_shape": list(raw.shape),
        "reduced_shape": list(reduced.shape),
        "unique_record_ids": int(metadata.record_id.nunique()),
        "raw_missing_values": int(np.isnan(raw).sum()),
        "reduced_missing_values": int(np.isnan(reduced).sum()),
        "labels_used": False,
        "output": output_path.name,
        "sha256": checksum,
    }
    (OUT / "output_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(metadata), "raw_shape": list(raw.shape),
                      "reduced_shape": list(reduced.shape),
                      "output": str(output_path), "sha256": checksum}, ensure_ascii=False))


if __name__ == "__main__":
    main()
