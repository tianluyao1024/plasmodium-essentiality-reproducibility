from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
DNA = "ACGT"


def fasta(path: Path) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    header = None
    chunks: list[str] = []
    def save() -> None:
        if header is None:
            return
        token = header.split()[0]
        m = re.search(r"\b(?:gene_id|gene)=([A-Za-z0-9_.-]+)", header)
        biological_gene_id = m.group(1) if m else token.split(".", 1)[0].split("-p", 1)[0]
        out[token] = (token, "".join(chunks).upper(), biological_gene_id)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                save(); header = line[1:]; chunks = []
            elif line:
                chunks.append(line)
    save()
    return out


def basic(cds: str, protein: str) -> list[float]:
    cds = cds.upper().replace("U", "T")
    cds = "".join(nt if nt in DNA else "N" for nt in cds)
    protein = protein.upper().replace("*", "")
    protein = "".join(aa if aa in AA + "X" else "X" for aa in protein)
    return [math.log1p(len(cds)), math.log1p(len(protein)), len(cds) % 3,
            (cds.count("G") + cds.count("C")) / max(1, len(cds)),
            cds.count("N") / max(1, len(cds)), protein.count("X") / max(1, len(protein)),
            float("*" in protein), *[protein.count(a) / max(1, len(protein.replace("X", ""))) for a in AA],
            *[cds.count(n) / max(1, len(cds.replace("N", ""))) for n in DNA]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract sequence, annotation, k-mer and STRING blocks from one species folder in the dataset root.")
    ap.add_argument("--species-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--protein", type=Path, help="Full proteome FASTA; otherwise reads split protein FASTAs")
    ap.add_argument("--cds", type=Path, help="Full CDS FASTA; otherwise reads split CDS FASTAs")
    ap.add_argument("--gaf", type=Path)
    ap.add_argument("--string-links", type=Path)
    ap.add_argument("--string-map", type=Path)
    ap.add_argument("--input-mode", choices=("auto", "split", "full"), default="auto")
    ap.add_argument("--split-filter", choices=("all", "train", "validation", "test", "essential", "nonessential"), default="all")
    ap.add_argument("--max-records", type=int, help="Optional record cap for smoke tests")
    args = ap.parse_args()
    root = args.species_dir
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    if args.input_mode != "split" and args.protein is None:
        found_prot = next(iter(sorted(root.glob("*AnnotatedProteins*.fasta"))), None)
        if found_prot is not None:
            args.protein = found_prot
    if args.input_mode != "split" and args.cds is None:
        found_cds = next(iter(sorted(root.glob("*AnnotatedCDSs*.fasta"))), None)
        if found_cds is not None:
            args.cds = found_cds
    if args.input_mode == "full" and not (args.protein and args.cds):
        raise FileNotFoundError("--input-mode full requires paired annotated protein and CDS FASTAs")
    use_full = args.input_mode == "full" or (args.input_mode == "auto" and args.protein and args.cds)
    if use_full:
        proteins, cdss = fasta(args.protein), fasta(args.cds)
        by_protein_gene = {value[2]: value for value in proteins.values()}
        by_cds_gene = {value[2]: value for value in cdss.values()}
        for gene_id in by_protein_gene.keys() & by_cds_gene.keys():
            protein_record, cds_record = by_protein_gene[gene_id], by_cds_gene[gene_id]
            pairs[gene_id].update(protein=protein_record[1], cds=cds_record[1], gene_id=gene_id, split="genome")
    else:
        for split_dir in ("train", "validation", "test", "essential", "nonessential"):
            if args.split_filter not in {"all", split_dir}:
                continue
            d = root / split_dir
            if not d.exists():
                continue
            pp, cc = d / "protein.fasta", d / "cds.fasta"
            if not pp.exists() or not cc.exists():
                continue
            proteins, cdss = fasta(pp), fasta(cc)
            for gid in proteins.keys() & cdss.keys():
                pairs[gid].update(protein=proteins[gid][1], cds=cdss[gid][1], gene_id=proteins[gid][2], split=split_dir)
    if not pairs:
        raise FileNotFoundError("No paired protein/CDS FASTA records found under --species-dir")
    ids = sorted(pairs)
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")
        ids = ids[:args.max_records]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": ids, "gene_id": [pairs[i]["gene_id"] for i in ids], "split": [pairs[i]["split"] for i in ids]}).to_csv(args.out / "record_index.tsv", sep="\t", index=False)
    np.save(args.out / "basic_seq_raw.npy", np.asarray([basic(pairs[i]["cds"], pairs[i]["protein"]) for i in ids], dtype=np.float32))
    (args.out / "basic_seq_names.json").write_text(json.dumps(["log1p_cds_len", "log1p_protein_len", "cds_len_mod3", "gc", "cds_N_fraction", "protein_X_fraction", "protein_stop_fraction", *[f"aa_{x}" for x in AA], *[f"nt_{x}" for x in DNA]], indent=2), encoding="utf-8")

    go_by_gene: dict[str, set[str]] = defaultdict(set)
    gaf = args.gaf or next(iter(sorted([*root.glob("*.gaf"), *root.glob("*.goa")])), None)
    if gaf and gaf.exists():
        op = gzip.open if gaf.suffix == ".gz" else open
        with op(gaf, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!"): continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 5: continue
                keys = {cols[1], cols[2]}
                for key in keys:
                    go_by_gene[key].add(cols[4])
                    go_by_gene[key.split(".")[0]].add(cols[4])
    terms = sorted({t for vals in go_by_gene.values() for t in vals})
    go_x = np.zeros((len(ids), len(terms)), dtype=np.float32)
    term_ix = {t: j for j, t in enumerate(terms)}
    for i, gid in enumerate(ids):
        annotation_id = pairs[gid]["gene_id"]
        for t in go_by_gene.get(annotation_id, set()): go_x[i, term_ix[t]] = 1.0
    np.save(args.out / "go_terms_raw.npy", go_x)
    (args.out / "go_terms_vocab.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")

    manifest = {"n_records": len(ids), "basic_seq_dim": 31, "go_term_dim": len(terms), "gaf_used": str(gaf) if gaf else None,
                "note": "Raw local-input blocks only. No PLM, k-mer SVD, STRING blocks or cross-species features are fabricated here."}
    (args.out / "raw_extraction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
