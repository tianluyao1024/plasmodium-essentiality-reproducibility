from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
EXCLUDE = {"Danio_rerio", "Candida_albicans__A21", "Bombyx_mori"}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def add_alias(aliases: set[str], value: object) -> None:
    if value is None or pd.isna(value):
        return
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return
    for token in re.split(r"[\s,;|]+", text):
        token = token.strip()
        if not token or token.lower() in {"nan", "none", "protein", "cds"}:
            continue
        cleaned = token.strip("[](){}<>\"'")
        if cleaned:
            aliases.add(cleaned)
            aliases.add(cleaned.upper())
            aliases.add(cleaned.lower())
            if "." in cleaned:
                aliases.add(cleaned.split(".", 1)[0])
                aliases.add(cleaned.split(".", 1)[0].upper())
                aliases.add(cleaned.split(".", 1)[0].lower())


def aliases_from_row(row: pd.Series) -> set[str]:
    aliases: set[str] = set()
    for col in ("record_id", "primary_id", "gene_id", "transcript_id"):
        add_alias(aliases, row.get(col, ""))
    for col in ("cds_header", "protein_header"):
        header = str(row.get(col, ""))
        # Never tokenize the full header. Generic product words such as
        # "putative" and "protein" can otherwise connect unrelated genes.
        for pattern in (
            r"symbol=([^\s|;]+)",
            r"gene_symbol:([^\s|;]+)",
            r"gene:([^\s|;]+)",
            r"transcript:([^\s|;]+)",
            r"Acc:([A-Za-z0-9_:\-.]+)",
            r"CGDID:([A-Za-z0-9_:\-.]+)",
            r"FlyBase:([A-Za-z0-9_:\-.]+)",
            r"UniProt/[A-Za-z-]+:([A-Za-z0-9_:\-.]+)",
            r"UniProtKB[:/]([A-Za-z0-9_:\-.]+)",
            r"source_header=([^\s|;]+)",
            r"original_header=([^\s|;]+)",
            r"\|([A-Z0-9_]+\.[0-9]+)\|",
            r"\|([A-Z]{2,}[A-Z0-9_]+\d+[a-zA-Z]?)\|",
            r"\|(orf19\.\d+)\|",
            r"\|(C[0-9A-Z_]+[WC]_[AB])\|",
            r"\|(CAGL0[A-Z][0-9]+g)\|",
            r"\|(PAS_[A-Za-z0-9_]+)\|",
            r"\|(Y[A-Z]{2}[0-9]{3}[WC])\|",
            r"\|(SP[A-Z0-9.]+)\|",
        ):
            for match in re.finditer(pattern, header, flags=re.IGNORECASE):
                add_alias(aliases, match.group(1))
    return aliases


def parse_go_files(go_dir: Path) -> dict[str, dict[str, object]]:
    by_alias: dict[str, dict[str, object]] = {}
    paths = sorted([p for p in go_dir.rglob("*") if p.is_file() and (p.suffix in {".goa", ".gz"} or ".gaf" in p.name)])
    for path in paths:
        species = path.parent.name
        if species == "Toxoplasma_gondii" and path.name.startswith("ToxoDB-"):
            print(f"skip_go\t{species}\t{path}", flush=True)
            continue
        print(f"parse_go\t{species}\t{path}", flush=True)
        with open_text(path) as handle:
            for line in handle:
                if not line or line.startswith("!"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue
                db, obj_id, obj_symbol, qualifier, go_id = parts[:5]
                evidence = parts[6]
                aspect = parts[8] if len(parts) > 8 else ""
                name = parts[9] if len(parts) > 9 else ""
                synonyms = parts[10] if len(parts) > 10 else ""
                gp_form = parts[16] if len(parts) > 16 else ""
                # Restrict lookup keys to stable identifiers. Indexing free-text
                # names or synonyms creates cross-gene contamination when common
                # biological words are shared by many records.
                alias_values = [obj_id, obj_symbol, gp_form]
                if db:
                    alias_values.append(f"{db}:{obj_id}")
                for raw in alias_values:
                    for alias in re.split(r"[|,;\s]+", str(raw)):
                        alias = alias.strip()
                        if not alias or alias == "-":
                            continue
                        for key in {alias, alias.upper(), alias.lower(), alias.split(".", 1)[0]}:
                            rec = by_alias.setdefault(
                                f"{species}\t{key}",
                                {"terms": set(), "aspect": Counter(), "evidence": Counter(), "interpro": set(), "pfam": set()},
                            )
                            rec["terms"].add(go_id)
                            if aspect:
                                rec["aspect"][aspect] += 1
                            if evidence:
                                rec["evidence"][evidence] += 1
                            for ipr in re.findall(r"InterPro:(IPR\d+)", line):
                                rec["interpro"].add(ipr)
                            for pfam in re.findall(r"Pfam:(PF\d+)", line):
                                rec["pfam"].add(pfam)
    return by_alias


def add_external_tgondii_domains(by_alias: dict[str, dict[str, object]], path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path, sep="\t").fillna("")
    species = "Toxoplasma_gondii"
    for _, row in df.iterrows():
        gene_id = str(row.get("tgondii_gt1_gene_id", "")).strip()
        if not gene_id:
            continue
        rec = by_alias.setdefault(
            f"{species}\t{gene_id}",
            {"terms": set(), "aspect": Counter(), "evidence": Counter(), "interpro": set(), "pfam": set()},
        )
        for value in str(row.get("go_ids", "")).split("|"):
            value = value.strip()
            if value:
                rec["terms"].add(value)
        for cell, target in [(row.get("interpro_ids", ""), "interpro"), (row.get("pfam_ids", ""), "pfam")]:
            for value in str(cell).split("|"):
                value = value.strip()
                if value:
                    rec[target].add(value)
        for alias in {gene_id.upper(), gene_id.lower()}:
            by_alias[f"{species}\t{alias}"] = rec
    print(f"parse_external_domains\t{species}\t{path}\tgenes={len(df)}", flush=True)


def terms_for_record(row: pd.Series, by_alias: dict[str, dict[str, object]]) -> tuple[set[str], Counter, Counter, set[str], set[str]]:
    species = str(row["species"])
    terms: set[str] = set()
    aspect = Counter()
    evidence = Counter()
    interpro: set[str] = set()
    pfam: set[str] = set()
    for alias in aliases_from_row(row):
        rec = by_alias.get(f"{species}\t{alias}")
        if rec:
            terms.update(rec["terms"])
            aspect.update(rec["aspect"])
            evidence.update(rec["evidence"])
            interpro.update(rec["interpro"])
            pfam.update(rec["pfam"])
    return terms, aspect, evidence, interpro, pfam


def build_matrix(
    records: list[dict[str, object]],
    vocab: list[str],
    evid_vocab: list[str],
    interpro_vocab: list[str],
    pfam_vocab: list[str],
) -> np.ndarray:
    term_to_i = {t: i for i, t in enumerate(vocab)}
    evid_to_i = {e: i for i, e in enumerate(evid_vocab)}
    interpro_to_i = {t: i for i, t in enumerate(interpro_vocab)}
    pfam_to_i = {t: i for i, t in enumerate(pfam_vocab)}
    n_base = 13 + len(evid_vocab)
    x = np.zeros((len(records), n_base + len(vocab) + len(interpro_vocab) + len(pfam_vocab)), dtype=np.float32)
    for i, rec in enumerate(records):
        terms = rec["terms"]
        aspect = rec["aspect"]
        evidence = rec["evidence"]
        interpro = rec["interpro"]
        pfam = rec["pfam"]
        total_ann = sum(aspect.values())
        x[i, 0] = 1.0 if terms else 0.0
        x[i, 1] = len(terms)
        x[i, 2] = total_ann
        for j, asp in enumerate(("P", "F", "C"), start=3):
            x[i, j] = aspect.get(asp, 0)
        x[i, 6] = sum(evidence.get(e, 0) for e in ("EXP", "IDA", "IPI", "IMP", "IGI", "IEP"))
        x[i, 7] = sum(evidence.get(e, 0) for e in ("ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR", "IRD", "RCA"))
        x[i, 8] = evidence.get("IEA", 0)
        x[i, 9] = 1.0 if interpro else 0.0
        x[i, 10] = len(interpro)
        x[i, 11] = 1.0 if pfam else 0.0
        x[i, 12] = len(pfam)
        for ev, c in evidence.items():
            j = evid_to_i.get(ev)
            if j is not None:
                x[i, 13 + j] = c
        offset = n_base
        for term in terms:
            j = term_to_i.get(term)
            if j is not None:
                x[i, offset + j] = 1.0
        offset += len(vocab)
        for term in interpro:
            j = interpro_to_i.get(term)
            if j is not None:
                x[i, offset + j] = 1.0
        offset += len(interpro_vocab)
        for term in pfam:
            j = pfam_to_i.get(term)
            if j is not None:
                x[i, offset + j] = 1.0
    counts = x[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]]
    x[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]] = np.log1p(counts)
    if evid_vocab:
        x[:, 13 : 13 + len(evid_vocab)] = np.log1p(x[:, 13 : 13 + len(evid_vocab)])
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dedup95_strict")
    parser.add_argument("--split-name", default="random_species_label_split_seed42")
    parser.add_argument("--go-dir", default="downloads/go_current")
    parser.add_argument("--out", default="eukaryote_model_datasets/features/dedup95_strict/go_ipr_pfam")
    parser.add_argument("--top-terms", type=int, default=768)
    parser.add_argument("--top-interpro", type=int, default=512)
    parser.add_argument("--top-pfam", type=int, default=256)
    args = parser.parse_args()

    split_dir = Path("eukaryote_model_datasets") / args.dataset / args.split_name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_alias = parse_go_files(Path(args.go_dir))
    add_external_tgondii_domains(
        by_alias,
        Path("downloads/uniprot_tgondii_gt1/tgondii_gt1_gene_uniprot_go_interpro_pfam.tsv"),
    )

    split_frames = {}
    split_records = {}
    coverage_rows = []
    for split in SPLITS:
        df = pd.read_csv(split_dir / f"{split}_metadata.tsv", sep="\t")
        df = df.loc[~df["species"].isin(EXCLUDE)].reset_index(drop=True)
        records = []
        for _, row in df.iterrows():
            terms, aspect, evidence, interpro, pfam = terms_for_record(row, by_alias)
            records.append({"terms": terms, "aspect": aspect, "evidence": evidence, "interpro": interpro, "pfam": pfam})
        split_frames[split] = df
        split_records[split] = records
        for species, sub in df.groupby("species", sort=True):
            idx = sub.index.to_numpy()
            matched = sum(1 for i in idx if split_records[split][i]["terms"])
            coverage_rows.append(
                {
                    "split": split,
                    "species": species,
                    "n": int(len(idx)),
                    "matched_go": int(matched),
                    "matched_interpro": int(sum(1 for i in idx if split_records[split][i]["interpro"])),
                    "matched_pfam": int(sum(1 for i in idx if split_records[split][i]["pfam"])),
                    "coverage": float(matched / len(idx)) if len(idx) else 0.0,
                }
            )

    term_counter = Counter()
    evid_counter = Counter()
    interpro_counter = Counter()
    pfam_counter = Counter()
    for rec in split_records["train"]:
        term_counter.update(rec["terms"])
        evid_counter.update(rec["evidence"])
        interpro_counter.update(rec["interpro"])
        pfam_counter.update(rec["pfam"])
    vocab = [t for t, _ in term_counter.most_common(args.top_terms)]
    evid_vocab = [e for e, _ in evid_counter.most_common(32)]
    interpro_vocab = [t for t, _ in interpro_counter.most_common(args.top_interpro)]
    pfam_vocab = [t for t, _ in pfam_counter.most_common(args.top_pfam)]
    feature_names = (
        [
            "go_has",
            "go_n_terms",
            "go_n_annotations",
            "go_aspect_P",
            "go_aspect_F",
            "go_aspect_C",
            "go_evidence_experimental",
            "go_evidence_phylo_comp",
            "go_evidence_iea",
            "interpro_has",
            "interpro_n_terms",
            "pfam_has",
            "pfam_n_terms",
        ]
        + [f"go_evidence_{e}" for e in evid_vocab]
        + [f"go_term_{t}" for t in vocab]
        + [f"interpro_{t}" for t in interpro_vocab]
        + [f"pfam_{t}" for t in pfam_vocab]
    )

    for split in SPLITS:
        df = split_frames[split]
        x = build_matrix(split_records[split], vocab, evid_vocab, interpro_vocab, pfam_vocab)
        np.save(out_dir / f"{split}_emb.npy", x)
        np.save(out_dir / f"{split}_ids.npy", df["record_id"].to_numpy(dtype=object))
        print(f"{split}\t{x.shape}\tmatched={int((x[:,0] > 0).sum())}", flush=True)

    pd.DataFrame(coverage_rows).to_csv(out_dir / "go_coverage_by_species.tsv", sep="\t", index=False)
    (out_dir / "go_feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (out_dir / "go_vocab.json").write_text(
        json.dumps({"terms": vocab, "evidence": evid_vocab, "interpro": interpro_vocab, "pfam": pfam_vocab}, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(coverage_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
