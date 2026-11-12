from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CROSS_PREFIXES = ("mmseqs_dynamic_", "homology30_", "homology50_", "ppi_dynamic_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine locally extracted blocks and cross-species features by gene_id.")
    ap.add_argument("--feature-names", required=True, type=Path, help="Canonical feature_names.json (628 names)")
    ap.add_argument("--non-cross", required=True, type=Path, help="TSV with gene_id and the 562 non-cross columns")
    ap.add_argument("--cross", required=True, type=Path, help="TSV with gene_id and all 66 dynamic/cross columns")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    names = json.loads(args.feature_names.read_text(encoding="utf-8"))
    if len(names) != 628 or len(set(names)) != 628:
        raise ValueError("Canonical schema must contain 628 unique feature names")
    cross_names = [n for n in names if n.startswith(CROSS_PREFIXES)]
    base_names = [n for n in names if n not in set(cross_names)]
    if (len(base_names), len(cross_names)) != (562, 66):
        raise ValueError(f"Unexpected canonical group sizes: non-cross={len(base_names)}, cross={len(cross_names)}")
    base = pd.read_csv(args.non_cross, sep="\t", dtype={"gene_id": str})
    cross = pd.read_csv(args.cross, sep="\t", dtype={"gene_id": str})
    for label, frame, expected in (("non-cross", base, base_names), ("cross", cross, cross_names)):
        if "gene_id" not in frame or frame.gene_id.duplicated().any():
            raise ValueError(f"{label}: gene_id missing or duplicated")
        missing = sorted(set(expected) - set(frame.columns))
        if missing:
            raise ValueError(f"{label}: missing {len(missing)} expected columns; examples={missing[:5]}")
    if set(base.gene_id) != set(cross.gene_id):
        raise ValueError("Feature groups have different gene_id sets")
    b, c = base.set_index("gene_id"), cross.set_index("gene_id")
    c = c.loc[b.index]
    out = pd.concat([b[base_names], c[cross_names]], axis=1).reindex(columns=names)
    if out.shape != (len(base), 628):
        raise ValueError(f"Unexpected combined shape: {out.shape}")
    out.insert(0, "gene_id", b.index.astype(str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False, na_rep="")
    print(f"Combined {len(out)} genes with 628 features")


if __name__ == "__main__":
    main()
