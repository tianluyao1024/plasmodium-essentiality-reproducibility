"""Write a SHA-256 manifest for the confidential reviewer-only input bundle."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(r"D:\ProtoEss_Reviewer_Inputs_v46")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        try:
            is_regular = path.is_file()
        except OSError:
            # MMseqs temporary compatibility links are not reproducibility inputs.
            continue
        if is_regular and path.name not in {"input_manifest_sha256.csv", "bundle_metadata.json"}:
            rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)))
    with (ROOT / "input_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["relative_path", "bytes", "sha256"]); writer.writerows(rows)
    print(f"Hashed {len(rows)} files")


if __name__ == "__main__":
    main()
