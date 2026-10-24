from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the v6 candidate payload inventory.")
    parser.add_argument("--release-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_dir.resolve()
    manifest_path = root / "release_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        payload[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["payload_files"] = payload
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload_count": len(payload), "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
