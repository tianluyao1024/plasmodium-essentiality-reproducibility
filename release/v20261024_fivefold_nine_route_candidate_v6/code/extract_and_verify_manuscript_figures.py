from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and verify the six embedded main-figure images from the manuscript DOCX.")
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--figure-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.figure_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with zipfile.ZipFile(args.docx) as archive:
        for item in manifest["figures"]:
            media_path = item["docx_media"]
            data = archive.read(media_path)
            actual = sha256_bytes(data)
            expected = item["embedded_png_sha256"]
            if actual != expected:
                raise ValueError(f"Embedded image checksum mismatch for {item['figure']}: {actual} != {expected}")
            output = args.output_dir / f"{item['figure']}.png"
            output.write_bytes(data)
            results.append({"figure": item["figure"], "docx_media": media_path,
                            "sha256": actual, "bytes": len(data), "output": output.name})

    report = {"docx": args.docx.name, "figures_extracted": len(results), "figures": results}
    (args.output_dir / "extraction_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
