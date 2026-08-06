"""Inventory raw dataset files without modifying or extracting them."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from zipfile import BadZipFile, ZipFile


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_zip(path: Path) -> dict[str, Any]:
    """Return a metadata-only inventory of a ZIP archive."""

    try:
        with ZipFile(path) as archive:
            files = [member for member in archive.infolist() if not member.is_dir()]
    except BadZipFile as error:
        raise ValueError(f"Not a valid ZIP archive: {path}") from error

    extensions = Counter(Path(member.filename).suffix.lower() or "<none>" for member in files)
    return {
        "member_count": len(files),
        "uncompressed_bytes": sum(member.file_size for member in files),
        "extensions": dict(sorted(extensions.items())),
        "members": [
            {
                "path": member.filename,
                "compressed_bytes": member.compress_size,
                "uncompressed_bytes": member.file_size,
                "crc32": f"{member.CRC:08x}",
            }
            for member in files
        ],
    }


def inspect_raw_directory(data_dir: Path) -> dict[str, Any]:
    """Inventory every file under a raw-data directory."""

    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")

    paths = sorted(path for path in data_dir.rglob("*") if path.is_file())
    records: list[dict[str, Any]] = []
    for path in paths:
        record: dict[str, Any] = {
            "path": path.relative_to(data_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".zip":
            record["zip"] = inspect_zip(path)
        records.append(record)

    return {
        "data_dir": str(data_dir),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory raw PAR-D files without modifying or extracting them."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Parent directories are created if needed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = inspect_raw_directory(args.data_dir)
    rendered = json.dumps(inventory, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

