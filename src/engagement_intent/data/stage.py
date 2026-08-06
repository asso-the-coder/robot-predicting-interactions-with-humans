"""Safely stage immutable raw ZIP archives under a processed-data directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile, ZipFile


def _validated_members(archive: ZipFile, output_dir: Path) -> list[str]:
    output_root = output_dir.resolve()
    member_names: list[str] = []
    for member in archive.infolist():
        destination = (output_dir / member.filename).resolve()
        if destination != output_root and output_root not in destination.parents:
            raise ValueError(f"Archive member escapes output directory: {member.filename}")
        member_names.append(member.filename)
    return member_names


def extract_zip_once(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract one ZIP after validating paths and refusing non-empty outputs."""

    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    try:
        with ZipFile(archive_path) as archive:
            member_names = _validated_members(archive, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            archive.extractall(output_dir)
    except BadZipFile as error:
        raise ValueError(f"Not a valid ZIP archive: {archive_path}") from error

    return [output_dir / member for member in member_names]


def stage_nested_zips(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract an archive and recursively stage ZIP members beside their archives."""

    extracted = extract_zip_once(archive_path, output_dir)
    staged_archives = [archive_path.resolve()]
    pending = [path for path in extracted if path.is_file() and path.suffix.lower() == ".zip"]

    while pending:
        nested_archive = pending.pop(0)
        nested_output = nested_archive.with_suffix("")
        nested_members = extract_zip_once(nested_archive, nested_output)
        staged_archives.append(nested_archive)
        pending.extend(
            path for path in nested_members if path.is_file() and path.suffix.lower() == ".zip"
        )

    return staged_archives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely extract a raw dataset ZIP and any nested ZIPs."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staged = stage_nested_zips(args.archive, args.output_dir)
    print(f"Staged {len(staged)} archive(s) under {args.output_dir.resolve()}")
    for archive in staged:
        print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

