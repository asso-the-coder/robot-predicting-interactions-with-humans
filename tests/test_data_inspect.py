from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from engagement_intent.data.inspect import inspect_raw_directory, inspect_zip, sha256_file


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"engagement-intent")

    assert sha256_file(source) == hashlib.sha256(b"engagement-intent").hexdigest()


def test_inspect_zip_inventories_members_without_extracting(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Lobby_1/example.csv", "frame,pid\n0,1\n")
        archive.writestr("README.md", "dataset documentation")

    result = inspect_zip(archive_path)

    assert result["member_count"] == 2
    assert result["extensions"] == {".csv": 1, ".md": 1}
    assert {member["path"] for member in result["members"]} == {
        "Lobby_1/example.csv",
        "README.md",
    }
    assert not (tmp_path / "Lobby_1").exists()


def test_inspect_raw_directory_reports_files_and_zip_contents(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("source notes", encoding="utf-8")
    archive_path = tmp_path / "release.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("records.csv", "label\n1\n")

    result = inspect_raw_directory(tmp_path)

    assert result["file_count"] == 2
    assert result["total_bytes"] == sum(path.stat().st_size for path in tmp_path.iterdir())
    zip_record = next(record for record in result["files"] if record["path"] == "release.zip")
    assert zip_record["zip"]["member_count"] == 1


def test_inspect_raw_directory_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inspect_raw_directory(tmp_path / "missing")

