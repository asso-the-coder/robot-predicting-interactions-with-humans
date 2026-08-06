from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from engagement_intent.data.stage import extract_zip_once, stage_nested_zips


def test_stage_nested_zips_preserves_archives_and_extracts_contents(tmp_path: Path) -> None:
    nested_archive = tmp_path / "lobby.zip"
    with ZipFile(nested_archive, "w") as archive:
        archive.writestr("scenario_001.csv", "pid,label\n1,1\n")

    release_archive = tmp_path / "release.zip"
    with ZipFile(release_archive, "w") as archive:
        archive.write(nested_archive, "Lobby_1/lobby.zip")

    nested_archive.unlink()
    output_dir = tmp_path / "processed"
    staged = stage_nested_zips(release_archive, output_dir)

    assert staged == [release_archive.resolve(), output_dir / "Lobby_1" / "lobby.zip"]
    assert (output_dir / "Lobby_1" / "lobby.zip").is_file()
    assert (output_dir / "Lobby_1" / "lobby" / "scenario_001.csv").is_file()
    assert release_archive.is_file()


def test_extract_zip_once_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="escapes output directory"):
        extract_zip_once(archive_path, tmp_path / "processed")

    assert not (tmp_path / "outside.txt").exists()


def test_extract_zip_once_refuses_nonempty_output(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("data.csv", "label\n0\n")

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "owned.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        extract_zip_once(archive_path, output_dir)

    assert (output_dir / "owned.txt").read_text(encoding="utf-8") == "do not overwrite"

