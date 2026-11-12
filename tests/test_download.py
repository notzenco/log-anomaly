"""Tests for dataset acquisition and checksum verification."""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path

import pytest

from log_anomaly.data.datasets import DATASET_NAMES, DATASETS, get_spec
from log_anomaly.data.download import (
    ChecksumMismatch,
    DownloadError,
    extract_member,
    fetch_dataset,
    fetch_spec,
    md5sum,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_archive(tmp_path: Path, member: str = "sample.log", body: str = "hello\n") -> Path:
    source = tmp_path / member
    source.write_text(body, encoding="utf-8")
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=member)
    return archive


def test_md5sum_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"payload" * 1000)
    assert md5sum(target) == hashlib.md5(b"payload" * 1000).hexdigest()  # noqa: S324


def test_fetch_dataset_extracts_member(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    spec = DATASETS["linux"].__class__(
        name="linux",
        description="test",
        archive_url=archive.as_uri(),
        archive_md5=md5sum(archive),
        member="sample.log",
        license_name="test",
        citation="test",
        source_url="test",
    )
    fetched = fetch_spec(spec, tmp_path / "dest")
    assert fetched.read_text(encoding="utf-8") == "hello\n"


def test_fetch_dataset_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    spec = DATASETS["linux"].__class__(
        name="linux",
        description="test",
        archive_url=archive.as_uri(),
        archive_md5="0" * 32,
        member="sample.log",
        license_name="test",
        citation="test",
        source_url="test",
    )
    with pytest.raises(ChecksumMismatch):
        fetch_spec(spec, tmp_path / "dest")
    # the unverified archive must not survive
    assert not (tmp_path / "dest" / "archives" / "sample.tar.gz").exists()


def test_extract_member_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="../escape.txt")
    with pytest.raises(DownloadError):
        extract_member(archive, "../escape.txt", tmp_path / "out")


def test_fetch_dataset_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        fetch_dataset("does-not-exist", Path("/tmp"))


def test_registry_entries_are_complete() -> None:
    assert DATASET_NAMES == ("linux", "openssh", "proxifier")
    for name in DATASET_NAMES:
        spec = get_spec(name)
        assert re.fullmatch(r"[0-9a-f]{32}", spec.archive_md5)
        assert spec.archive_url.startswith("https://")
        assert "Loghub" in spec.citation
        assert "Loghub" in spec.source_url


def test_shell_script_checksums_match_registry() -> None:
    """The standalone download script must not drift from the Python registry."""
    script = (REPO_ROOT / "scripts" / "download_data.sh").read_text(encoding="utf-8")
    for name in DATASET_NAMES:
        spec = get_spec(name)
        assert spec.archive_md5 in script, f"{name} md5 missing from download script"
        assert spec.archive_name in script, f"{name} archive name missing from download script"
