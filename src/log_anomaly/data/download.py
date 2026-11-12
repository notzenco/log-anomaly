"""Fetch public datasets and verify them against a pinned checksum.

The rule this module enforces: a dataset is only usable if the bytes on disk
hash to the checksum recorded in the registry. A mismatch aborts the download
and removes the offending file rather than letting unverified data into the
pipeline.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from log_anomaly.data.datasets import DATASET_NAMES, DatasetSpec, get_spec

_CHUNK = 1 << 20
_USER_AGENT = "log-anomaly/0.1 (+https://github.com/notzenco)"


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or fails verification."""


class ChecksumMismatch(DownloadError):
    """Raised when a downloaded file does not match the pinned MD5."""


def md5sum(path: Path) -> str:
    """Return the hex MD5 digest of ``path``."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against a pinned value, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,  # noqa: S310 - http(s) only
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, _CHUNK)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {url}: {exc}") from exc
    partial.replace(destination)


def _assert_safe_member(info: tarfile.TarInfo) -> None:
    """Reject archive members whose paths escape the extraction root."""
    name = Path(info.name)
    if name.is_absolute() or ".." in name.parts:
        raise DownloadError(f"refusing to extract unsafe archive member {info.name!r}")


def extract_member(archive_path: Path, member: str, destination: Path) -> Path:
    """Extract ``member`` from the tar archive at ``archive_path``."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        try:
            info = archive.getmember(member)
        except KeyError:
            raise DownloadError(f"{member!r} not found in {archive_path.name}") from None
        _assert_safe_member(info)
        try:
            archive.extract(info, path=destination, filter="data")
        except TypeError:  # extraction filters landed in Python 3.12
            archive.extract(info, path=destination)
    extracted = destination / member
    if not extracted.exists():
        raise DownloadError(f"extraction of {member!r} produced no file")
    return extracted


def fetch_dataset(name: str, dest: Path, force: bool = False) -> Path:
    """Download and verify one dataset, returning the path to the log file.

    The download is skipped when a previously fetched copy is present and its
    checksum already matches the pinned value.
    """
    spec = get_spec(name)
    return fetch_spec(spec, dest, force=force)


def fetch_spec(spec: DatasetSpec, dest: Path, force: bool = False) -> Path:
    """Download and verify ``spec``, returning the path to the log file."""
    archive_dir = dest / "archives"
    log_dir = dest / "logs"
    archive_path = archive_dir / spec.archive_name
    target = log_dir / spec.member

    if not force and archive_path.exists() and md5sum(archive_path) == spec.archive_md5:
        if target.exists():
            return target
    elif not force and archive_path.exists():
        archive_path.unlink()

    if force or not archive_path.exists():
        _download(spec.archive_url, archive_path)

    actual = md5sum(archive_path)
    if actual != spec.archive_md5:
        archive_path.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"checksum mismatch for {spec.archive_name}: "
            f"expected {spec.archive_md5}, got {actual}"
        )

    return extract_member(archive_path, spec.member, log_dir)


def fetch_all(dest: Path, force: bool = False) -> dict[str, Path]:
    """Download every registered dataset."""
    return {name: fetch_dataset(name, dest, force=force) for name in DATASET_NAMES}


def dataset_paths(dest: Path) -> dict[str, Path]:
    """Return the expected log paths for datasets already present under ``dest``."""
    paths: dict[str, Path] = {}
    for name in DATASET_NAMES:
        candidate = dest / "logs" / get_spec(name).member
        if candidate.exists():
            paths[name] = candidate
    return paths
