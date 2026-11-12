"""Dataset acquisition: registry, download and checksum verification."""

from __future__ import annotations

from log_anomaly.data.datasets import DATASET_NAMES, DatasetSpec, get_spec
from log_anomaly.data.download import (
    ChecksumMismatch,
    DownloadError,
    dataset_paths,
    fetch_all,
    fetch_dataset,
    fetch_spec,
    md5sum,
)

__all__ = [
    "DATASET_NAMES",
    "ChecksumMismatch",
    "DatasetSpec",
    "DownloadError",
    "dataset_paths",
    "fetch_all",
    "fetch_dataset",
    "fetch_spec",
    "get_spec",
    "md5sum",
]
