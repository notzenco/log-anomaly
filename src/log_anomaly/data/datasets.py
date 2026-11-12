"""Registry of public, openly licensed log datasets used by this project.

Every entry records the provenance required to cite the data: where it comes
from, what licence it carries and how it must be credited. Raw data is never
committed to this repository; use :mod:`log_anomaly.data.download` (or
``scripts/download_data.sh``) to fetch and verify it.
"""

from __future__ import annotations

from dataclasses import dataclass

LOGHUB_DOI = "10.5281/zenodo.8196385"
LOGHUB_URL = "https://github.com/logpai/loghub"
LOGHUB_CITATION = (
    "Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. "
    "Loghub: A Large Collection of System Log Datasets for AI-driven Log "
    "Analytics. IEEE International Symposium on Software Reliability "
    "Engineering (ISSRE), 2023."
)
LOGHUB_LICENSE = (
    "Free for research or academic work; the Loghub repository URL must be "
    "referenced and the Loghub paper cited in any usage or distribution."
)

_ZENODO = "https://zenodo.org/api/records/8196385/files/{key}/content"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to fetch, verify and cite one dataset."""

    name: str
    description: str
    archive_url: str
    archive_md5: str
    member: str
    license_name: str
    citation: str
    source_url: str

    @property
    def archive_name(self) -> str:
        return self.archive_url.rsplit("/", 1)[-1]

    def attribution(self) -> str:
        return f"{self.name}: {self.license_name} Source: {self.source_url} Citation: {self.citation}"


DATASETS: dict[str, DatasetSpec] = {
    "linux": DatasetSpec(
        name="linux",
        description=(
            "Linux /var/log/messages from one server over 264 days: SSH and PAM "
            "authentication events, su sessions and service logs."
        ),
        archive_url=_ZENODO.format(key="Linux.tar.gz"),
        archive_md5="6d1802d7778126f21c001c6aa7b6b106",
        member="Linux.log",
        license_name=LOGHUB_LICENSE,
        citation=LOGHUB_CITATION,
        source_url=f"{LOGHUB_URL} (Loghub, DOI {LOGHUB_DOI})",
    ),
    "openssh": DatasetSpec(
        name="openssh",
        description=(
            "OpenSSH server log over 28 days (655k lines) from a host exposed to "
            "the internet: successful logins interleaved with background scanning."
        ),
        archive_url=_ZENODO.format(key="SSH.tar.gz"),
        archive_md5="a44e40b4348697dabf2bba56885e0a38",
        member="SSH.log",
        license_name=LOGHUB_LICENSE,
        citation=LOGHUB_CITATION,
        source_url=f"{LOGHUB_URL} (Loghub, DOI {LOGHUB_DOI})",
    ),
    "proxifier": DatasetSpec(
        name="proxifier",
        description=(
            "Proxifier proxy log from Windows (21k lines): application, "
            "destination host and port, bytes transferred and connection lifetime."
        ),
        archive_url=_ZENODO.format(key="Proxifier.tar.gz"),
        archive_md5="2612fce12cc3d16599ddb3db8e9c477a",
        member="Proxifier.log",
        license_name=LOGHUB_LICENSE,
        citation=LOGHUB_CITATION,
        source_url=f"{LOGHUB_URL} (Loghub, DOI {LOGHUB_DOI})",
    ),
}

DATASET_NAMES = tuple(sorted(DATASETS))


def get_spec(name: str) -> DatasetSpec:
    """Return the spec for ``name`` or raise a helpful error."""
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASET_NAMES)}") from None
