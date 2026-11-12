"""Log parsing: raw log lines in, normalised events out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from log_anomaly.parsing.proxifier import parse_proxifier, parse_proxifier_lines
from log_anomaly.parsing.schema import EVENT_DTYPES, EVENT_TYPES, empty_events
from log_anomaly.parsing.syslog import parse_syslog, parse_syslog_lines

__all__ = [
    "EVENT_DTYPES",
    "EVENT_TYPES",
    "ParseStats",
    "detect_format",
    "empty_events",
    "parse_events",
    "parse_file",
    "parse_proxifier",
    "parse_proxifier_lines",
    "parse_syslog",
    "parse_syslog_lines",
]

ParseStats = dict[str, int]


def detect_format(path: str | Path, sample_lines: int = 50) -> str:
    """Guess the log format by looking at the first non-empty line of ``path``.

    The two supported formats are disjoint in their opening characters, so one
    sample line is enough.
    """
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped[1:3].isdigit() and stripped[3] == ".":
                return "proxifier"
            return "syslog"
    return "syslog"


def parse_file(
    path: str | Path,
    *,
    log_format: str = "auto",
    anchor_year: int = 2004,
    keep_raw: bool = True,
) -> tuple[pd.DataFrame, ParseStats]:
    """Parse ``path`` into the normalised event frame."""
    chosen = detect_format(path) if log_format == "auto" else log_format
    if chosen == "proxifier":
        return parse_proxifier(path, anchor_year=anchor_year, keep_raw=keep_raw)
    if chosen == "syslog":
        return parse_syslog(path, anchor_year=anchor_year, keep_raw=keep_raw)
    raise ValueError(f"unknown log format {chosen!r}")


parse_events = parse_file
