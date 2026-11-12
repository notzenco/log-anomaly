"""Parse Proxifier proxy logs into the normalised event schema.

Proxifier records outbound TCP connections made by applications:

    [10.30 16:49:06] chrome.exe - manhua.163.com:443 close, 11971 bytes sent,
    18220 bytes received, lifetime 00:09

The timestamp is ``day.month`` with no year, and the file is not stored in
chronological order, so the year is fixed rather than inferred and the caller
is expected to sort by time before windowing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from log_anomaly.parsing.schema import EVENT_DTYPES, empty_events

#: Sentinel host for proxy logs, which do not record the machine they ran on.
PROXIFIER_HOST = "proxifier-host"

# Proxifier stamps lines as ``[MM.DD HH:MM:SS]`` in this corpus.
_LINE_RE = re.compile(
    r"^\[(?P<month>\d{1,2})\.(?P<day>\d{1,2})\s+(?P<clock>\d{2}:\d{2}:\d{2})\]\s+"
    r"(?P<app>\S+)(?:\s+\*(?P<pid>\d+))?\s+-\s+"
    r"(?P<dest>\S+?)(?::(?P<port>\d+))?(?:\s+(?P<rest>.*))?$"
)
_BYTES_SENT_RE = re.compile(r"(?P<value>[\d,]+)\s+bytes(?:\s+\([^)]*\))?\s+sent")
_BYTES_RECEIVED_RE = re.compile(r"(?P<value>[\d,]+)\s+bytes(?:\s+\([^)]*\))?\s+received")
_LIFETIME_RE = re.compile(r"lifetime\s+(?P<value>[<\d][^\s]*(?:\s+\S+)?)")


def _parse_duration(value: str) -> float | None:
    value = value.strip()
    if value.startswith("<"):
        digits = re.match(r"<(\d+)", value)
        return float(digits.group(1)) if digits else None
    parts = value.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    seconds = 0
    for number in numbers:
        seconds = seconds * 60 + number
    return float(seconds)


def parse_proxifier_lines(
    lines: Iterable[str],
    *,
    anchor_year: int = 2017,
    keep_raw: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse Proxifier ``lines`` into an event frame plus parse stats."""
    records: list[dict[str, object]] = []
    timestamps: list[datetime] = []
    rejected = 0
    seen = 0
    index = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        seen += 1
        match = _LINE_RE.match(line)
        if match is None:
            rejected += 1
            continue
        hour, minute, second = (int(part) for part in match.group("clock").split(":"))
        try:
            timestamps.append(
                datetime(
                    anchor_year,
                    int(match.group("month")),
                    int(match.group("day")),
                    hour,
                    minute,
                    second,
                    tzinfo=UTC,
                )
            )
        except ValueError:
            rejected += 1
            continue

        rest = match.group("rest") or ""
        sent_match = _BYTES_SENT_RE.search(rest)
        received_match = _BYTES_RECEIVED_RE.search(rest)
        lifetime_match = _LIFETIME_RE.search(rest)

        record: dict[str, object] = {
            "event_index": index,
            "host": PROXIFIER_HOST,
            "program": "proxifier",
            "pid": int(match.group("pid")) if match.group("pid") else None,
            "event_type": "network_conn",
            "user": None,
            "src_ip": None,
            "src_port": None,
            "dst_ip": match.group("dest"),
            "dst_port": int(match.group("port")) if match.group("port") else None,
            "parent_process": None,
            "process": match.group("app"),
            "bytes_sent": int(sent_match.group("value").replace(",", "")) if sent_match else None,
            "bytes_received": (
                int(received_match.group("value").replace(",", "")) if received_match else None
            ),
            "duration_s": _parse_duration(lifetime_match.group("value")) if lifetime_match else None,
            "message": rest,
        }
        if keep_raw:
            record["raw"] = line
        records.append(record)
        index += 1

    if not records:
        return empty_events(), {"lines": seen, "parsed": 0, "rejected": rejected}

    frame = pd.DataFrame(records)
    frame["ts"] = pd.Series(timestamps, index=frame.index, dtype="datetime64[ns, UTC]")
    frame = frame.reindex(columns=list(EVENT_DTYPES))
    for column, dtype in EVENT_DTYPES.items():
        if frame[column].isna().all():
            frame[column] = pd.Series(dtype=dtype, index=frame.index)
    frame = frame.astype(EVENT_DTYPES)
    return frame, {"lines": seen, "parsed": len(frame), "rejected": rejected}


def parse_proxifier(
    path: str | Path,
    *,
    anchor_year: int = 2017,
    keep_raw: bool = True,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse a Proxifier log file from disk."""
    with Path(path).open("r", encoding=encoding, errors="replace") as handle:
        return parse_proxifier_lines(handle, anchor_year=anchor_year, keep_raw=keep_raw)
