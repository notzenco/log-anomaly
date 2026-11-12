"""Parse BSD/ISO syslog lines into the normalised event schema.

Two timestamp shapes are supported:

* BSD syslog (``Jun 14 15:16:01 host prog[pid]: msg``) - the Loghub Linux and
  OpenSSH corpora use this, and it carries **no year**.
* ISO syslog (``2024-06-14T15:16:01Z host prog[pid]: msg``) - what modern
  collectors emit, used by the synthetic corpus.

For the BSD shape the year is inferred: the caller supplies an anchor year and
the parser bumps it whenever a timestamp would otherwise move backwards, so a
log spanning December to January stays monotonic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from log_anomaly.parsing.schema import EVENT_DTYPES, empty_events

_MONTHS = {
    name: number
    for number, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

_BSD_RE = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<clock>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<rest>.*)$"
)
_ISO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<host>\S+)\s+(?P<rest>.*)$"
)
_PID_RE = re.compile(r"\[(?P<pid>\d+)\]\s*$")

_KV_RE = re.compile(r"(?P<key>\w+)=(?P<value>\"[^\"]*\"|\S+)")

_ACCEPTED_RE = re.compile(r"Accepted\s+(?P<method>\S+)\s+for\s+(?P<user>\S+)\s+from\s+(?P<src>\S+)")
_FAILED_PASSWORD_RE = re.compile(
    r"Failed\s+(?P<method>password|publickey|none)\s+for\s+(?:invalid user\s+)?(?P<user>\S+)\s+from\s+(?P<src>\S+)"
)
_INVALID_USER_RE = re.compile(r"Invalid user\s+(?P<user>\S+)\s+from\s+(?P<src>\S+)")
_AUTH_FAILURE_RE = re.compile(r"authentication failure;(?P<body>.*)")
_RHOST_RE = re.compile(r"rhost=(?P<src>\S*)")
_USER_FIELD_RE = re.compile(r"\buser=(?P<user>\S*)")
_SUDO_FAIL_RE = re.compile(r"sudo:\s*(?P<body>.*)")
_SUDO_ACTOR_RE = re.compile(r"^(?P<user>\S+)\s*:")
_LOGNAME_RE = re.compile(r"\blogname=(?P<user>\S*)")
_SESSION_OPEN_RE = re.compile(r"session opened for user\s+(?P<user>\S+)")
_SESSION_CLOSE_RE = re.compile(r"session closed for user\s+(?P<user>\S+)")
_DISCONNECT_RE = re.compile(r"(?:Connection closed by|Received disconnect from)\s+(?P<src>\S+)")
_MAX_ATTEMPTS_RE = re.compile(r"maximum authentication attempts exceeded for\s+(?P<user>\S+)\s+from\s+(?P<src>\S+)")
_FAILED_LOGIN_RE = re.compile(r"FAILED LOGIN\s+\d+\s+FROM\s+(?P<src>\S+)\s+USER\s+=\s*(?P<user>\S*)")


def _program_and_message(rest: str) -> tuple[str, str, int | None]:
    """Split ``prog[pid]: message`` into a program family, message and pid."""
    index = rest.find(": ")
    if index == -1:
        if rest.endswith(":"):
            return rest[:-1].strip(), "", None
        return rest.strip(), "", None
    head, message = rest[:index], rest[index + 2 :]
    pid: int | None = None
    pid_match = _PID_RE.search(head)
    if pid_match:
        pid = int(pid_match.group("pid"))
        head = head[: pid_match.start()]
    head = head.strip()
    if "sshd" in head:
        family = "sshd"
    else:
        first = head.split()[0] if head.split() else ""
        family = first.split("(")[0]
    return family, message, pid


def _key_values(message: str) -> dict[str, str]:
    return {m.group("key"): m.group("value").strip('"') for m in _KV_RE.finditer(message)}


def _classify(program: str, message: str) -> dict[str, object]:
    """Map a syslog message onto the event taxonomy and extract entities."""
    out: dict[str, object] = {
        "event_type": "other",
        "user": None,
        "src_ip": None,
        "src_port": None,
    }
    family = program.lower()

    if family == "sudo":
        if "incorrect password" in message or "authentication failure" in message:
            out["event_type"] = "sudo_failure"
            # the actor is in logname=; user= is the account being escalated to
            actor = _LOGNAME_RE.search(message)
            if actor:
                out["user"] = actor.group("user") or None
            else:
                user_field = _USER_FIELD_RE.search(message)
                if user_field:
                    out["user"] = user_field.group("user") or None
        elif "COMMAND=" in message:
            out["event_type"] = "sudo_success"
            actor = _SUDO_ACTOR_RE.match(message)
            if actor:
                out["user"] = actor.group("user")
        return out

    if family in {"sshd", "ssh", "login", "su"}:
        match = _ACCEPTED_RE.search(message)
        if match:
            out["event_type"] = "auth_success"
            out["user"] = match.group("user")
            out["src_ip"] = match.group("src")
            return out
        match = _FAILED_PASSWORD_RE.search(message)
        if match:
            out["event_type"] = "auth_failure"
            out["user"] = match.group("user")
            out["src_ip"] = match.group("src")
            return out
        match = _INVALID_USER_RE.search(message)
        if match:
            out["event_type"] = "auth_failure"
            out["user"] = match.group("user")
            out["src_ip"] = match.group("src")
            return out
        match = _MAX_ATTEMPTS_RE.search(message)
        if match:
            out["event_type"] = "auth_failure"
            out["user"] = match.group("user")
            out["src_ip"] = match.group("src")
            return out
        match = _AUTH_FAILURE_RE.search(message)
        if match:
            body = match.group("body")
            out["event_type"] = "auth_failure"
            rhost = _RHOST_RE.search(body)
            if rhost:
                out["src_ip"] = rhost.group("src") or None
            user_field = _USER_FIELD_RE.search(body)
            if user_field:
                out["user"] = user_field.group("user") or None
            return out
        match = _FAILED_LOGIN_RE.search(message)
        if match:
            out["event_type"] = "auth_failure"
            out["src_ip"] = match.group("src")
            out["user"] = match.group("user") or None
            return out
        if "check pass; user unknown" in message or "check pass" in message:
            out["event_type"] = "auth_failure"
            return out
        match = _DISCONNECT_RE.search(message)
        if match:
            out["event_type"] = "auth_failure"
            out["src_ip"] = match.group("src")
            return out
        match = _SESSION_OPEN_RE.search(message)
        if match:
            out["event_type"] = "session_open"
            out["user"] = match.group("user")
            return out
        match = _SESSION_CLOSE_RE.search(message)
        if match:
            out["event_type"] = "session_close"
            out["user"] = match.group("user")
            return out

    if family == "process":
        fields = _key_values(message)
        out["event_type"] = "process_exec"
        out["user"] = fields.get("user") or None
        out["process"] = fields.get("exe") or None
        out["parent_process"] = fields.get("ppname") or None
        out["_ppid"] = fields.get("ppid")
        return out

    if family == "conn":
        fields = _key_values(message)
        out["event_type"] = "network_conn"
        out["user"] = fields.get("user") or None
        out["process"] = fields.get("exe") or None
        destination = fields.get("dst", "")
        if ":" in destination:
            host, _, port = destination.rpartition(":")
            out["dst_ip"] = host
            out["dst_port"] = int(port) if port.isdigit() else None
        else:
            out["dst_ip"] = destination or None
        out["bytes_sent"] = int(fields["bytes_sent"]) if fields.get("bytes_sent", "").isdigit() else None
        received = fields.get("bytes_received", "")
        out["bytes_received"] = int(received) if received.isdigit() else None
        return out

    return out


def _day_of_year(month: int, day: int) -> int:
    """Day of year for a month/day pair, ignoring leap years."""
    cumulative = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return cumulative[month - 1] + day


def _assign_timestamps(
    parsed: list[tuple[int, int, int, int, int]],
    anchor_year: int,
    infer_rollover: bool,
) -> list[datetime]:
    """Turn (month, day, hour, minute, second) tuples into ordered datetimes.

    The year is bumped only when the *date* moves backwards by more than half
    a year, which is what a December-to-January wrap looks like. Small
    backwards steps are treated as the out-of-order lines that real syslog
    files contain, and a backwards step within the same day never bumps the
    year - concurrent writers are routinely a few seconds out of order.
    """
    result: list[datetime] = []
    current_year = anchor_year
    previous: datetime | None = None
    for month, day, hour, minute, second in parsed:
        try:
            candidate = datetime(current_year, month, day, hour, minute, second, tzinfo=UTC)
        except ValueError:
            # 29 February in a non-leap year: clamp to the 28th.
            candidate = datetime(current_year, month, 28, hour, minute, second, tzinfo=UTC)
        if infer_rollover and previous is not None:
            jump = _day_of_year(month, day) - _day_of_year(previous.month, previous.day)
            if jump < -180:
                try:
                    later = candidate.replace(year=candidate.year + 1)
                except ValueError:
                    later = candidate
                if later.date() >= previous.date():
                    candidate = later
                    current_year = later.year
        result.append(candidate)
        previous = candidate
    return result


def parse_syslog_lines(
    lines: Iterable[str],
    *,
    anchor_year: int = 2004,
    keep_raw: bool = True,
    infer_rollover: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse syslog ``lines`` into an event frame.

    Returns the frame and a stats dict with the number of lines seen, the
    number parsed and the number rejected.
    """
    records: list[dict[str, object]] = []
    raw_stamps: list[tuple[int, int, int, int, int] | None] = []
    iso_times: list[datetime | None] = []
    rejected = 0
    seen = 0
    index = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        seen += 1
        match = _BSD_RE.match(line)
        iso_time: datetime | None = None
        if match:
            month = _MONTHS.get(match.group("month").lower())
            if month is None:
                rejected += 1
                continue
            day = int(match.group("day"))
            hour, minute, second = (int(part) for part in match.group("clock").split(":"))
            raw_stamps.append((month, day, hour, minute, second))
            iso_times.append(None)
            host = match.group("host")
            rest = match.group("rest")
        else:
            iso_match = _ISO_RE.match(line)
            if iso_match is None:
                rejected += 1
                continue
            iso_time = datetime.fromisoformat(iso_match.group("ts").replace("Z", "+00:00"))
            if iso_time.tzinfo is None:
                iso_time = iso_time.replace(tzinfo=UTC)
            raw_stamps.append(None)
            iso_times.append(iso_time)
            host = iso_match.group("host")
            rest = iso_match.group("rest")

        program, message, pid = _program_and_message(rest)
        fields = _classify(program, message)
        record: dict[str, object] = {
            "event_index": index,
            "host": host,
            "program": program,
            "pid": pid,
            "message": message,
        }
        record.update(fields)
        if keep_raw:
            record["raw"] = line
        records.append(record)
        index += 1

    bsd_positions = [i for i, stamp in enumerate(raw_stamps) if stamp is not None]
    assigned = _assign_timestamps(
        [raw_stamps[i] for i in bsd_positions],  # type: ignore[misc]
        anchor_year,
        infer_rollover,
    )
    lookup = dict(zip(bsd_positions, assigned, strict=True))
    timestamps = [
        iso_times[i] if iso_times[i] is not None else lookup[i]
        for i in range(len(raw_stamps))
    ]

    frame = pd.DataFrame(records)
    if frame.empty:
        return empty_events(), {"lines": seen, "parsed": 0, "rejected": rejected}
    frame["ts"] = pd.Series(timestamps, index=frame.index, dtype="datetime64[ns, UTC]")
    frame = frame.reindex(columns=list(EVENT_DTYPES))
    for column, dtype in EVENT_DTYPES.items():
        if column not in frame.columns or frame[column].isna().all():
            frame[column] = pd.Series(dtype=dtype, index=frame.index)
    frame = frame.astype(EVENT_DTYPES)
    return frame, {"lines": seen, "parsed": len(frame), "rejected": rejected}


def parse_syslog(
    path: str | Path,
    *,
    anchor_year: int = 2004,
    keep_raw: bool = True,
    infer_rollover: bool = True,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse a syslog file from disk."""
    with Path(path).open("r", encoding=encoding, errors="replace") as handle:
        return parse_syslog_lines(
            handle,
            anchor_year=anchor_year,
            keep_raw=keep_raw,
            infer_rollover=infer_rollover,
        )
