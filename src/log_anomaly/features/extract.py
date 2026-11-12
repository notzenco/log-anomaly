"""Time-windowed feature engineering over a stream of log events.

Every feature is computed from a **trailing** window that ends at the event
being described, so scoring a log file offline reproduces exactly what an
online scorer would have produced at that timestamp. Nothing in the extractor
looks at the future, and no label is ever read.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

#: Feature groups, so a report can say which signals were live for a corpus.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "auth": (
        "auth_fail_count_1h",
        "auth_fail_count_10m",
        "auth_fail_count_zscore",
        "src_fail_count_10m",
        "src_fail_count_1h",
        "src_distinct_users_1h",
        "src_user_entropy_1h",
        "login_count_1h",
        "user_distinct_hosts_1h",
        "sudo_fail_count_1h",
        "new_user_host_pair",
        "new_src_for_user",
        "unknown_user",
        "off_hours",
        "hour_profile_distance",
        "root_activity",
    ),
    "process": (
        "proc_pair_is_new",
        "proc_pair_rarity",
        "proc_new_for_user",
        "proc_freq_rarity",
    ),
    "network": (
        "net_conn_count_1h",
        "net_new_dest_for_proc",
        "net_dest_rarity",
        "net_bytes_sent_zscore",
        "net_total_bytes_1h",
    ),
    "context": (
        "type_auth_failure",
        "type_auth_success",
        "type_session",
        "type_sudo",
        "type_process",
        "type_network",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)

#: Events that count as a login for the user-scoped counters.
_LOGIN_TYPES = frozenset({"auth_success", "session_open"})
_FAILURE_TYPES = frozenset({"auth_failure"})
_SUDO_FAIL_TYPES = frozenset({"sudo_failure"})


@dataclass
class SlidingWindow:
    """Per-key trailing time window.

    ``push`` appends a value and drops everything older than the window, then
    returns the surviving entries so callers can count, sum or take distinct
    values without re-scanning the frame.
    """

    window: timedelta
    _data: defaultdict[Hashable, deque[tuple[datetime, object]]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def push(self, key: Hashable, ts: datetime, value: object = None) -> list[tuple[datetime, object]]:
        entries = self._data[key]
        entries.append((ts, value))
        cutoff = ts - self.window
        while entries and entries[0][0] < cutoff:
            entries.popleft()
        return list(entries)

    def count(self, key: Hashable) -> int:
        entries = self._data.get(key)
        return len(entries) if entries else 0


@dataclass
class RunningStats:
    """Welford running mean and standard deviation."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def zscore(self, value: float) -> float:
        spread = self.std
        if spread <= 0.0:
            return 0.0
        return (value - self.mean) / spread


@dataclass
class FeatureSet:
    """Features for every event, plus the events they describe."""

    features: pd.DataFrame
    events: pd.DataFrame
    feature_names: tuple[str, ...]
    availability: dict[str, float]

    def split(self, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (features, events) restricted to ``mask``."""
        return self.features[mask], self.events[mask]


def _entropy(values: Iterable[object]) -> float:
    """Shannon entropy (nats) of a value distribution."""
    counts: dict[object, int] = defaultdict(int)
    total = 0
    for value in values:
        counts[value] += 1
        total += 1
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _circular_block(hours: set[int]) -> tuple[int, int]:
    """Return the (start, end) of the circular block that covers ``hours``.

    A user who works nights has hours like {22, 23, 0, 1}; treating that as
    the block 22 -> 1 (rather than 0 -> 23) is what keeps a midnight login
    from looking like daytime activity.
    """
    ordered = sorted(hours)
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    best_gap = -1
    best_index = 0
    for index, hour in enumerate(ordered):
        following = ordered[(index + 1) % len(ordered)]
        gap = (following - hour) % 24
        if gap == 0:
            continue
        if gap > best_gap:
            best_gap = gap
            best_index = index
    start = ordered[(best_index + 1) % len(ordered)]
    end = ordered[best_index]
    return start, end


def _in_block(hour: int, start: int, end: int, margin: int = 0) -> bool:
    """Is ``hour`` inside the circular block ``start``..``end``?"""
    begin = (start - margin) % 24
    finish = (end + margin) % 24
    length = (finish - begin) % 24
    return ((hour - begin) % 24) <= length


def _circular_distance(hour: int, reference: float) -> float:
    """Shortest distance in hours between ``hour`` and a fractional reference."""
    difference = abs(hour - reference) % 24
    return float(min(difference, 24 - difference))


def _circular_centre(hours: set[int]) -> float:
    """Circular mean of a set of hours, as a fractional hour of day."""
    if not hours:
        return 0.0
    angles = [hour / 24 * 2 * math.pi for hour in hours]
    sine = sum(math.sin(angle) for angle in angles)
    cosine = sum(math.cos(angle) for angle in angles)
    if abs(sine) < 1e-9 and abs(cosine) < 1e-9:
        return float(sum(hours) / len(hours))
    mean = math.atan2(sine, cosine) / (2 * math.pi) * 24
    return mean % 24


def extract_features(
    events: pd.DataFrame,
    *,
    hour_window: timedelta = timedelta(hours=1),
    short_window: timedelta = timedelta(minutes=10),
    min_profile_events: int = 8,
) -> FeatureSet:
    """Build the feature matrix for ``events``.

    The events are sorted by time first; the returned frames share the sorted
    order, and ``event_index`` carries the original position.
    """
    if events.empty:
        raise ValueError("cannot extract features from an empty event frame")

    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)

    stamps: list[datetime] = ordered["ts"].tolist()
    users: list[str] = ordered["user"].fillna("").tolist()
    hosts: list[str] = ordered["host"].fillna("").tolist()
    sources: list[str] = ordered["src_ip"].fillna("").tolist()
    programs: list[str] = ordered["process"].fillna("").tolist()
    parents: list[str] = ordered["parent_process"].fillna("").tolist()
    types: list[str] = ordered["event_type"].fillna("other").tolist()
    destinations: list[str] = ordered["dst_ip"].fillna("").tolist()
    sent_bytes: list[int] = ordered["bytes_sent"].fillna(0).tolist()
    pids: list[object] = ordered["pid"].tolist()
    size = len(ordered)
    ppids: list[object] = ordered["ppid"].tolist() if "ppid" in ordered else [None] * size

    size = len(ordered)
    values: dict[str, np.ndarray] = {
        name: np.zeros(size, dtype=np.float64) for name in FEATURE_NAMES
    }

    user_fail_hour = SlidingWindow(hour_window)
    user_fail_short = SlidingWindow(short_window)
    source_fail_hour = SlidingWindow(hour_window)
    source_fail_short = SlidingWindow(short_window)
    source_users = SlidingWindow(hour_window)
    user_logins = SlidingWindow(hour_window)
    user_login_hosts = SlidingWindow(hour_window)
    user_sudo_failures = SlidingWindow(hour_window)
    program_connections = SlidingWindow(hour_window)
    program_bytes = SlidingWindow(hour_window)

    user_fail_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    program_byte_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    user_hour_profile: dict[str, list[int]] = defaultdict(lambda: [0] * 24)
    user_hour_total: dict[str, int] = defaultdict(int)
    user_hours_seen: dict[str, set[int]] = defaultdict(set)
    user_hour_block: dict[str, tuple[int, int]] = {}

    seen_users: set[str] = set()
    seen_user_host: set[tuple[str, str]] = set()
    seen_user_source: set[tuple[str, str]] = set()
    seen_user_program: set[tuple[str, str]] = set()
    seen_proc_pairs: set[tuple[str, str]] = set()
    seen_program_dest: set[tuple[str, str]] = set()
    proc_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    user_program_counts: dict[tuple[str, str], int] = defaultdict(int)
    program_dest_counts: dict[tuple[str, str], int] = defaultdict(int)

    #: (host, pid) -> when the login was last counted, so a login episode is
    #: counted once even when the source logs both an accept and a session open.
    last_login_seen: dict[tuple[str, object], datetime] = {}

    #: (host, pid) -> executable, so a process event's parent can be named.
    pid_to_executable: dict[tuple[str, object], str] = {}

    for position in range(size):
        ts = stamps[position]
        user = users[position]
        host = hosts[position]
        source = sources[position]
        program = programs[position]
        parent = parents[position]
        event_type = types[position]
        destination = destinations[position]

        is_failure = event_type in _FAILURE_TYPES
        is_login = event_type in _LOGIN_TYPES
        is_sudo_failure = event_type in _SUDO_FAIL_TYPES

        if is_failure:
            if user:
                values["auth_fail_count_1h"][position] = len(user_fail_hour.push(user, ts))
                values["auth_fail_count_10m"][position] = len(user_fail_short.push(user, ts))
                fail_hour = values["auth_fail_count_1h"][position]
                values["auth_fail_count_zscore"][position] = user_fail_stats[user].zscore(fail_hour)
                user_fail_stats[user].update(fail_hour)
            if source:
                values["src_fail_count_1h"][position] = len(source_fail_hour.push(source, ts))
                values["src_fail_count_10m"][position] = len(source_fail_short.push(source, ts))
                if user:
                    entries = source_users.push(source, ts, user)
                    values["src_distinct_users_1h"][position] = len({value for _, value in entries})
                    values["src_user_entropy_1h"][position] = _entropy(
                        value for _, value in entries
                    )

        if is_login:
            key = (host, pids[position])
            previous = last_login_seen.get(key)
            if previous is None or (ts - previous) > timedelta(seconds=60):
                last_login_seen[key] = ts
                if user:
                    values["login_count_1h"][position] = len(user_logins.push(user, ts))
                    entries = user_login_hosts.push(user, ts, host)
                    values["user_distinct_hosts_1h"][position] = len(
                        {value for _, value in entries}
                    )

        if is_sudo_failure and user:
            values["sudo_fail_count_1h"][position] = len(user_sudo_failures.push(user, ts))

        if event_type == "process_exec":
            if not parent:
                # Resolve the parent executable from the process table so the
                # (parent, child) pair is meaningful.
                ppid = ppids[position]
                if ppid is not None and ppid is not pd.NA:
                    parent = pid_to_executable.get((host, ppid), f"pid:{ppid}")
            pair = (parent, program)
            proc_pair_counts[pair] += 1
            values["proc_pair_is_new"][position] = 0.0 if pair in seen_proc_pairs else 1.0
            seen_proc_pairs.add(pair)
            values["proc_pair_rarity"][position] = -math.log(proc_pair_counts[pair])
            if user:
                key = (user, program)
                user_program_counts[key] += 1
                values["proc_new_for_user"][position] = 0.0 if key in seen_user_program else 1.0
                seen_user_program.add(key)
                values["proc_freq_rarity"][position] = -math.log(user_program_counts[key])
            pid = pids[position]
            if program and pid is not None and pid is not pd.NA:
                pid_to_executable[(host, pid)] = program

        if event_type == "network_conn":
            if program:
                values["net_conn_count_1h"][position] = len(program_connections.push(program, ts))
                total = sum(value for _, value in program_bytes.push(program, ts, sent_bytes[position]))
                values["net_total_bytes_1h"][position] = total
                values["net_bytes_sent_zscore"][position] = program_byte_stats[program].zscore(total)
                program_byte_stats[program].update(total)
            if destination:
                key = (program, destination)
                program_dest_counts[key] += 1
                values["net_new_dest_for_proc"][position] = (
                    0.0 if key in seen_program_dest else 1.0
                )
                seen_program_dest.add(key)
                values["net_dest_rarity"][position] = -math.log(program_dest_counts[key])

        # -- identity and profile features --------------------------------
        if user:
            values["unknown_user"][position] = 0.0 if user in seen_users else 1.0
            seen_users.add(user)

            pair = (user, host)
            values["new_user_host_pair"][position] = 0.0 if pair in seen_user_host else 1.0
            seen_user_host.add(pair)

            if source:
                key = (user, source)
                values["new_src_for_user"][position] = 0.0 if key in seen_user_source else 1.0
                seen_user_source.add(key)

            profile = user_hour_profile[user]
            total_before = user_hour_total[user]
            hour = ts.hour
            if total_before >= min_profile_events:
                block = user_hour_block.get(user)
                if block is None:
                    block = _circular_block(user_hours_seen[user])
                    user_hour_block[user] = block
                values["off_hours"][position] = (
                    0.0 if _in_block(hour, block[0], block[1], margin=2) else 1.0
                )
                profile[hour] += 1
                user_hours_seen[user].add(hour)
                user_hour_block.pop(user, None)
                centre = _circular_centre(user_hours_seen[user])
                values["hour_profile_distance"][position] = _circular_distance(hour, centre)
            else:
                profile[hour] += 1
                user_hours_seen[user].add(hour)
                user_hour_block.pop(user, None)
            user_hour_total[user] = total_before + 1

            values["root_activity"][position] = 1.0 if user == "root" else 0.0

        values["type_auth_failure"][position] = 1.0 if event_type == "auth_failure" else 0.0
        values["type_auth_success"][position] = 1.0 if event_type == "auth_success" else 0.0
        values["type_session"][position] = (
            1.0 if event_type in {"session_open", "session_close"} else 0.0
        )
        values["type_sudo"][position] = 1.0 if event_type.startswith("sudo") else 0.0
        values["type_process"][position] = 1.0 if event_type == "process_exec" else 0.0
        values["type_network"][position] = 1.0 if event_type == "network_conn" else 0.0

    features = pd.DataFrame(values, columns=list(FEATURE_NAMES))

    availability = {
        "auth": float((ordered["event_type"].isin(_FAILURE_TYPES | _LOGIN_TYPES)).mean()),
        "process": float((ordered["event_type"] == "process_exec").mean()),
        "network": float((ordered["event_type"] == "network_conn").mean()),
    }

    return FeatureSet(
        features=features,
        events=ordered.reset_index(drop=True),
        feature_names=FEATURE_NAMES,
        availability=availability,
    )
