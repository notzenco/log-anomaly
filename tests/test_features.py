"""Tests for feature engineering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from log_anomaly.features import FEATURE_NAMES, extract_features
from log_anomaly.parsing.schema import EVENT_DTYPES

BASE = datetime(2024, 5, 6, 9, 0, tzinfo=UTC)


def make_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in EVENT_DTYPES.items()})
    for index, row in enumerate(rows):
        for key, value in row.items():
            frame.loc[index, key] = value
    frame = frame.astype(EVENT_DTYPES)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def login(offset_minutes: int, user: str, host: str, source: str, **extra) -> dict[str, object]:
    row: dict[str, object] = {
        "event_index": 0,
        "ts": BASE + timedelta(minutes=offset_minutes),
        "host": host,
        "program": "sshd",
        "pid": 1000 + offset_minutes,
        "event_type": "auth_success",
        "user": user,
        "src_ip": source,
        "message": "",
    }
    row.update(extra)
    return row


def failure(offset_minutes: int, user: str, host: str, source: str, **extra) -> dict[str, object]:
    row: dict[str, object] = {
        "event_index": 0,
        "ts": BASE + timedelta(minutes=offset_minutes),
        "host": host,
        "program": "sshd",
        "pid": 2000 + offset_minutes,
        "event_type": "auth_failure",
        "user": user,
        "src_ip": source,
        "message": "",
    }
    row.update(extra)
    return row


def test_features_have_one_row_per_event() -> None:
    events = make_events([login(0, "alice", "web01", "10.0.0.1"), login(10, "alice", "web01", "10.0.0.1")])
    result = extract_features(events)
    assert len(result.features) == 2
    assert list(result.features.columns) == list(FEATURE_NAMES)
    assert result.events["ts"].is_monotonic_increasing


def test_empty_frame_is_rejected() -> None:
    events = make_events([])
    with pytest.raises(ValueError):
        extract_features(events)


def test_failure_burst_accumulates_in_the_short_window() -> None:
    rows = [failure(i, "alice", "web01", "10.0.0.9") for i in range(6)]
    result = extract_features(make_events(rows))
    counts = result.features["auth_fail_count_10m"].tolist()
    assert counts == [1, 2, 3, 4, 5, 6]


def test_failure_burst_expires_after_the_window() -> None:
    rows = [failure(0, "alice", "web01", "10.0.0.9"), failure(30, "alice", "web01", "10.0.0.9")]
    result = extract_features(make_events(rows))
    assert result.features["auth_fail_count_1h"].tolist() == [1, 2]
    assert result.features["auth_fail_count_10m"].tolist() == [1, 1]


def test_new_user_host_pair_is_flagged_once() -> None:
    rows = [
        login(0, "alice", "web01", "10.0.0.1"),
        login(60, "alice", "web01", "10.0.0.1"),
        login(120, "alice", "web02", "10.0.0.1"),
    ]
    result = extract_features(make_events(rows))
    assert result.features["new_user_host_pair"].tolist() == [1.0, 0.0, 1.0]


def test_unknown_user_is_flagged_once() -> None:
    rows = [login(0, "alice", "web01", "10.0.0.1"), login(60, "alice", "web01", "10.0.0.1")]
    result = extract_features(make_events(rows))
    assert result.features["unknown_user"].tolist() == [1.0, 0.0]


def test_source_spraying_many_users_raises_distinct_user_count() -> None:
    rows = [failure(i, f"user{i}", "web01", "10.0.0.9") for i in range(5)]
    result = extract_features(make_events(rows))
    assert result.features["src_distinct_users_1h"].tolist() == [1, 2, 3, 4, 5]
    assert result.features["src_user_entropy_1h"].iloc[-1] > 0


def test_off_hours_needs_a_profile_before_it_fires() -> None:
    # ten logins spread over 09:00-18:00, then one at 02:00 the next day
    early = [login(i * 60, "alice", "web01", "10.0.0.1") for i in range(10)]
    night = login(17 * 60, "alice", "web01", "10.0.0.1")
    result = extract_features(make_events([*early, night]))
    assert result.features["off_hours"].iloc[:10].tolist() == [0.0] * 10
    assert result.features["off_hours"].iloc[10] == 1.0


def test_night_shift_user_is_not_flagged_at_night() -> None:
    # a worker whose activity wraps around midnight: 22:00-03:00
    rows = []
    for cycle in range(2):
        for index in range(6):
            offset_hours = 13 + index + 24 * cycle
            rows.append(login(offset_hours * 60, "alice", "web01", "10.0.0.1"))
    rows.append(login(51 * 60, "alice", "web01", "10.0.0.1"))  # 12:00 two days later
    result = extract_features(make_events(rows))
    flags = result.features["off_hours"]
    stamps = result.events["ts"].tolist()

    at_night = stamps.index(BASE + timedelta(hours=41))  # 02:00
    at_noon = stamps.index(BASE + timedelta(hours=51))  # 12:00
    assert flags.iloc[at_night] == 0.0, "a 02:00 login is normal for this account"
    assert flags.iloc[at_noon] == 1.0, "a 12:00 login is not"


def test_erratic_account_has_no_off_hours() -> None:
    # an account active in 19 distinct hours (00:00-18:00) has no normal
    # hour to deviate from, so a 21:00 login must not fire off_hours
    rows = [login((-9 + i) * 60, "alice", "web01", "10.0.0.1") for i in range(19)]
    rows.append(login(12 * 60, "alice", "web01", "10.0.0.1"))  # 21:00
    result = extract_features(make_events(rows))
    assert result.features["off_hours"].iloc[-1] == 0.0


def test_rare_process_pair_is_flagged_and_parent_resolved() -> None:
    rows = []
    for index in range(3):
        rows.append(
            {
                "event_index": 0,
                "ts": BASE + timedelta(minutes=index),
                "host": "web01",
                "program": "process",
                "pid": 500 + index,
                "event_type": "process_exec",
                "user": "alice",
                "process": "/bin/bash",
                "ppid": 400,
                "message": "",
            }
        )
    rows.append(
        {
            "event_index": 0,
            "ts": BASE + timedelta(minutes=10),
            "host": "web01",
            "program": "process",
            "pid": 600,
            "event_type": "process_exec",
            "user": "alice",
            "process": "/bin/bash",
            "ppid": 400,
            "message": "",
        }
    )
    rows.append(
        {
            "event_index": 0,
            "ts": BASE + timedelta(minutes=20),
            "host": "web01",
            "program": "process",
            "pid": 700,
            "event_type": "process_exec",
            "user": "alice",
            "process": "/usr/bin/nc",
            "ppid": 500,
            "message": "",
        }
    )
    result = extract_features(make_events(rows))
    is_new = result.features["proc_pair_is_new"].tolist()
    assert is_new[0] == 1.0
    assert is_new[1] == 0.0
    # the last event runs a program under a parent never seen with it
    assert is_new[-1] == 1.0


def test_features_are_causal() -> None:
    """Extracting from a prefix must reproduce the full-run features exactly."""
    rows = [login(i * 7, f"user{i % 4}", "web01", "10.0.0.1") for i in range(30)]
    rows += [failure(i * 3, "user1", "web01", "10.0.0.9") for i in range(20)]
    rows.sort(key=lambda row: row["ts"])
    events = make_events(rows)
    full = extract_features(events)
    prefix = extract_features(events.iloc[:40])
    pd.testing.assert_frame_equal(
        full.features.iloc[:40].reset_index(drop=True),
        prefix.features.reset_index(drop=True),
    )


def test_availability_is_reported_per_group() -> None:
    rows = [login(0, "alice", "web01", "10.0.0.1")]
    rows.append(
        {
            "event_index": 0,
            "ts": BASE + timedelta(minutes=1),
            "host": "web01",
            "program": "conn",
            "pid": 900,
            "event_type": "network_conn",
            "user": "alice",
            "process": "/usr/bin/curl",
            "dst_ip": "203.0.113.5",
            "dst_port": 443,
            "bytes_sent": 1000,
            "bytes_received": 2000,
            "message": "",
        }
    )
    result = extract_features(make_events(rows))
    assert result.availability["network"] == 0.5
    assert result.availability["process"] == 0.0
