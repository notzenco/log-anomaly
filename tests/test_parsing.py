"""Tests for the log parsers."""

from __future__ import annotations

import pytest

from log_anomaly.parsing import detect_format, parse_syslog_lines
from log_anomaly.parsing.proxifier import parse_proxifier_lines


def parse_one(line: str, **kwargs):
    frame, stats = parse_syslog_lines([line], **kwargs)
    assert stats["parsed"] == 1
    return frame.iloc[0]


def test_failed_password_is_an_auth_failure() -> None:
    row = parse_one("Jun 14 15:16:01 web01 sshd[19939]: Failed password for invalid user admin from 203.0.113.7 port 44321 ssh2")
    assert row["event_type"] == "auth_failure"
    assert row["user"] == "admin"
    assert row["src_ip"] == "203.0.113.7"
    assert row["host"] == "web01"
    assert row["program"] == "sshd"
    assert row["pid"] == 19939


def test_accepted_password_is_an_auth_success() -> None:
    row = parse_one("Dec 10 09:32:20 gateway sshd[24680]: Accepted password for alice from 198.51.100.4 port 49116 ssh2")
    assert row["event_type"] == "auth_success"
    assert row["user"] == "alice"
    assert row["src_ip"] == "198.51.100.4"


def test_pam_authentication_failure_extracts_user_and_source() -> None:
    row = parse_one(
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; "
        "logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4  user=root"
    )
    assert row["event_type"] == "auth_failure"
    assert row["user"] == "root"
    assert row["src_ip"] == "218.188.2.4"
    assert row["program"] == "sshd"


def test_invalid_user_is_an_auth_failure() -> None:
    row = parse_one("Dec 10 06:55:46 LabSZ sshd[24200]: Invalid user webmaster from 173.234.31.186")
    assert row["event_type"] == "auth_failure"
    assert row["user"] == "webmaster"
    assert row["src_ip"] == "173.234.31.186"


def test_su_session_open_and_close() -> None:
    opened = parse_one("Jun 15 04:06:18 combo su(pam_unix)[21416]: session opened for user cyrus by (uid=0)")
    assert opened["event_type"] == "session_open"
    assert opened["user"] == "cyrus"
    closed = parse_one("Jun 15 04:06:19 combo su(pam_unix)[21416]: session closed for user cyrus")
    assert closed["event_type"] == "session_close"


def test_sudo_events_are_classified() -> None:
    success = parse_one("Jun 15 04:06:18 combo sudo[999]: alice : TTY=pts/0 ; PWD=/home/alice ; USER=root ; COMMAND=/bin/bash")
    assert success["event_type"] == "sudo_success"
    assert success["user"] == "alice"
    failure = parse_one(
        "Jun 15 04:06:18 combo sudo: pam_unix(sudo:auth): authentication failure; "
        "logname=alice uid=1000 euid=0 tty=pts/0 ruser=alice  user=root"
    )
    assert failure["event_type"] == "sudo_failure"
    # the actor (logname) is the account that tried to escalate, not the target
    assert failure["user"] == "alice"


def test_process_exec_event_is_parsed() -> None:
    row = parse_one("2024-06-14T15:16:01Z web01 process[900]: user=alice pid=900 ppid=1 exe=/usr/bin/python3")
    assert row["event_type"] == "process_exec"
    assert row["user"] == "alice"
    assert row["process"] == "/usr/bin/python3"


def test_network_connection_event_is_parsed() -> None:
    row = parse_one(
        "2024-06-14T15:16:05Z web01 conn[901]: user=alice exe=/usr/bin/curl dst=203.0.113.5:443 "
        "bytes_sent=1024 bytes_received=2048"
    )
    assert row["event_type"] == "network_conn"
    assert row["dst_ip"] == "203.0.113.5"
    assert row["dst_port"] == 443
    assert row["bytes_sent"] == 1024
    assert row["bytes_received"] == 2048


def test_year_rolls_over_at_the_year_boundary() -> None:
    frame, _ = parse_syslog_lines(
        [
            "Dec 31 23:59:58 host sshd[1]: Accepted password for alice from 10.0.0.1 port 1 ssh2",
            "Jan 01 00:00:02 host sshd[2]: Accepted password for alice from 10.0.0.1 port 1 ssh2",
        ],
        anchor_year=2020,
    )
    assert frame["ts"].iloc[0].year == 2020
    assert frame["ts"].iloc[1].year == 2021


def test_out_of_order_lines_do_not_bump_the_year() -> None:
    frame, _ = parse_syslog_lines(
        [
            "Jan 02 10:00:00 host sshd[1]: Accepted password for alice from 10.0.0.1 port 1 ssh2",
            "Jan 02 09:00:00 host sshd[2]: Accepted password for alice from 10.0.0.1 port 1 ssh2",
        ],
        anchor_year=2020,
    )
    assert frame["ts"].iloc[1].year == 2020


def test_malformed_lines_are_counted_as_rejected() -> None:
    frame, stats = parse_syslog_lines(["this is not a log line", ""])
    assert stats["parsed"] == 0
    assert stats["rejected"] == 1
    assert frame.empty


def test_proxifier_line_is_parsed() -> None:
    frame, stats = parse_proxifier_lines(
        [
            "[10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 11971 bytes (11.6 KB) sent, "
            "18220 bytes (17.7 KB) received, lifetime 00:09",
            "[07.27 10:32:29] chrome.exe *64 - manhua.163.com:443 open through proxy p.example:80 HTTPS",
        ],
        anchor_year=2017,
    )
    assert stats["parsed"] == 2
    first = frame.iloc[0]
    assert first["process"] == "chrome.exe"
    assert first["dst_ip"] == "proxy.cse.cuhk.edu.hk"
    assert first["dst_port"] == 5070
    assert first["bytes_sent"] == 11971
    assert first["bytes_received"] == 18220
    assert first["duration_s"] == 9
    second = frame.iloc[1]
    assert second["pid"] == 64
    assert second["dst_port"] == 443


def test_proxifier_status_messages_are_rejected() -> None:
    frame, stats = parse_proxifier_lines(
        ["[07.26 16:40:55] Warning : Connections do not originate from the applications."]
    )
    assert stats["parsed"] == 0
    assert stats["rejected"] == 1
    assert frame.empty


def test_detect_format(tmp_path) -> None:
    syslog_file = tmp_path / "a.log"
    syslog_file.write_text("Jun 14 15:16:01 host sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2\n")
    proxy_file = tmp_path / "b.log"
    proxy_file.write_text("[10.30 16:49:06] chrome.exe - example.com:80 close, 0 bytes sent, 0 bytes received, lifetime <1 sec\n")
    assert detect_format(syslog_file) == "syslog"
    assert detect_format(proxy_file) == "proxifier"


@pytest.mark.parametrize("line", ["", "   ", "\n"])
def test_blank_lines_are_ignored(line: str) -> None:
    _, stats = parse_syslog_lines([line])
    assert stats["lines"] == 0
