"""Tests for the synthetic corpus generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from log_anomaly.data.synthetic import (
    ATTACK_FAMILIES,
    SyntheticConfig,
    read_labels,
    write_corpus,
)
from log_anomaly.parsing import parse_file

SMALL = SyntheticConfig(days=3, hosts=3, users=8, seed=7, start="2024-05-06")


def _generate(tmp_path: Path, config: SyntheticConfig = SMALL):
    log = tmp_path / "synth.log"
    labels = tmp_path / "labels.csv"
    summary = write_corpus(config, log, labels, tmp_path / "meta.json")
    return log, labels, summary


def test_corpus_is_deterministic(tmp_path: Path) -> None:
    first, _, _ = _generate(tmp_path / "a")
    second, _, _ = _generate(tmp_path / "b")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_log_file_is_marked_synthetic(tmp_path: Path) -> None:
    log, _, _ = _generate(tmp_path)
    assert log.read_text(encoding="utf-8").startswith("# SYNTHETIC LOG")


def test_labels_align_with_parsed_events(tmp_path: Path) -> None:
    log, labels, summary = _generate(tmp_path)
    frame, stats = parse_file(log, anchor_year=SMALL.start_datetime.year)
    assert stats["parsed"] == summary["events"]
    assert stats["rejected"] == 1  # the synthetic header is not an event

    rows = read_labels(labels)
    assert len(rows) == len(frame)
    assert [int(row["event_index"]) for row in rows] == frame["event_index"].tolist()


def test_every_attack_family_is_generated(tmp_path: Path) -> None:
    _, labels, summary = _generate(tmp_path)
    produced = set(summary["attack_counts"])  # type: ignore[arg-type]
    assert set(ATTACK_FAMILIES) <= produced


def test_process_events_are_not_labelled_as_attacks_except_rare_process(tmp_path: Path) -> None:
    _, labels, _ = _generate(tmp_path)
    frame = pd.DataFrame(rows := read_labels(labels))
    assert rows
    processes = frame[frame["event_type"] == "process_exec"]
    assert set(processes["attack_type"]) <= {"benign", "rare_process"}


def test_default_corpus_has_a_low_base_rate(tmp_path: Path) -> None:
    """The shipped configuration must be in the regime where precision matters."""
    _, _, summary = _generate(tmp_path, SyntheticConfig())
    assert 0 < summary["base_rate"] < 0.05
    assert summary["events"] > 10_000


def test_events_are_chronologically_ordered(tmp_path: Path) -> None:
    log, _, _ = _generate(tmp_path)
    frame, _ = parse_file(log, anchor_year=SMALL.start_datetime.year)
    assert frame["ts"].is_monotonic_increasing


def test_injected_brute_force_is_bursty_and_single_sourced(tmp_path: Path) -> None:
    _, labels, _ = _generate(tmp_path)
    frame = pd.DataFrame(read_labels(labels))
    brute = frame[frame["attack_type"] == "brute_force"]
    assert len(brute) >= SMALL.brute_force_attempts
    # every brute force campaign targets one account from one source address
    per_source = brute.groupby("src_ip")["user"].nunique()
    assert (per_source == 1).all()
