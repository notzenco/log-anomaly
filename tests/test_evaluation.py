"""Tests for splits and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from log_anomaly.evaluation.metrics import (
    alert_volume,
    build_episodes,
    classification_metrics,
    episode_metrics,
    per_type_metrics,
    ranking_metrics,
    select_threshold,
)
from log_anomaly.evaluation.splits import time_split


def stamps(days: int = 10) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC"))


def test_time_split_is_ordered_disjoint_and_complete() -> None:
    series = stamps()
    split = time_split(series, train_fraction=0.6, validation_fraction=0.2)
    assert not (split.train & split.validation).any()
    assert not (split.validation & split.test).any()
    assert (split.train | split.validation | split.test).all()
    assert series[split.train].max() < series[split.validation].min()
    assert series[split.validation].max() < series[split.test].min()
    assert split.sizes() == {"train": 144, "validation": 48, "test": 48}


def test_time_split_rejects_fractions_without_a_test_window() -> None:
    with pytest.raises(ValueError):
        time_split(stamps(), train_fraction=0.9, validation_fraction=0.2)


def test_classification_metrics_on_a_known_case() -> None:
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.1, 0.7, 0.2])
    metrics = classification_metrics(labels, scores, threshold=0.5)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_ranking_metrics_report_the_base_rate() -> None:
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    metrics = ranking_metrics(labels, scores)
    assert metrics["base_rate"] == 0.25
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_select_threshold_separates_and_needs_positives() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(labels, scores)
    assert 0.2 < threshold <= 0.9
    with pytest.raises(ValueError):
        select_threshold(np.zeros(4, dtype=int), scores)


def test_alert_volume_counts_per_day() -> None:
    series = stamps(days=2)
    scores = np.linspace(0, 1, len(series))
    volume = alert_volume(scores, series, threshold=0.5)
    assert volume["alerts"] == 24
    assert volume["alerts_per_day"] == pytest.approx(24 / (47 / 24), rel=1e-6)


def test_per_type_metrics_give_recall_per_family() -> None:
    labels = np.array([1, 1, 1, 1, 0])
    types = np.array(["brute_force", "brute_force", "rare_process", "rare_process", "benign"])
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.9])
    per_type = per_type_metrics(labels, scores, types, threshold=0.5)
    assert per_type["brute_force"] == {"support": 2, "detected": 1, "recall": 0.5}
    assert per_type["rare_process"]["recall"] == 1.0


def test_episodes_group_bursts_and_measure_detection() -> None:
    series = stamps(days=2)
    types = np.array(["brute_force"] * 3 + ["brute_force"] + ["background_scan"] * len(series[4:]))
    users = np.array(["alice"] * len(series), dtype=object)
    sources = np.array(["1.2.3.4"] * len(series), dtype=object)
    attack = np.zeros(len(series), dtype=bool)
    attack[[0, 1, 2, 10]] = True
    episodes = build_episodes(types, users, sources, series, attack_mask=attack)
    assert len(episodes) == 2  # hour 0-2 burst, then the hour-10 singleton
    scores = np.zeros(len(series))
    scores[2] = 1.0
    metrics = episode_metrics(episodes, scores, threshold=0.5, timestamps=series)
    assert metrics["episodes_detected"] == 1
    assert metrics["episode_detection_rate"] == 0.5
    assert metrics["median_detection_delay_s"] == 7200.0
