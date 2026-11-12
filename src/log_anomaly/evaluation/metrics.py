"""Metrics for anomaly detection.

Two things are measured that a standard classifier report would miss:

* **Alert volume per day**, because an analyst's workload, not a metric,
  decides whether a detector is usable.
* **Episode detection rate**, because one brute-force campaign is one
  incident, not sixty positive rows, and a detector that catches the last
  attempt of a campaign has still failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

#: Attack families whose "actor" is the source address rather than the account.
_SOURCE_ACTOR_TYPES = frozenset({"brute_force", "credential_stuffing", "background_scan"})

#: Attack events within this gap of each other belong to the same episode.
EPISODE_GAP = timedelta(hours=1)

#: Background scanning is continuous, so its episodes are grouped more loosely.
EPISODE_GAP_BY_TYPE: dict[str, timedelta] = {"background_scan": timedelta(hours=6)}


@dataclass(frozen=True)
class Episode:
    """One attack episode: a burst of related attack events."""

    attack_type: str
    actor: str
    start: pd.Timestamp
    end: pd.Timestamp
    indices: np.ndarray

    @property
    def size(self) -> int:
        return len(self.indices)


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    beta: float = 1.0,
) -> float:
    """Pick the threshold maximising F-beta on the given (validation) data.

    ``beta`` above 1 weights recall more heavily, which is the right
    orientation for security triage.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        raise ValueError("cannot select a threshold without positive examples")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = beta**2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denominator > 0, (1 + beta**2) * precision * recall / denominator, 0.0)
    # the last precision/recall pair corresponds to no threshold at all
    best = int(np.nanargmax(fbeta[:-1])) if len(fbeta) > 1 else 0
    return float(thresholds[best])


def select_budget_threshold(
    scores: np.ndarray,
    timestamps: pd.Series,
    budget_per_day: float,
) -> float:
    """Threshold that keeps the alert rate at or below ``budget_per_day``.

    The threshold is the quantile of the score distribution that leaves at
    most ``budget_per_day`` events above it. This is the operating point an
    analyst can actually work with.
    """
    scores = np.asarray(scores, dtype=float)
    stamps = pd.to_datetime(pd.Series(timestamps), utc=True)
    if len(scores) == 0:
        raise ValueError("cannot pick a budget threshold without scores")
    span_days = max((stamps.max() - stamps.min()).total_seconds() / 86400, 1e-9)
    allowance = budget_per_day * span_days
    keep = min(allowance, len(scores))
    if keep <= 0:
        return float(np.max(scores)) + 1.0
    threshold = float(np.quantile(scores, 1.0 - keep / len(scores)))
    return threshold


def _safe(value: float) -> float:
    return 0.0 if value is None or not np.isfinite(value) else float(value)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Precision, recall and F-scores at one threshold."""
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta = 5.0
    f5_denominator = beta**2 * precision + recall
    f5 = (
        (1 + beta**2) * precision * recall / f5_denominator
        if f5_denominator
        else 0.0
    )
    return {
        "precision": _safe(precision),
        "recall": _safe(recall),
        "f1": _safe(f1),
        "f5": _safe(f5),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Base-rate aware ranking metrics."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out: dict[str, float] = {"base_rate": float(labels.mean()) if len(labels) else 0.0}
    if labels.sum() == 0:
        out.update({"pr_auc": 0.0, "roc_auc": 0.0})
        return out
    out["pr_auc"] = _safe(average_precision_score(labels, scores))
    out["roc_auc"] = _safe(roc_auc_score(labels, scores))
    return out


def alert_volume(
    scores: np.ndarray,
    timestamps: pd.Series,
    threshold: float,
    users: np.ndarray | None = None,
    sources: np.ndarray | None = None,
    hosts: np.ndarray | None = None,
) -> dict[str, float]:
    """How much work the threshold creates: alerts per day, entities per day."""
    scores = np.asarray(scores, dtype=float)
    stamps = pd.to_datetime(pd.Series(timestamps), utc=True)
    predicted = scores >= threshold
    alerts = int(predicted.sum())
    span_days = (stamps.max() - stamps.min()).total_seconds() / 86400
    days = max(span_days, 1e-9)

    entities: set[tuple[str, str, str]] = set()
    if users is not None and sources is not None and hosts is not None:
        for index in np.flatnonzero(predicted):
            entities.add(
                (
                    str(users[index]),
                    str(sources[index]),
                    str(hosts[index]),
                )
            )
    return {
        "alerts": alerts,
        "days": float(span_days),
        "alerts_per_day": float(alerts / days),
        "alerted_entities": len(entities),
        "alerted_entities_per_day": float(len(entities) / days),
    }


def per_type_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    attack_types: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Recall broken down by attack family.

    Per-family *precision* is deliberately not reported: a family's events are
    exactly its labels, so the only alerts that can fall on a family are
    already true positives and the number would be 1 by construction.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    types = np.asarray(attack_types, dtype=object)

    out: dict[str, dict[str, float]] = {}
    predicted = scores >= threshold
    for attack_type in sorted(set(types[labels == 1])):
        mask = types == attack_type
        positives = mask & (labels == 1)
        true_positive = int(np.sum(positives & predicted))
        false_negative = int(np.sum(positives & ~predicted))
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        out[str(attack_type)] = {
            "support": int(positives.sum()),
            "detected": true_positive,
            "recall": _safe(recall),
        }
    return out


def build_episodes(
    attack_types: np.ndarray,
    users: np.ndarray,
    sources: np.ndarray,
    timestamps: pd.Series,
    *,
    attack_mask: np.ndarray | None = None,
    gap: timedelta = EPISODE_GAP,
) -> list[Episode]:
    """Group attack events into episodes.

    Two attack events belong to the same episode when they share a type and
    an actor, and are less than ``gap`` apart. Only attack events take part,
    so pass ``attack_mask`` to exclude benign traffic.
    """
    types = np.asarray(attack_types, dtype=object)
    users = np.asarray(users, dtype=object)
    sources = np.asarray(sources, dtype=object)
    stamps = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)

    if attack_mask is None:
        attack_mask = np.ones(len(types), dtype=bool)
    attack_mask = np.asarray(attack_mask, dtype=bool)

    groups: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(attack_mask & np.isin(types, list(_SOURCE_ACTOR_TYPES))):
        groups.setdefault((str(types[index]), str(sources[index])), []).append(int(index))
    for index in np.flatnonzero(attack_mask & ~np.isin(types, list(_SOURCE_ACTOR_TYPES))):
        groups.setdefault((str(types[index]), str(users[index])), []).append(int(index))

    episodes: list[Episode] = []
    for (attack_type, actor), indices in groups.items():
        indices.sort(key=lambda item: stamps.iloc[item])
        type_gap = EPISODE_GAP_BY_TYPE.get(attack_type, gap)
        current = [indices[0]]
        for index in indices[1:]:
            if stamps.iloc[index] - stamps.iloc[current[-1]] > type_gap:
                episodes.append(_make_episode(attack_type, actor, current, stamps))
                current = [index]
            else:
                current.append(index)
        episodes.append(_make_episode(attack_type, actor, current, stamps))
    return episodes


def _make_episode(
    attack_type: str, actor: str, indices: list[int], stamps: pd.Series
) -> Episode:
    array = np.array(indices, dtype=int)
    return Episode(
        attack_type=attack_type,
        actor=actor,
        start=stamps.iloc[indices[0]],
        end=stamps.iloc[indices[-1]],
        indices=array,
    )


def episode_metrics(
    episodes: list[Episode],
    scores: np.ndarray,
    threshold: float,
    timestamps: pd.Series,
) -> dict[str, float]:
    """How many attack episodes were detected at all, and how quickly."""
    scores = np.asarray(scores, dtype=float)
    stamps = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)
    detected = 0
    delays: list[float] = []
    per_type: dict[str, dict[str, int]] = {}
    for episode in episodes:
        hits = episode.indices[scores[episode.indices] >= threshold]
        bucket = per_type.setdefault(episode.attack_type, {"episodes": 0, "detected": 0})
        bucket["episodes"] += 1
        if len(hits):
            detected += 1
            bucket["detected"] += 1
            delays.append((stamps.iloc[hits[0]] - episode.start).total_seconds())
    total = len(episodes)
    delay_summary = {
        "median_detection_delay_s": float(np.median(delays)) if delays else 0.0,
        "max_detection_delay_s": float(np.max(delays)) if delays else 0.0,
    }
    return {
        "episodes": total,
        "episodes_detected": detected,
        "episode_detection_rate": float(detected / total) if total else 0.0,
        **delay_summary,
        "per_type": {name: dict(values) for name, values in sorted(per_type.items())},
    }
