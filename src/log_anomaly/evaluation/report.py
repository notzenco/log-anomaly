"""The evaluation driver: split, fit, pick a threshold, measure, report.

The protocol is deliberately boring and fixed in advance:

1. cut the corpus into train / validation / test **by time**;
2. fit the model on the training window only;
3. choose the operating threshold on the validation window;
4. apply that threshold, unchanged, to the test window;
5. report the base rate next to every precision number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from log_anomaly.evaluation.metrics import (
    alert_volume,
    build_episodes,
    classification_metrics,
    episode_metrics,
    per_type_metrics,
    ranking_metrics,
    select_budget_threshold,
    select_threshold,
)
from log_anomaly.evaluation.splits import time_split
from log_anomaly.features import FEATURE_NAMES, FeatureSet, extract_features
from log_anomaly.models import MODEL_NAMES, TrainedModel, fit_model


@dataclass
class EvaluationResult:
    """Everything one model produced on one corpus.

    The ``test_*`` arrays are kept so callers can draw curves from the exact
    scores that produced the metrics, instead of refitting the model.
    """

    model: str
    corpus: str
    metrics: dict[str, Any]
    feature_names: tuple[str, ...]
    thresholds: dict[str, float]
    test_scores: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    test_types: np.ndarray | None = None
    test_timestamps: pd.Series | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "corpus": self.corpus,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
        }


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Read a labels CSV into a frame indexed by ``event_index``."""
    frame = pd.read_csv(labels_path)
    required = {"event_index", "label", "attack_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"labels file is missing columns: {sorted(missing)}")
    frame = frame.set_index("event_index")
    frame["label"] = frame["label"].astype(int)
    return frame


def align_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Align a label frame to the rows of an event frame, by ``event_index``."""
    aligned = labels.reindex(events["event_index"].to_numpy())
    if aligned["label"].isna().any():
        raise ValueError("labels do not cover every event in the log")
    return aligned.reset_index(drop=True)


def evaluate_model(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    model_name: str,
    corpus: str = "corpus",
    feature_set: FeatureSet | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    feature_names: Sequence[str] = FEATURE_NAMES,
    exclude_types: Sequence[str] = (),
    clean_training: bool = False,
    alert_budget: float = 20.0,
) -> EvaluationResult:
    """Evaluate one model with the time-based protocol.

    ``exclude_types`` drops event types from the *measurement* only: they are
    still in the training data the model sees, but they do not count towards
    precision or recall. The intended use is to take known, constant
    background scanning out of the measurement, the way an operations team
    filters it upstream.

    ``clean_training`` additionally removes attack events from the training
    window. That is an oracle setting no deployment has, and it exists only
    to measure how much contamination costs.
    """
    if feature_set is None:
        feature_set = extract_features(events)
    features = feature_set.features
    ordered = feature_set.events
    aligned = align_labels(ordered, labels)

    truth = aligned["label"].to_numpy()
    types = aligned["attack_type"].to_numpy()
    users = ordered["user"].fillna("").to_numpy()
    sources = ordered["src_ip"].fillna("").to_numpy()
    hosts = ordered["host"].fillna("").to_numpy()
    stamps = ordered["ts"]

    split = time_split(
        stamps, train_fraction=train_fraction, validation_fraction=validation_fraction
    )

    matrix = features.to_numpy(dtype=np.float64)

    train_mask = split.train.copy()
    if clean_training:
        train_mask = train_mask & (truth == 0)

    trained: TrainedModel = fit_model(
        model_name,
        matrix[train_mask],
        feature_names,
        metadata={
            "corpus": corpus,
            "model": model_name,
            "split": split.boundaries,
            "clean_training": clean_training,
            "excluded_types": list(exclude_types),
        },
    )
    scores = trained.score(matrix)

    measured = ~np.isin(types, list(exclude_types))
    validation_mask = split.validation & measured
    test_mask = split.test & measured

    validation_labels = truth[validation_mask]
    validation_scores = scores[validation_mask]
    f1_threshold = select_threshold(validation_labels, validation_scores, beta=1.0)
    f5_threshold = select_threshold(validation_labels, validation_scores, beta=5.0)
    budget_threshold = select_budget_threshold(
        validation_scores, stamps[validation_mask], alert_budget
    )

    test_labels = truth[test_mask]
    test_scores = scores[test_mask]
    test_stamps = stamps[test_mask]
    test_users = users[test_mask]
    test_sources = sources[test_mask]
    test_hosts = hosts[test_mask]
    test_types = types[test_mask]

    metrics: dict[str, Any] = {
        "split": split.boundaries,
        "split_sizes": split.sizes(),
        "availability": feature_set.availability,
        "excluded_types": list(exclude_types),
        "clean_training": clean_training,
        "ranking": ranking_metrics(test_labels, test_scores),
        "validation": ranking_metrics(validation_labels, validation_scores),
        "operating_points": {},
        "per_attack_type": {},
    }

    for label, threshold in (
        ("f1", f1_threshold),
        ("f5", f5_threshold),
        ("budget", budget_threshold),
    ):
        point = classification_metrics(test_labels, test_scores, threshold)
        point.update(
            alert_volume(
                test_scores,
                test_stamps,
                threshold,
                users=test_users,
                sources=test_sources,
                hosts=test_hosts,
            )
        )
        metrics["operating_points"][label] = point

    metrics["per_attack_type"] = per_type_metrics(
        test_labels, test_scores, test_types, f1_threshold
    )

    test_episodes = build_episodes(
        test_types,
        test_users,
        test_sources,
        test_stamps,
        attack_mask=test_labels == 1,
    )
    metrics["episodes"] = episode_metrics(
        test_episodes, test_scores, f1_threshold, test_stamps
    )

    return EvaluationResult(
        model=model_name,
        corpus=corpus,
        metrics=metrics,
        feature_names=tuple(feature_names),
        thresholds={"f1": f1_threshold, "f5": f5_threshold, "budget": budget_threshold},
        test_scores=test_scores,
        test_labels=test_labels,
        test_types=test_types,
        test_timestamps=test_stamps.reset_index(drop=True),
    )


def evaluate_all(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    corpus: str = "corpus",
    models: Sequence[str] = MODEL_NAMES,
    feature_set: FeatureSet | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    exclude_types: Sequence[str] = (),
    clean_training: bool = False,
    alert_budget: float = 20.0,
) -> list[EvaluationResult]:
    """Evaluate every model on the same split."""
    if feature_set is None:
        feature_set = extract_features(events)
    results = []
    for name in models:
        results.append(
            evaluate_model(
                events,
                labels,
                model_name=name,
                corpus=corpus,
                feature_set=feature_set,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                exclude_types=exclude_types,
                clean_training=clean_training,
                alert_budget=alert_budget,
            )
        )
    return results


def results_to_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    """Flatten results into one row per model for reporting."""
    rows = []
    for result in results:
        ranking = result.metrics["ranking"]
        f1_point = result.metrics["operating_points"]["f1"]
        f5_point = result.metrics["operating_points"]["f5"]
        budget_point = result.metrics["operating_points"]["budget"]
        episodes = result.metrics["episodes"]
        rows.append(
            {
                "corpus": result.corpus,
                "model": result.model,
                "base_rate": ranking["base_rate"],
                "pr_auc": ranking["pr_auc"],
                "roc_auc": ranking["roc_auc"],
                "precision": f1_point["precision"],
                "recall": f1_point["recall"],
                "f1": f1_point["f1"],
                "f5_precision": f5_point["precision"],
                "f5_recall": f5_point["recall"],
                "alerts_per_day": f1_point["alerts_per_day"],
                "alerts_per_day_f5": f5_point["alerts_per_day"],
                "budget_precision": budget_point["precision"],
                "budget_recall": budget_point["recall"],
                "alerts_per_day_budget": budget_point["alerts_per_day"],
                "episode_detection_rate": episodes["episode_detection_rate"],
                "episodes": episodes["episodes"],
            }
        )
    return pd.DataFrame(rows)
