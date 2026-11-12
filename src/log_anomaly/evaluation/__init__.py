"""Evaluation: time-based splits, base-rate aware metrics and reporting."""

from __future__ import annotations

from log_anomaly.evaluation.metrics import (
    Episode,
    alert_volume,
    build_episodes,
    classification_metrics,
    episode_metrics,
    per_type_metrics,
    ranking_metrics,
    select_threshold,
)
from log_anomaly.evaluation.report import (
    EvaluationResult,
    align_labels,
    evaluate_all,
    evaluate_model,
    load_labels,
    results_to_frame,
)
from log_anomaly.evaluation.splits import Split, time_split

__all__ = [
    "Episode",
    "EvaluationResult",
    "Split",
    "align_labels",
    "alert_volume",
    "build_episodes",
    "classification_metrics",
    "episode_metrics",
    "evaluate_all",
    "evaluate_model",
    "load_labels",
    "per_type_metrics",
    "ranking_metrics",
    "results_to_frame",
    "select_threshold",
    "time_split",
]
