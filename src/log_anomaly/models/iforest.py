"""Isolation Forest: anomaly score from how easily an event can be isolated."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from log_anomaly.models.base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    """Wrapper around :class:`sklearn.ensemble.IsolationForest`."""

    name = "iforest"
    description = "Isolation Forest (100 trees, no hyper-parameter search)"

    def __init__(self, *, n_estimators: int = 100, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names: tuple[str, ...] = ()
        self.model_: IsolationForest | None = None
        self.median_: np.ndarray | None = None
        self.train_matrix_: np.ndarray | None = None

    def fit(self, features: np.ndarray, feature_names: Sequence[str]) -> IsolationForestModel:
        self.feature_names = tuple(feature_names)
        matrix = np.asarray(features, dtype=np.float64)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(matrix)
        self.median_ = np.median(matrix, axis=0)
        # keep only a sample: it exists for feature_importance, and the full
        # training matrix would bloat every saved artifact
        self.train_matrix_ = matrix[:2000].copy()
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        return -self.model_.score_samples(matrix)

    def contributions(self, features: np.ndarray) -> np.ndarray:
        """Occlusion attribution: how much the score drops without each feature.

        Each feature is replaced by its training median and the score is
        recomputed. The drop is the feature's contribution.
        """
        if self.model_ is None or self.median_ is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        baseline = self.score(matrix)
        out = np.zeros_like(matrix)
        for index in range(matrix.shape[1]):
            occluded = matrix.copy()
            occluded[:, index] = self.median_[index]
            out[:, index] = np.maximum(0.0, baseline - self.score(occluded))
        return out

    def feature_importance(self) -> dict[str, float]:
        """Mean occlusion contribution per feature on a sample of the training set."""
        if self.train_matrix_ is None:
            raise RuntimeError("model is not fitted")
        sample = self.train_matrix_[: min(2000, len(self.train_matrix_))]
        importance = self.contributions(sample).mean(axis=0)
        return dict(zip(self.feature_names, importance.tolist(), strict=True))
