"""Statistical baseline: how surprising is each feature, summed up.

This is the model an analyst would approximate by hand. It has no learning
beyond the empirical distribution of the training window, and every point of
the score can be traced back to a feature, which makes it the honest thing to
beat.

The surprise of a value is the negative log of the probability of seeing
something at least that extreme in the training window:

    surprise(x) = -log P(X >= x)     (or -log P(X <= x) for inverted features)

This is bounded by ``log(2n)``, so a single feature with a huge dynamic range
cannot swamp the rest, and it degrades gracefully on the many features that
are zero almost everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from log_anomaly.models.base import AnomalyModel

#: Features where a *lower* value is the surprising one.
#:
#: The convention in this project is that every feature is oriented so that a
#: larger value is the more suspicious one. The rarity features already obey
#: it: they are negative log frequencies, so a rare pair has a value near zero
#: and a common pair a very negative one. Keeping this set makes the
#: orientation explicit and lets a future feature opt out.
INVERTED_FEATURES: frozenset[str] = frozenset()


class RarityBaseline(AnomalyModel):
    """Sum of per-feature surprise, computed from training statistics only."""

    name = "rarity"
    description = "sum of per-feature empirical-tail surprise"

    def __init__(self) -> None:
        self.feature_names: tuple[str, ...] = ()
        self.sorted_train_: np.ndarray | None = None
        self.train_sample_: np.ndarray | None = None
        self.inverted_: np.ndarray | None = None
        self.floor_: float = 1.0

    def fit(self, features: np.ndarray, feature_names: Sequence[str]) -> RarityBaseline:
        self.feature_names = tuple(feature_names)
        matrix = np.asarray(features, dtype=np.float64)
        self.sorted_train_ = np.sort(matrix, axis=0)
        self.train_sample_ = matrix[:2000].copy()
        self.floor_ = 1.0 / (2 * max(len(matrix), 1))
        self.inverted_ = np.array(
            [name in INVERTED_FEATURES for name in self.feature_names], dtype=bool
        )
        return self

    def contributions(self, features: np.ndarray) -> np.ndarray:
        if self.sorted_train_ is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        total = self.sorted_train_.shape[0]
        surprise = np.zeros_like(matrix)

        for index in range(matrix.shape[1]):
            column = self.sorted_train_[:, index]
            values = matrix[:, index]
            if self.inverted_[index]:
                at_or_below = np.searchsorted(column, values, side="right") / total
                probability = np.maximum(at_or_below, self.floor_)
            else:
                below = np.searchsorted(column, values, side="left") / total
                probability = np.maximum(1.0 - below, self.floor_)
            surprise[:, index] = -np.log(probability)

        return surprise

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.contributions(features).sum(axis=1)

    def feature_importance(self) -> dict[str, float]:
        """Mean surprise each feature contributes on a sample of training data."""
        if self.train_sample_ is None:
            raise RuntimeError("model is not fitted")
        importance = self.contributions(self.train_sample_).mean(axis=0)
        return dict(zip(self.feature_names, importance.tolist(), strict=True))
