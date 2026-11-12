"""Common interface for the anomaly models.

Every model turns a feature matrix into an anomaly score where **higher means
more anomalous**, and can explain a score by attributing it to features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class AnomalyModel(ABC):
    """Base class for the anomaly models."""

    #: Registry key, also used on the command line.
    name: str = "base"

    #: Human readable description used in reports.
    description: str = ""

    @abstractmethod
    def fit(self, features: np.ndarray, feature_names: Sequence[str]) -> AnomalyModel:
        """Fit the model on the training features."""

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row; higher means more anomalous."""

    def contributions(self, features: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the score, shaped ``(n_rows, n_features)``.

        Contributions are non-negative and expressed in the same units as
        :meth:`score`.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot explain its scores")

    def explain(
        self,
        features: np.ndarray,
        feature_names: Sequence[str],
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Explain each row: the ``top_k`` features that drove the score."""
        contributions = self.contributions(features)
        out: list[list[tuple[str, float]]] = []
        for row in contributions:
            order = np.argsort(row)[::-1][:top_k]
            out.append([(feature_names[index], float(row[index])) for index in order])
        return out

    def feature_importance(self) -> dict[str, float]:
        """Mean contribution per feature over the training set, if available."""
        raise NotImplementedError(f"{type(self).__name__} has no feature importance")
