"""Linear autoencoder: PCA reconstruction error.

The classical linear autoencoder. The training window is standardised and
projected onto its principal subspace; an event that the subspace cannot
reconstruct is anomalous. Unlike Isolation Forest this has a global,
covariance-based inductive bias, and it gives *exact* per-feature
attributions because the reconstruction error decomposes additively.

A deep autoencoder (torch, Keras) would be the obvious next step, but it adds
hundreds of megabytes of dependencies for a model that on a few thousand rows
of tabular features would not beat this by enough to matter.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from log_anomaly.models.base import AnomalyModel


class PCAAutoencoder(AnomalyModel):
    """Reconstruction error of a principal-subspace projection."""

    name = "pca-ae"
    description = "linear autoencoder (PCA reconstruction error)"

    def __init__(self, *, variance_threshold: float = 0.95, max_components: int = 10) -> None:
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.feature_names: tuple[str, ...] = ()
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    def fit(self, features: np.ndarray, feature_names: Sequence[str]) -> PCAAutoencoder:
        self.feature_names = tuple(feature_names)
        matrix = np.asarray(features, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(matrix)
        scaled = self.scaler_.transform(matrix)
        self.pca_ = PCA(n_components=self.variance_threshold, random_state=0).fit(scaled)
        if int(self.pca_.n_components_) > self.max_components:
            # cap the subspace so reconstruction error stays a bottleneck
            self.pca_ = PCA(n_components=self.max_components, random_state=0).fit(scaled)
        return self

    def _reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        scaled = self.scaler_.transform(matrix)
        reconstructed = self.pca_.inverse_transform(self.pca_.transform(scaled))
        error = scaled - reconstructed
        return error

    def contributions(self, features: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction error."""
        error = self._reconstruction_error(features)
        return error**2

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.contributions(features).sum(axis=1)

    def feature_importance(self) -> dict[str, float]:
        raise NotImplementedError(
            "the linear autoencoder attributes per event, not per feature globally"
        )

    @property
    def n_components(self) -> int:
        if self.pca_ is None:
            raise RuntimeError("model is not fitted")
        return int(self.pca_.n_components_)
