"""Anomaly models: a statistical baseline, Isolation Forest and a linear autoencoder."""

from __future__ import annotations

from log_anomaly.models.base import AnomalyModel
from log_anomaly.models.iforest import IsolationForestModel
from log_anomaly.models.pca_autoencoder import PCAAutoencoder
from log_anomaly.models.rarity import INVERTED_FEATURES, RarityBaseline
from log_anomaly.models.registry import (
    MODEL_NAMES,
    TrainedModel,
    build_model,
    fit_model,
)

__all__ = [
    "INVERTED_FEATURES",
    "MODEL_NAMES",
    "AnomalyModel",
    "IsolationForestModel",
    "PCAAutoencoder",
    "RarityBaseline",
    "TrainedModel",
    "build_model",
    "fit_model",
]
