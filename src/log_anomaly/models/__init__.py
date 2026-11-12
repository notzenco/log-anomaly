"""Anomaly models: a statistical baseline, Isolation Forest and a linear autoencoder."""

from __future__ import annotations

from log_anomaly.models.base import AnomalyModel
from log_anomaly.models.iforest import IsolationForestModel
from log_anomaly.models.rarity import INVERTED_FEATURES, RarityBaseline

__all__ = [
    "INVERTED_FEATURES",
    "AnomalyModel",
    "IsolationForestModel",
    "RarityBaseline",
]
