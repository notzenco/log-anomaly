"""Model registry and the serialisable bundle used by the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from log_anomaly.models.base import AnomalyModel
from log_anomaly.models.iforest import IsolationForestModel
from log_anomaly.models.pca_autoencoder import PCAAutoencoder
from log_anomaly.models.rarity import RarityBaseline

#: Every model the project ships, in the order the report compares them.
MODEL_NAMES: tuple[str, ...] = ("rarity", "iforest", "pca-ae")

_MODEL_CLASSES: dict[str, type[AnomalyModel]] = {
    "rarity": RarityBaseline,
    "iforest": IsolationForestModel,
    "pca-ae": PCAAutoencoder,
}


def build_model(name: str, **kwargs: Any) -> AnomalyModel:
    """Instantiate a model by registry name."""
    try:
        model_class = _MODEL_CLASSES[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}") from None
    return model_class(**kwargs)


@dataclass
class TrainedModel:
    """A fitted model plus everything needed to score new data."""

    model: AnomalyModel
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.model.score(features)

    def explain(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        return self.model.explain(features, self.feature_names, top_k=top_k)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrainedModel:
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, TrainedModel):
            raise TypeError(f"{path} does not contain a TrainedModel")
        return bundle


def fit_model(
    name: str,
    features: np.ndarray,
    feature_names: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> TrainedModel:
    """Fit a named model and wrap it in a :class:`TrainedModel`."""
    model = build_model(name)
    model.fit(features, feature_names)
    return TrainedModel(
        model=model,
        feature_names=tuple(feature_names),
        metadata=metadata or {},
    )
