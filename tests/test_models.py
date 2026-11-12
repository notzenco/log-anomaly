"""Tests for the anomaly models."""

from __future__ import annotations

import numpy as np
import pytest

from log_anomaly.models.rarity import RarityBaseline

NAMES = ("fail_count", "distinct_users", "off_hours", "bytes")


def normal_matrix(rows: int = 400, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.poisson(1.0, rows),
            rng.poisson(2.0, rows),
            rng.binomial(1, 0.05, rows),
            rng.normal(1000, 200, rows),
        ]
    ).astype(np.float64)


def test_rarity_scores_outlier_above_normals() -> None:
    matrix = normal_matrix()
    model = RarityBaseline().fit(matrix, NAMES)
    outlier = np.array([[60.0, 2.0, 0.0, 1000.0]])
    assert model.score(outlier)[0] > np.quantile(model.score(matrix), 0.99)


def test_rarity_contributions_sum_to_score_and_explain_the_outlier() -> None:
    matrix = normal_matrix()
    model = RarityBaseline().fit(matrix, NAMES)
    outlier = np.array([[60.0, 2.0, 0.0, 1000.0]])
    contributions = model.contributions(outlier)
    assert contributions.shape == (1, 4)
    assert (contributions >= 0).all()
    np.testing.assert_allclose(contributions.sum(axis=1), model.score(outlier))
    explanation = model.explain(outlier, NAMES, top_k=2)
    assert explanation[0][0][0] == "fail_count"


def test_rarity_unfitted_raises() -> None:
    with pytest.raises(RuntimeError):
        RarityBaseline().score(normal_matrix(5))


def test_rarity_feature_importance_covers_every_feature() -> None:
    model = RarityBaseline().fit(normal_matrix(), NAMES)
    importance = model.feature_importance()
    assert set(importance) == set(NAMES)
    assert all(value >= 0 for value in importance.values())
