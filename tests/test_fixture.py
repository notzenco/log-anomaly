"""End-to-end test on the tiny committed SYNTHETIC fixture.

This is what CI runs: no downloads, no generated state, just the fixture
under ``tests/fixtures`` through parse -> features -> model. The fixture's
base rate (17.5%) is far above the report corpus; it tests plumbing and
ranking direction, not detection quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from log_anomaly.evaluation import align_labels, load_labels
from log_anomaly.evaluation.splits import time_split
from log_anomaly.features import extract_features
from log_anomaly.models import fit_model
from log_anomaly.parsing import parse_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_tiny_fixture_flows_through_the_pipeline() -> None:
    log_path = FIXTURES / "tiny_synth.log"
    assert log_path.read_text(encoding="utf-8").startswith("# SYNTHETIC LOG")

    events, stats = parse_file(log_path, anchor_year=2024)
    labels = load_labels(FIXTURES / "tiny_labels.csv")
    assert stats["parsed"] == len(labels) == 451

    feature_set = extract_features(events)
    aligned = align_labels(feature_set.events, labels)
    split = time_split(feature_set.events["ts"])
    matrix = feature_set.features.to_numpy(dtype=np.float64)

    trained = fit_model("rarity", matrix[split.train], feature_set.feature_names)
    test_scores = trained.score(matrix[split.test])
    test_labels = aligned["label"].to_numpy()[split.test]
    assert test_labels.sum() > 0
    # attacks rank above benign events on the held-out window
    assert np.median(test_scores[test_labels == 1]) > np.median(test_scores[test_labels == 0])
    explanations = trained.explain(matrix[split.test][:3], top_k=3)
    assert all(len(explanation) == 3 for explanation in explanations)
