"""Tests for the command line interface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from log_anomaly.cli import main


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    assert main(["synth", "--out", str(out), "--days", "10", "--hosts", "3", "--users", "10"]) == 0
    return out


def test_synth_writes_a_labelled_corpus_marked_synthetic(corpus: Path, capsys) -> None:
    assert (corpus / "synth.log").read_text(encoding="utf-8").startswith("# SYNTHETIC LOG")
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    assert meta["synthetic"] is True
    assert len(pd.read_csv(corpus / "labels.csv")) == meta["events"]


def test_train_then_score_prints_ranked_alerts_with_reasons(corpus: Path, tmp_path: Path, capsys) -> None:
    model = tmp_path / "model.joblib"
    assert main(
        ["train", "--log", str(corpus / "synth.log"), "--model", "rarity", "--out", str(model)]
    ) == 0
    assert model.exists()
    capsys.readouterr()

    assert main(
        ["score", str(corpus / "synth.log"), "--model", str(model), "--top-k", "5"]
    ) == 0
    out = capsys.readouterr().out
    assert "5 alerts" in out
    assert out.count("reason:") == 5
    scores = [
        float(match.group(1))
        for line in out.splitlines()
        if (match := re.match(r"\s*\d+\s+score=([\d.eE+-]+)", line))
    ]
    assert len(scores) == 5
    assert scores == sorted(scores, reverse=True)


def test_score_threshold_can_filter_everything_out(corpus: Path, tmp_path: Path, capsys) -> None:
    model = tmp_path / "model.joblib"
    assert main(
        ["train", "--log", str(corpus / "synth.log"), "--model", "rarity", "--out", str(model)]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "score",
            str(corpus / "synth.log"),
            "--model",
            str(model),
            "--threshold",
            "1e18",
        ]
    ) == 0
    assert "0 alerts" in capsys.readouterr().out


def test_evaluate_writes_results_and_marks_them_synthetic(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    assert main(
        [
            "evaluate",
            "--log",
            str(corpus / "synth.log"),
            "--labels",
            str(corpus / "labels.csv"),
            "--out",
            str(out),
            "--models",
            "rarity",
        ]
    ) == 0
    frame = pd.read_csv(out / "results.csv")
    assert frame["model"].tolist() == ["rarity"]
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["results"][0]["metrics"]["per_attack_type"]


def test_score_with_a_missing_model_fails_cleanly(corpus: Path, tmp_path: Path, capsys) -> None:
    code = main(["score", str(corpus / "synth.log"), "--model", str(tmp_path / "nope.joblib")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_unknown_dataset_is_an_argument_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["download", "--dataset", "nope"])
    assert excinfo.value.code == 2
