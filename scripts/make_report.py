#!/usr/bin/env python
"""Generate every number and chart that REPORT.md cites.

Reads (or first generates) the SYNTHETIC corpus under ``data/synthetic`` and,
when present, the real Loghub logs under ``data/raw/logs``. Writes results to
``reports/`` (git-ignored) and charts to ``docs/figures/`` (committed, so
REPORT.md renders on GitHub).

Usage:
    uv run python scripts/make_report.py [--skip-real]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from log_anomaly.data.synthetic import SyntheticConfig, write_corpus
from log_anomaly.evaluation import evaluate_all, load_labels, results_to_frame
from log_anomaly.evaluation.metrics import per_type_metrics
from log_anomaly.evaluation.splits import time_split
from log_anomaly.features import extract_features
from log_anomaly.models import fit_model
from log_anomaly.parsing import parse_file

ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT / "data" / "synthetic"
RAW_DIR = ROOT / "data" / "raw" / "logs"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"

REAL_CORPORA = {
    "linux": ("Linux.log", 2004),
    "openssh": ("SSH.log", 2017),
    "proxifier": ("Proxifier.log", 2017),
}


def ensure_synthetic() -> tuple[Path, Path]:
    log = SYNTH_DIR / "synth.log"
    labels = SYNTH_DIR / "labels.csv"
    if not (log.exists() and labels.exists()):
        write_corpus(SyntheticConfig(), log, labels, SYNTH_DIR / "meta.json")
    return log, labels


def chart_pr_curves(results, base_rate: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in results:
        precision, recall, _ = precision_recall_curve(result.test_labels, result.test_scores)
        ax.plot(recall, precision, label=f"{result.model} (PR-AUC {result.metrics['ranking']['pr_auc']:.3f})")
    ax.axhline(base_rate, color="grey", linestyle="--", label=f"base rate ({base_rate:.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves, SYNTHETIC test window")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves.png", dpi=150)
    plt.close(fig)


def chart_score_distributions(results) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(11, 3.2), sharey=True)
    for ax, result in zip(axes, results, strict=True):
        scores = result.test_scores
        labels = result.test_labels
        lo, hi = np.quantile(scores, [0.001, 0.999])
        bins = np.linspace(lo, hi, 60)
        ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="benign", density=True)
        ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="attack", density=True)
        ax.axvline(result.thresholds["f1"], color="red", linestyle="--", label="F1 threshold")
        ax.set_title(result.model, fontsize=10)
        ax.set_xlabel("anomaly score")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Score distributions by class, SYNTHETIC test window")
    fig.tight_layout()
    fig.savefig(FIGURES / "score_distributions.png", dpi=150)
    plt.close(fig)


def chart_per_type_recall(results) -> None:
    types = sorted(results[0].metrics["per_attack_type"])
    x = np.arange(len(types))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for offset, result in enumerate(results):
        recalls = [result.metrics["per_attack_type"][t]["recall"] for t in types]
        ax.bar(x + offset * width, recalls, width=width, label=result.model)
    ax.set_xticks(x + width)
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("event recall at the F1 threshold")
    ax.set_title("Per-attack-type recall, SYNTHETIC test window")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_type_recall.png", dpi=150)
    plt.close(fig)


def chart_alert_volume(results) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in results:
        stamps = pd.to_datetime(result.test_timestamps, utc=True)
        days = max((stamps.max() - stamps.min()).total_seconds() / 86400, 1e-9)
        thresholds = np.unique(np.quantile(result.test_scores, np.linspace(0.9, 0.99999, 60)))
        volumes, recalls = [], []
        positives = max(int(result.test_labels.sum()), 1)
        for threshold in thresholds:
            predicted = result.test_scores >= threshold
            volumes.append(predicted.sum() / days)
            recalls.append((predicted & (result.test_labels == 1)).sum() / positives)
        ax.plot(volumes, recalls, label=result.model)
    ax.set_xscale("log")
    ax.set_xlabel("alerts per day (log scale)")
    ax.set_ylabel("event recall")
    ax.set_title("Recall bought per daily alert volume, SYNTHETIC test window")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "alert_volume.png", dpi=150)
    plt.close(fig)


def false_positive_analysis(events, labels, feature_set) -> dict[str, Any]:
    """What pca-ae gets wrong on the synthetic test window, in detail."""
    from log_anomaly.evaluation import evaluate_model

    result = evaluate_model(
        events, labels, model_name="pca-ae", corpus="synthetic", feature_set=feature_set
    )
    split = time_split(feature_set.events["ts"])
    ordered = feature_set.events[split.test].reset_index(drop=True)
    matrix = feature_set.features.to_numpy(dtype=np.float64)[split.test]
    trained = fit_model(
        "pca-ae", feature_set.features.to_numpy(dtype=np.float64)[split.train], feature_set.feature_names
    )
    threshold = result.thresholds["f1"]
    predicted = result.test_scores >= threshold
    is_fp = predicted & (result.test_labels == 0)
    is_fn = ~predicted & (result.test_labels == 1)

    fp_events = ordered[is_fp]
    contributions = trained.model.contributions(matrix[is_fp])
    mean_contrib = contributions.mean(axis=0)
    top_features = sorted(
        zip(feature_set.feature_names, mean_contrib.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:6]

    examples = []
    fp_positions = np.flatnonzero(is_fp)
    order = fp_positions[np.argsort(result.test_scores[fp_positions])[::-1][:8]]
    explanations = trained.explain(matrix[order], top_k=3)
    for position, explanation in zip(order, explanations, strict=True):
        event = ordered.iloc[position]
        examples.append(
            {
                "score": float(result.test_scores[position]),
                "event_type": str(event["event_type"]),
                "reason": [(name, round(value, 3)) for name, value in explanation],
                "raw": str(event["raw"])[:200],
            }
        )

    return {
        "model": "pca-ae",
        "false_positives": int(is_fp.sum()),
        "false_negatives": int(is_fn.sum()),
        "fp_by_event_type": fp_events["event_type"].value_counts().astype(int).to_dict(),
        "fp_top_features": [(name, round(value, 4)) for name, value in top_features],
        "fp_examples": examples,
        "fn_by_attack_type": pd.Series(result.test_types[is_fn]).value_counts().astype(int).to_dict(),
        "per_type_at_budget": per_type_metrics(
            result.test_labels, result.test_scores, result.test_types, result.thresholds["budget"]
        ),
    }


def real_data_section() -> dict[str, Any]:
    """Unsupervised quantities on the real Loghub logs: no labels exist."""
    out: dict[str, Any] = {}
    for name, (member, year) in REAL_CORPORA.items():
        path = RAW_DIR / member
        if not path.exists():
            out[name] = {"present": False}
            continue
        events, stats = parse_file(path, anchor_year=year)
        feature_set = extract_features(events)
        split = time_split(feature_set.events["ts"], train_fraction=0.6, validation_fraction=0.2)
        matrix = feature_set.features.to_numpy(dtype=np.float64)
        entry: dict[str, Any] = {
            "present": True,
            "parse_stats": stats,
            "span": [str(feature_set.events["ts"].min()), str(feature_set.events["ts"].max())],
            "event_types": feature_set.events["event_type"].value_counts().astype(int).to_dict(),
            "availability": feature_set.availability,
            "models": {},
        }
        test_events = feature_set.events[split.test].reset_index(drop=True)
        test_stamps = test_events["ts"]
        days = max((test_stamps.max() - test_stamps.min()).total_seconds() / 86400, 1e-9)
        for model_name in ("rarity", "pca-ae"):
            trained = fit_model(model_name, matrix[split.train], feature_set.feature_names)
            scores = trained.score(matrix[split.test])
            budget_threshold = float(np.quantile(scores, 1 - min(20 * days, len(scores)) / len(scores)))
            entry["models"][model_name] = {
                "alerts_per_day_at_20_per_day_quantile": float((scores >= budget_threshold).sum() / days),
                "score_quantiles": {
                    str(q): float(np.quantile(scores, q)) for q in (0.5, 0.99, 0.999, 0.9999)
                },
            }
            if name == "linux" and model_name == "rarity":
                top = np.argsort(scores)[::-1][:30]
                explanations = trained.explain(matrix[split.test][top], top_k=3)
                rows = []
                for rank, (position, explanation) in enumerate(
                    zip(top, explanations, strict=True), 1
                ):
                    event = test_events.iloc[position]
                    rows.append(
                        {
                            "rank": rank,
                            "score": float(scores[position]),
                            "ts": str(event["ts"]),
                            "event_type": str(event["event_type"]),
                            "user": str(event["user"]),
                            "src_ip": str(event["src_ip"]),
                            "reason": "; ".join(f"{n}={v:.3g}" for n, v in explanation),
                            "raw": str(event["raw"]).strip(),
                            "verdict": "",
                        }
                    )
                pd.DataFrame(rows).to_csv(REPORTS / "audit_candidates_linux.csv", index=False)
        out[name] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true", help="skip the real Loghub corpora")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    log_path, labels_path = ensure_synthetic()
    events, _ = parse_file(log_path, anchor_year=2024)
    labels = load_labels(labels_path)
    feature_set = extract_features(events)

    variants = {
        "main": {},
        "exclude_background_scan": {"exclude_types": ("background_scan",)},
        "clean_training_oracle": {
            "exclude_types": ("background_scan",),
            "clean_training": True,
        },
    }
    summary: dict[str, Any] = {"synthetic": True, "variants": {}}
    for variant, kwargs in variants.items():
        results = evaluate_all(
            events, labels, corpus="synthetic", feature_set=feature_set, **kwargs
        )
        frame = results_to_frame(results)
        frame.to_csv(REPORTS / f"results_{variant}.csv", index=False)
        summary["variants"][variant] = frame.to_dict(orient="records")
        if variant == "main":
            test_types = results[0].test_types
            test_labels_arr = results[0].test_labels
            summary["test_base_rates"] = {
                str(t): {
                    "events": int((test_types == t).sum()),
                    "share_of_test": float((test_types == t).mean()),
                }
                for t in sorted(set(test_types[test_labels_arr == 1]))
            }
            summary["split"] = results[0].metrics["split"]
            summary["split_sizes"] = results[0].metrics["split_sizes"]
            summary["per_attack_type"] = {
                r.model: r.metrics["per_attack_type"] for r in results
            }
            summary["episodes"] = {r.model: r.metrics["episodes"] for r in results}
            summary["operating_points"] = {
                r.model: r.metrics["operating_points"] for r in results
            }
            chart_pr_curves(results, float(test_labels_arr.mean()))
            chart_score_distributions(results)
            chart_per_type_recall(results)
            chart_alert_volume(results)

    summary["false_positives"] = false_positive_analysis(events, labels, feature_set)
    if not args.skip_real:
        summary["real"] = real_data_section()

    (REPORTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {REPORTS / 'summary.json'} and charts under {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
