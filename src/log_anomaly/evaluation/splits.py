"""Time-based splitting.

Splits are cut on the *duration* of the corpus rather than on the number of
rows, so a quiet week and a busy week each get their share, and no event
from the test window can influence a training statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Boolean masks for the train / validation / test windows."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    boundaries: dict[str, str]

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "validation": int(self.validation.sum()),
            "test": int(self.test.sum()),
        }


def time_split(
    timestamps: pd.Series,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> Split:
    """Split ``timestamps`` into three consecutive windows.

    ``train_fraction`` and ``validation_fraction`` are fractions of the total
    *duration*; the remainder is the test window.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for a test window")

    values = pd.to_datetime(timestamps, utc=True)
    start = values.min()
    end = values.max()
    span = end - start
    train_end = start + span * train_fraction
    validation_end = start + span * (train_fraction + validation_fraction)

    train = (values < train_end).to_numpy()
    validation = ((values >= train_end) & (values < validation_end)).to_numpy()
    test = (values >= validation_end).to_numpy()

    boundaries = {
        "start": _iso(start),
        "train_end": _iso(train_end),
        "validation_end": _iso(validation_end),
        "end": _iso(end),
    }
    return Split(train=train, validation=validation, test=test, boundaries=boundaries)


def _iso(value: datetime) -> str:
    return value.isoformat()
