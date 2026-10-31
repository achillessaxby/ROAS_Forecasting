"""Time-respecting splits (review finding C2).

v1 used ``train_test_split(test_size=0.2, random_state=42)`` and ``RandomizedSearchCV(cv=5)``
on cohort x month rows. Rows from the same cohort, and from *later* calendar months, ended
up in both train and test, so test scores leaked the future. Every split here is defined
by calendar time: a model only ever sees transactions that happened before the period it
is evaluated on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(frame: pd.DataFrame, boundary: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Train = transactions before ``boundary``; test = transactions on/after it."""
    b = pd.Timestamp(boundary)
    tp = frame["transaction_period"]
    return np.flatnonzero(tp < b), np.flatnonzero(tp >= b)


def expanding_window_folds(
    frame: pd.DataFrame, n_splits: int = 3, val_months: int = 3, min_train_months: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV on calendar months of ``transaction_period``.

    Fold i trains on everything before boundary_i and validates on the next ``val_months``.
    Returns at most ``n_splits`` folds (fewer if the history is short).
    """
    months = np.sort(frame["transaction_period"].dt.to_period("M").unique())
    folds = []
    for i in range(n_splits, 0, -1):
        end = len(months) - (i - 1) * val_months
        start = end - val_months
        if start < min_train_months:
            continue
        b0 = months[start].to_timestamp()
        b1 = months[end - 1].to_timestamp() + pd.offsets.MonthBegin(1)
        tp = frame["transaction_period"]
        tr = np.flatnonzero(tp < b0)
        va = np.flatnonzero((tp >= b0) & (tp < b1))
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any training transaction is not strictly earlier than every test one."""
    if len(train) and len(test) and train["transaction_period"].max() >= test["transaction_period"].min():
        raise AssertionError(
            f"leakage: train ends {train['transaction_period'].max():%Y-%m} "
            f"but test starts {test['transaction_period'].min():%Y-%m}"
        )
