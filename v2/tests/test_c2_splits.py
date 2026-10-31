"""C2 - splits are by calendar time; no future rows in training."""
import pandas as pd
import pytest

from roas.data import build_training_frame
from roas.splits import assert_no_leakage, expanding_window_folds, temporal_split


def test_temporal_split_no_leakage(games):
    frame = build_training_frame(next(iter(games.values())))
    tr, te = temporal_split(frame, "2025-01-01")
    assert_no_leakage(frame.iloc[tr], frame.iloc[te])


def test_expanding_folds_respect_time(games):
    frame = build_training_frame(next(iter(games.values())))
    folds = expanding_window_folds(frame)
    assert len(folds) == 3
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)
    for tr, va in folds:
        assert_no_leakage(frame.iloc[tr], frame.iloc[va])


def test_leakage_detected():
    a = pd.DataFrame({"transaction_period": pd.to_datetime(["2025-03-01"])})
    b = pd.DataFrame({"transaction_period": pd.to_datetime(["2025-02-01"])})
    with pytest.raises(AssertionError):
        assert_no_leakage(a, b)
