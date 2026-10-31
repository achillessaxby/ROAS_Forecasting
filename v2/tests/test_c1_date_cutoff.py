"""C1 - training cutoff must keep every month before as_of."""
import pandas as pd

from roas.data import apply_window, month_start, parse_date


def test_v1_bug_reproduced():
    # v1 built "1/10/2026" for 1 Oct 2026; pandas reads it month-first as 10 Jan 2026
    assert pd.Timestamp("1/10/2026") == pd.Timestamp("2026-01-10")


def test_month_start_is_a_timestamp():
    assert month_start("2026-10-30") == pd.Timestamp("2026-10-01")


def test_parse_date_is_day_first():
    assert parse_date(["03/04/2025"]).iloc[0] == pd.Timestamp("2025-04-03")


def test_window_keeps_all_months_before_cutoff():
    months = pd.date_range("2026-01-01", "2026-10-01", freq="MS")
    df = pd.DataFrame({"cohort": months[0], "transaction_period": months, "elapsed_months": range(1, 11)})
    kept = apply_window(df, "2026-10-30")
    assert kept["transaction_period"].max() == pd.Timestamp("2026-09-01")
    assert len(kept) == 9  # v1 would have kept only January


def test_window_drops_old_cohorts():
    df = pd.DataFrame({"cohort": pd.to_datetime(["2022-12-01", "2023-01-01"]),
                       "transaction_period": pd.to_datetime(["2023-02-01", "2023-02-01"]), "elapsed_months": [3, 2]})
    assert apply_window(df, "2024-01-01")["cohort"].min() == pd.Timestamp("2023-01-01")
