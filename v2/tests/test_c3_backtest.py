"""C3 - back-test re-trains at each origin and scores only later months."""
import pandas as pd

from roas.backtest import run_backtest


def test_walk_forward_is_out_of_sample(games, settings):
    subset = {g: games[g] for g in list(games)[:2]}
    preds, summary = run_backtest(subset, "2025-09-01", n_cases=2, horizon=6, settings=settings)
    assert not preds.empty
    # every scored month is on/after its forecast origin
    assert (preds["months_ahead"] >= 1).all()
    for case, origin in preds.groupby("case")["origin"].first().items():
        assert origin == (pd.Timestamp("2025-09-01") - pd.DateOffset(months=case + 5)).strftime("%Y-%m")
    assert {"mae_model", "mae_baseline", "band_coverage", "skill_vs_baseline"} <= set(summary.columns)
    assert (preds["bad"] <= preds["expected"]).all() and (preds["expected"] <= preds["good"]).all()


def test_model_beats_naive_baseline(games, settings):
    subset = {g: games[g] for g in list(games)[:3]}
    _, summary = run_backtest(subset, "2025-09-01", n_cases=2, horizon=6, settings=settings)
    overall = summary[summary.level == "overall"].iloc[0]
    assert overall.mae_model < overall.mae_baseline
