"""Walk-forward back-test (review findings C3, M2, M4).

v1 scored the *final* models (trained on every cohort) against the latest cohorts, so the
"case" error tables were in-sample. Here, for every forecast origin the models are
re-trained on data available at that date only, then scored on months realised after it.

One case definition for everything (M4):
    Case k = a forecast made (k + 5) months before the latest data month.
    Case 1 therefore has 6 realised months to score, case 6 has 11.

Every prediction is compared with two baselines, both built only from data before the origin:
  * ``baseline``    - naive: average curve of the three most recent cohorts at that month
  * ``last_cohort`` - the v1 serving method: the most recent cohort's curve, reused as-is
and band coverage - how often the actual lands between bad and good - is reported
(target ~50% for a 25-75 band).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import CANONICAL_CHANNELS, DEFAULT_SETTINGS, Settings
from .data import TARGET, apply_window, month_start
from .forecast import forecast_scenario
from .pipeline import game_profile, train_models
from .scaling import estimate_alphas
from .splits import assert_no_leakage

log = logging.getLogger(__name__)
BUCKETS = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 99, "13+")]


def _baseline_curves(hist: pd.DataFrame, origin: pd.Timestamp) -> tuple[dict[int, float], dict[int, float]]:
    """(naive 3-cohort average, latest-cohort curve) per elapsed month, from data before origin."""
    seen = hist[hist["transaction_period"] < origin]
    naive, last = {}, {}
    for m, grp in seen.groupby("elapsed_months"):
        recent = grp.sort_values("cohort")
        naive[int(m)] = float(recent[TARGET].tail(3).mean())
        last[int(m)] = float(recent[TARGET].iloc[-1])
    return naive, last


def run_backtest(
    games: Mapping[str, pd.DataFrame],
    latest: str | pd.Timestamp,
    n_cases: int = 6,
    horizon: int = 12,
    categories: Mapping[str, str] | None = None,
    settings: Settings = DEFAULT_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = month_start(latest)
    categories = categories or {}
    rows = []
    for case in range(1, n_cases + 1):
        origin = latest - pd.DateOffset(months=case + 5)
        models, windowed = train_models(games, origin, settings)          # re-trained per origin
        alphas = estimate_alphas(windowed, categories, settings)
        for g, model in models.items():
            full = apply_window(games[g], latest, settings)                 # everything realised by `latest`
            assert_no_leakage(windowed[g], full[full["transaction_period"] >= origin])
            prof = game_profile(windowed[g])
            base, last = _baseline_curves(windowed[g], origin)
            first = origin - pd.DateOffset(months=12)
            for cohort, cdf in full[full["cohort"] >= first].groupby("cohort"):
                m1 = cdf[cdf["elapsed_months"] == 1]
                if m1.empty:
                    continue
                spend = float(m1["total_cac"].iloc[0])
                mix = m1[CANONICAL_CHANNELS].iloc[0]
                shares = (mix / mix.sum()).to_dict() if mix.sum() > 0 else prof["default_shares"]
                observed = {int(r.elapsed_months): float(r[TARGET]) for _, r in cdf.iterrows()
                            if r["transaction_period"] < origin and pd.notna(r[TARGET])}
                target_rows = cdf[(cdf["transaction_period"] >= origin) & cdf[TARGET].notna()]
                if target_rows.empty:
                    continue
                h = int(min(target_rows["elapsed_months"].max(), settings.max_horizon))
                fc = forecast_scenario(model, cohort, spend, shares, h, observed=observed,
                                       spend_range=(prof["spend_min"], prof["spend_max"]),
                                       alpha=alphas[g].alpha, settings=settings).set_index("elapsed_months")
                for _, r in target_rows.iterrows():
                    m = int(r["elapsed_months"])
                    if m > h:
                        continue
                    ahead = (r["transaction_period"].to_period("M") - origin.to_period("M")).n + 1
                    if ahead > horizon:
                        continue
                    rows.append({
                        "case": case, "origin": origin.strftime("%Y-%m"), "game_id": g,
                        "cohort": cohort.strftime("%Y-%m"), "elapsed_months": m, "months_ahead": ahead,
                        "new_cohort": cohort >= origin, "actual": float(r[TARGET]),
                        "bad": fc.at[m, "roas_bad"], "expected": fc.at[m, "roas_expected"],
                        "good": fc.at[m, "roas_good"], "baseline": base.get(m, np.nan),
                        "last_cohort": last.get(m, np.nan),
                    })
        log.info("case %d (origin %s): %d games scored", case, origin.date(), len(models))
    preds = pd.DataFrame(rows)
    return preds, summarise(preds, settings)


def summarise(preds: pd.DataFrame, settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    p = preds.dropna(subset=["baseline", "last_cohort"]).copy()
    p["abs_err"] = (p["actual"] - p["expected"]).abs()
    p["abs_err_base"] = (p["actual"] - p["baseline"]).abs()
    p["abs_err_last"] = (p["actual"] - p["last_cohort"]).abs()
    p["covered"] = (p["actual"] >= p["bad"]) & (p["actual"] <= p["good"])
    qs = settings.quantiles
    def pin(df):
        out = []
        for q, col in zip(qs, ["bad", "expected", "good"], strict=False):
            d = df["actual"] - df[col]
            out.append(np.maximum(q * d, (q - 1) * d).mean())
        return float(np.mean(out))

    def stats(df, level, key):
        mae, mae_b, mae_l = df["abs_err"].mean(), df["abs_err_base"].mean(), df["abs_err_last"].mean()
        return {"level": level, "key": key, "n": len(df), "mae_model": mae, "mae_baseline": mae_b,
                "mae_last_cohort": mae_l,
                "skill_vs_baseline": 1 - mae / mae_b if mae_b > 0 else np.nan,
                "skill_vs_last_cohort": 1 - mae / mae_l if mae_l > 0 else np.nan,
                "band_coverage": df["covered"].mean(), "pinball": pin(df)}

    out = [stats(p, "overall", "all")]
    for lo, hi, name in BUCKETS:
        sub = p[(p["months_ahead"] >= lo) & (p["months_ahead"] <= hi)]
        if len(sub):
            out.append(stats(sub, "months_ahead", name))
    for case, sub in p.groupby("case"):
        out.append(stats(sub, "case", str(case)))
    for flag, sub in p.groupby("new_cohort"):
        out.append(stats(sub, "cohort_type", "new (not yet acquired)" if flag else "live (partly observed)"))
    return pd.DataFrame(out)
