"""Recursive, scenario-aware forecasting (review finding C4).

v1's API returned the latest cohort's stored curve times one scalar, so the requested
cohort month and channel mix had almost no effect. Here the per-game quantile models are
scored for the *requested* scenario: spend, channel shares, cohort month and the calendar
month of each forecast month all enter the features. Each quantile keeps its own lag path
(bad / expected / good), and the three are re-sorted per month so they never cross.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import CANONICAL_CHANNELS, DEFAULT_SETTINGS, Settings
from .models import QuantileModelSet
from .scaling import out_of_range_factor

LABELS = ("bad", "expected", "good")


def scenario_static_row(cohort_month: pd.Timestamp, total_cac: float, shares: Mapping[str, float]) -> dict[str, float]:
    row = {"log_total_cac": float(np.log1p(total_cac)), "cohort_month": int(cohort_month.month)}
    for c in CANONICAL_CHANNELS:
        row[f"share_{c}"] = float(shares.get(c, 0.0))
    return row


def normalise_mix(channel_spends: Mapping[str, float] | None, default_shares: Mapping[str, float],
                  total_cac: float | None) -> tuple[float, dict[str, float]]:
    """Return (total_cac, shares). Spends given -> their sum and mix; else default mix."""
    if channel_spends:
        total = float(sum(channel_spends.values()))
        if total <= 0:
            raise ValueError("channel_spends must sum to a positive amount")
        if total_cac is not None and abs(total - total_cac) > 0.01 * max(total, total_cac):
            raise ValueError(f"total_cac ({total_cac:,.0f}) does not match sum of channel_spends ({total:,.0f})")
        return total, {c: channel_spends.get(c, 0.0) / total for c in CANONICAL_CHANNELS}
    if total_cac is None or total_cac <= 0:
        raise ValueError("provide total_cac or channel_spends")
    return float(total_cac), {c: float(default_shares.get(c, 0.0)) for c in CANONICAL_CHANNELS}


def forecast_scenario(
    model: QuantileModelSet,
    cohort_month: str | pd.Timestamp,
    total_cac: float,
    shares: Mapping[str, float],
    horizon: int,
    *,
    observed: Mapping[int, float] | None = None,
    spend_range: tuple[float, float] = (0.0, float("inf")),
    alpha: float = DEFAULT_SETTINGS.default_alpha,
    settings: Settings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """Forecast months 1..horizon for one cohort scenario.

    ``observed`` maps elapsed month -> actual ROAS for months already realised (used by the
    back-test and for live cohorts); those months are returned as actuals and feed the lags.
    """
    cohort = pd.Timestamp(cohort_month).normalize().replace(day=1)
    observed = dict(observed or {})
    static = scenario_static_row(cohort, total_cac, shares)
    factor = out_of_range_factor(total_cac, *spend_range, alpha=alpha, settings=settings)
    nq = len(model.quantiles)
    paths = [[] for _ in range(nq)]  # per-quantile ROAS history
    rows = []
    for m in range(1, horizon + 1):
        cal = cohort + pd.DateOffset(months=m - 1)
        if m in observed:
            vals = np.full(nq, float(observed[m]))
            source = "actual"
        else:
            X = []
            for i in range(nq):
                hist = paths[i]
                lags = {f"lag_{k}": (hist[-k] if len(hist) >= k else np.nan) for k in range(1, settings.n_lags + 1)}
                X.append({**static, "elapsed_months": m, "current_month": cal.month, **lags})
            pred = model.predict(pd.DataFrame(X))
            vals = np.array([pred[i, i] for i in range(nq)]) * factor
            vals = np.sort(np.maximum(vals, 0.0))
            source = "forecast"
        for i in range(nq):
            paths[i].append(float(vals[i]))
        rows.append({"elapsed_months": m, "calendar_month": cal.strftime("%Y-%m"), "source": source,
                     **{f"roas_{LABELS[i] if nq == 3 else i}": float(vals[i]) for i in range(nq)}})
    out = pd.DataFrame(rows)
    labels = [LABELS[i] if nq == 3 else str(i) for i in range(nq)]
    cum = np.sort(np.cumsum(out[[f"roas_{lab}" for lab in labels]].to_numpy(), axis=0), axis=1)
    for i, lab in enumerate(labels):
        out[f"cumulative_{lab}"] = cum[:, i]
        out[f"revenue_{lab}"] = out[f"roas_{lab}"] * total_cac
    out.attrs["spend_factor"] = factor
    return out
