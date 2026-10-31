"""Diminishing-returns adjustment for spend outside the range a game was trained on.

Why it exists: tree models cannot extrapolate. Within the observed spend range the
quantile models score the requested spend and channel mix directly (finding C4). Outside
it, the v1 idea is kept - ROAS falls as spend grows - but its estimation is fixed:

M5  v1 estimated alpha from the 5 highest-spend month-1 cohorts and kept only *positive*
    estimates, which biases alpha upward. Here alpha comes from a robust (Theil-Sen) slope
    of log month-1 ROAS on log spend over all eligible cohorts, negative estimates are kept
    during pooling, and each game is shrunk toward its category by sample size.
M6  v1's channel-mix multiplier (beta) was close to a constant because channel shares sum
    to one. It is removed: the models now use channel shares as features directly.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import DEFAULT_SETTINGS, Settings
from .data import TARGET


@dataclass(frozen=True)
class AlphaEstimate:
    game_alpha: float      # raw slope-based estimate (may be negative)
    n_cohorts: int
    category: str | None
    category_alpha: float
    alpha: float           # shrunk and clipped value used at serving time


def estimate_game_alpha(df: pd.DataFrame, min_cohorts: int = 4) -> tuple[float, int]:
    """Elasticity-based alpha: ROAS ~ spend^(-alpha) on month-1 cohorts.

    For small changes, 1 / (1 + alpha * inc) ~= (1 + inc)^(-alpha), so the negative
    log-log slope is the alpha of the v1 formula.
    """
    m1 = df[(df["elapsed_months"] == 1) & (df[TARGET] > 0) & (df["total_cac"] > 0)]
    m1 = m1.drop_duplicates("cohort")
    if len(m1) < min_cohorts:
        return float("nan"), len(m1)
    slope, *_ = theilslopes(np.log(m1[TARGET].to_numpy()), np.log(m1["total_cac"].to_numpy()))
    return float(-slope), len(m1)


def estimate_alphas(
    games: Mapping[str, pd.DataFrame],
    game_category: Mapping[str, str],
    settings: Settings = DEFAULT_SETTINGS,
) -> dict[str, AlphaEstimate]:
    raw = {g: estimate_game_alpha(df) for g, df in games.items()}
    by_cat: dict[str, list] = {}
    for g, (a, _) in raw.items():
        if not np.isnan(a):
            by_cat.setdefault(game_category.get(g, "unknown"), []).append(a)
    cat_alpha = {c: float(np.median(v)) for c, v in by_cat.items()}

    out = {}
    k = settings.alpha_shrinkage_k
    for g, (a, n) in raw.items():
        cat = game_category.get(g, "unknown")
        ca = cat_alpha.get(cat, settings.default_alpha)
        shrunk = ca if np.isnan(a) else (n * a + k * ca) / (n + k)
        out[g] = AlphaEstimate(a, n, cat, ca, float(np.clip(shrunk, 0.0, 2.0)))
    return out


def out_of_range_factor(
    total_cac: float, spend_min: float, spend_max: float, alpha: float, settings: Settings = DEFAULT_SETTINGS
) -> float:
    """Multiplier applied to model ROAS when spend lies outside the training range.

    Inside [spend_min, spend_max] -> 1.0 (the model handles it). Above the range the
    v1 form 1 / (1 + alpha * gamma * inc) is applied relative to spend_max; below it,
    relative to spend_min (ROAS rises as spend falls), bounded to stay finite.
    """
    if spend_min <= total_cac <= spend_max:
        return 1.0
    ref = spend_max if total_cac > spend_max else spend_min
    inc = (total_cac - ref) / ref
    combined = alpha * settings.gamma(total_cac) * inc
    combined = max(combined, -0.5)  # at most doubles ROAS when spend is cut
    return 1.0 / (1.0 + combined)
