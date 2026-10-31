"""Loading, cleaning and feature engineering for per-game cohort data.

Review findings addressed here
------------------------------
C1  Training cutoff. v1 built the "current month" as the string "1/10/2026" and compared
    it with parsed dates; pandas read it month-first (10 Jan 2026) and silently dropped up
    to eleven months of the newest data. Here every date is parsed with an explicit format
    and the cutoff is a ``pd.Timestamp`` passed in as ``as_of`` (never ``datetime.now()``),
    which also makes training reproducible (M8).
C5  Channel names are canonicalised through an explicit alias map; unknown channels are
    summed into ``MC_Other`` and reported, never dropped.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL_CHANNELS, CHANNEL_ALIASES, DEFAULT_SETTINGS, Settings

log = logging.getLogger(__name__)

TARGET = "transaction_period_ltv_cac"
DATE_FORMAT = "%d/%m/%Y"  # source files use day-first dates


# --------------------------------------------------------------------------- dates
def parse_date(values: pd.Series | Iterable[str]) -> pd.Series:
    """Parse source dates explicitly as day-first ``dd/mm/yyyy``; fall back to ISO.

    Never relies on pandas' format inference, which is what caused C1.
    """
    s = pd.Series(values)
    out = pd.to_datetime(s, format=DATE_FORMAT, errors="coerce")
    missing = out.isna() & s.notna()
    if missing.any():
        out.loc[missing] = pd.to_datetime(s[missing], format="ISO8601", errors="coerce")
    return out


def month_start(as_of: str | pd.Timestamp) -> pd.Timestamp:
    """First day of the month containing ``as_of`` as a Timestamp (not a string)."""
    ts = pd.Timestamp(as_of)
    return ts.normalize().replace(day=1)


# --------------------------------------------------------------------------- channels
def canonical_channel(name: str) -> str | None:
    """Map a raw channel column/key to its canonical name, or None if unknown."""
    if name in CANONICAL_CHANNELS:
        return name
    return CHANNEL_ALIASES.get(name.strip().lower().replace(" ", "_").replace("-", "_"))


def canonicalise_channels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame whose channel columns are exactly ``CANONICAL_CHANNELS``.

    Unknown ``MC_*`` columns are summed into ``MC_Other``; their names are returned so the
    caller can log them (v1 dropped some of them silently).
    """
    df = df.copy()
    raw_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("mc_")]
    out = {c: np.zeros(len(df)) for c in CANONICAL_CHANNELS}
    unmapped: list[str] = []
    for col in raw_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        target = canonical_channel(col)
        if target is None:
            unmapped.append(col)
            target = "MC_Other"
        out[target] = out[target] + vals
    df = df.drop(columns=raw_cols)
    for c in CANONICAL_CHANNELS:
        df[c] = out[c]
    if unmapped:
        log.info("Channels mapped to MC_Other: %s", sorted(unmapped))
    return df, unmapped


# --------------------------------------------------------------------------- loading
def load_game(path: str | Path) -> pd.DataFrame:
    """Read one ``<GAME>_Dataset_*.csv`` and return a cleaned, typed frame."""
    df = pd.read_csv(path)
    # v1 files sometimes carry a duplicated ``elapsed_months`` column (``elapsed_months.1``)
    df = df.loc[:, ~df.columns.str.endswith(".1")]
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort"] = parse_date(df["cohort"])
    df["transaction_period"] = parse_date(df["transaction_period"])
    df = df.dropna(subset=["cohort", "transaction_period", "elapsed_months"])
    df["elapsed_months"] = df["elapsed_months"].astype(int)

    # Spend reconciliation (same rule as v1, vectorised)
    tc = pd.to_numeric(df.get("total_cac"), errors="coerce")
    cas = pd.to_numeric(df.get("client_actual_spend", pd.Series(np.nan, index=df.index)), errors="coerce")
    tc = tc.where(tc.fillna(0) > 0, cas)
    df["total_cac"] = tc
    df = df[df["total_cac"].fillna(0) > 0]

    df, _ = canonicalise_channels(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df.sort_values(["cohort", "elapsed_months"]).reset_index(drop=True)


def apply_window(df: pd.DataFrame, as_of: str | pd.Timestamp, settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Keep cohorts acquired on/after ``recency_start`` and transactions strictly before
    the month of ``as_of`` (the C1 fix: Timestamp-to-Timestamp comparison)."""
    cutoff = month_start(as_of)
    start = pd.Timestamp(settings.recency_start)
    keep = (df["cohort"] >= start) & (df["transaction_period"] < cutoff)
    return df.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- features
def channel_share_columns() -> list[str]:
    return [f"share_{c}" for c in CANONICAL_CHANNELS]


def feature_columns(settings: Settings = DEFAULT_SETTINGS) -> list[str]:
    return (
        ["log_total_cac", "cohort_month", "current_month", "elapsed_months"]
        + channel_share_columns()
        + [f"lag_{k}" for k in range(1, settings.n_lags + 1)]
    )


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at acquisition time: spend, channel mix, calendar."""
    df = df.copy()
    spend = df[CANONICAL_CHANNELS].sum(axis=1)
    denom = spend.where(spend > 0, np.nan)
    for c in CANONICAL_CHANNELS:
        df[f"share_{c}"] = (df[c] / denom).fillna(0.0)
    df["log_total_cac"] = np.log1p(df["total_cac"].astype(float))
    df["cohort_month"] = df["cohort"].dt.month
    # calendar month of the transaction; elapsed_months is 1-indexed (month 1 = acquisition month)
    df["current_month"] = ((df["cohort_month"] - 1 + df["elapsed_months"] - 1) % 12) + 1
    return df


def add_lags(df: pd.DataFrame, settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """ROAS lags aligned on ``elapsed_months`` within each cohort.

    v1 used ``groupby().shift()``, which mis-aligns lags when a month is missing. Here each
    lag is a join on (cohort, elapsed_months - k), so a gap yields NaN instead of a wrong value.
    """
    df = df.copy()
    base = df[["cohort", "elapsed_months", TARGET]]
    for k in range(1, settings.n_lags + 1):
        lag = base.rename(columns={TARGET: f"lag_{k}"}).copy()
        lag["elapsed_months"] = lag["elapsed_months"] + k
        df = df.merge(lag, on=["cohort", "elapsed_months"], how="left")
    return df


def build_training_frame(df: pd.DataFrame, settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    df = add_lags(add_static_features(df), settings)
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def load_directory(data_dir: str | Path, pattern: str = "*_Dataset_*.csv") -> dict[str, pd.DataFrame]:
    """Load every game file in a folder -> {game_id: cleaned frame}."""
    games = {}
    for p in sorted(Path(data_dir).glob(pattern)):
        game_id = p.name.split("_Dataset_")[0]
        games[game_id] = load_game(p)
    return games
