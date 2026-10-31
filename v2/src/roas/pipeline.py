"""Training pipeline and command-line entry point.

    python -m roas.pipeline train    --data data/synthetic --as-of 2025-09-01 --registry registry --promote
    python -m roas.pipeline backtest --data data/synthetic --as-of 2025-09-01 --cases 6 --out reports

Replaces v1's 36 copy-pasted per-game notebooks with one parameterised function
(finding Q2). ``--as-of`` is required, so a re-run on the same data reproduces the same
models (M8); the data hash, settings and library versions are written to the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL_CHANNELS, DEFAULT_SETTINGS, Settings
from .data import TARGET, apply_window, build_training_frame, feature_columns, load_directory, month_start
from .models import QuantileModelSet
from .registry import Registry
from .scaling import estimate_alphas

log = logging.getLogger("roas.pipeline")
MIN_ROWS = 30


def load_categories(path: str | Path | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["game_id"], df["Category"], strict=False))


def game_profile(df: pd.DataFrame) -> dict:
    """Spend range and typical channel mix from month-1 rows (used when a request gives
    only total spend, and for the out-of-range adjustment)."""
    m1 = df[df["elapsed_months"] == 1].drop_duplicates("cohort").sort_values("cohort")
    spend = m1[CANONICAL_CHANNELS].sum(axis=1)
    recent = m1.tail(6)
    rs = recent[CANONICAL_CHANNELS].sum(axis=1).replace(0, np.nan)
    shares = (recent[CANONICAL_CHANNELS].div(rs, axis=0)).mean().fillna(0.0)
    return {
        "spend_min": float(m1["total_cac"].min()),
        "spend_max": float(m1["total_cac"].max()),
        "default_shares": {c: round(float(v), 6) for c, v in shares.items()},
        "n_cohorts": int(len(m1)),
        "last_cohort": m1["cohort"].max().strftime("%Y-%m") if len(m1) else None,
        "max_elapsed_observed": int(df["elapsed_months"].max()) if len(df) else 0,
        "_spend_total_check": float(spend.sum()),
    }


def data_hash(frames: Mapping[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for g in sorted(frames):
        h.update(g.encode())
        h.update(pd.util.hash_pandas_object(frames[g], index=False).values.tobytes())
    return h.hexdigest()[:16]


def train_models(
    games: Mapping[str, pd.DataFrame],
    as_of: str | pd.Timestamp,
    settings: Settings = DEFAULT_SETTINGS,
    tune: bool = False,
) -> tuple[dict[str, QuantileModelSet], dict[str, pd.DataFrame]]:
    """Fit one quantile model set per game on data strictly before ``as_of``."""
    feats = feature_columns(settings)
    models, windowed = {}, {}
    for g, df in games.items():
        w = apply_window(df, as_of, settings)
        frame = build_training_frame(w, settings)
        if len(frame) < MIN_ROWS:
            log.warning("skip %s: %d rows before %s", g, len(frame), as_of)
            continue
        params = QuantileModelSet.tune(frame, feats, TARGET, settings) if tune else {}
        m = QuantileModelSet(features=feats, quantiles=settings.quantiles, backend=settings.backend, params=params)
        m.fit(frame, frame[TARGET], settings)
        m.metadata = {"game_id": g, "trained_through": str(month_start(as_of).date()), "n_rows": len(frame)}
        models[g], windowed[g] = m, w
    return models, windowed


def train(
    data_dir: str | Path,
    registry_root: str | Path,
    as_of: str,
    categories_csv: str | Path | None = None,
    settings: Settings = DEFAULT_SETTINGS,
    tune: bool = False,
    promote: bool = False,
) -> str:
    games = load_directory(data_dir)
    categories = load_categories(categories_csv or Path(data_dir) / "game_metrics.csv")
    models, windowed = train_models(games, as_of, settings, tune)
    alphas = estimate_alphas(windowed, categories, settings)
    version = f"v{month_start(as_of):%Y-%m}"
    reg = Registry(registry_root)
    manifest = {
        "version": version,
        "as_of": str(month_start(as_of).date()),
        "data_hash": data_hash(windowed),
        "settings": settings.to_dict(),
        "games": {},
    }
    for g, m in models.items():
        m.save(reg.version_dir(version) / g)
        prof = game_profile(windowed[g])
        prof.pop("_spend_total_check")
        a = alphas[g]
        manifest["games"][g] = {**prof, "category": a.category, "alpha": a.alpha,
                                "alpha_raw": None if np.isnan(a.game_alpha) else a.game_alpha,
                                "category_alpha": a.category_alpha}
    reg.write_manifest(version, manifest)
    if promote:
        reg.promote(version)
    log.info("trained %d games -> %s (promoted=%s)", len(models), version, promote)
    return version


GATE = {"min_skill_vs_baseline": 0.0, "coverage_low": 0.35, "coverage_high": 0.75}


def promote_if_passing(registry_root: str | Path, version: str, summary_csv: str | Path, gate: dict = GATE) -> bool:
    """Automated promotion gate: move CURRENT to ``version`` only if the walk-forward back-test
    beats the naive baseline and the 25-75 band coverage is within bounds. Results are written
    into the version's manifest either way, so every version carries its own evidence."""
    import json

    s = pd.read_csv(summary_csv)
    overall = s[s["level"] == "overall"].iloc[0]
    checks = {
        "beats_baseline": bool(overall["skill_vs_baseline"] > gate["min_skill_vs_baseline"]),
        "coverage_in_bounds": bool(gate["coverage_low"] <= overall["band_coverage"] <= gate["coverage_high"]),
    }
    passed = all(checks.values())
    reg = Registry(registry_root)
    mpath = reg.version_dir(version) / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["backtest"] = {"gate": gate, "checks": checks, "passed": passed,
                            **{k: float(overall[k]) for k in ("mae_model", "mae_baseline", "skill_vs_baseline",
                                                              "band_coverage", "pinball")}}
    reg.write_manifest(version, manifest)
    if passed:
        reg.promote(version)
    log.info("gate %s for %s: %s", "PASSED" if passed else "FAILED", version, checks)
    return passed


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="roas.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--registry", required=True)
    t.add_argument("--as-of", required=True)
    t.add_argument("--categories")
    t.add_argument("--tune", action="store_true")
    t.add_argument("--promote", action="store_true")
    b = sub.add_parser("backtest")
    b.add_argument("--data", required=True)
    b.add_argument("--as-of", required=True, help="latest data month; origins step back from here")
    b.add_argument("--cases", type=int, default=6)
    b.add_argument("--horizon", type=int, default=12)
    b.add_argument("--out", required=True)
    g = sub.add_parser("promote", help="promote a version only if its back-test passes the gate")
    g.add_argument("--registry", required=True)
    g.add_argument("--version", required=True)
    g.add_argument("--summary", required=True)
    args = p.parse_args(argv)
    if args.cmd == "promote":
        ok = promote_if_passing(args.registry, args.version, args.summary)
        print("PROMOTED" if ok else "GATE FAILED: CURRENT unchanged")
        raise SystemExit(0 if ok else 1)
    if args.cmd == "train":
        print(train(args.data, args.registry, args.as_of, args.categories, tune=args.tune, promote=args.promote))
    else:
        from .backtest import run_backtest

        games = load_directory(args.data)
        cats = load_categories(Path(args.data) / "game_metrics.csv")
        preds, summary = run_backtest(games, args.as_of, n_cases=args.cases, horizon=args.horizon, categories=cats)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out / "backtest_predictions.csv", index=False)
        summary.to_csv(out / "backtest_summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
