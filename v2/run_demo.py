"""One-command, cross-platform demo of the v2 POC (no `make` needed; works on Windows too).

    python run_demo.py

Steps: synthetic data -> train (as of 2025-09) -> walk-forward back-test -> promotion gate ->
sample scenario forecasts. Everything is written under ./data and ./registry (git-ignored).
Takes about 1-2 minutes on a laptop.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE)]  # run from a fresh clone even before `pip install -e .`

import pandas as pd  # noqa: E402

from roas.data import load_directory  # noqa: E402
from roas.models import resolve_backend  # noqa: E402
from roas.pipeline import load_categories, promote_if_passing, train  # noqa: E402
from roas.synthetic import generate  # noqa: E402


def step(n, text):
    print(f"\n[{n}/5] {text}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default="2025-09-01")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--registry", default="registry")
    p.add_argument("--cases", type=int, default=6)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    t0 = time.time()
    print(f"ROAS forecasting v2 POC - model backend: {resolve_backend()}")

    step(1, f"Generating synthetic data in {a.data}/ (no client data is used)")
    generate(a.data, latest=a.as_of)

    step(2, f"Training per-game quantile models on data before {a.as_of}")
    version = train(a.data, a.registry, a.as_of)
    print(f"      trained version {version} (not live yet)")

    step(3, f"Walk-forward back-test: {a.cases} forecast origins, models re-trained at each one")
    from roas.backtest import run_backtest

    games = load_directory(a.data)
    preds, summary = run_backtest(games, a.as_of, n_cases=a.cases, categories=load_categories(Path(a.data) / "game_metrics.csv"))
    out = Path(a.registry) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "backtest_predictions.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    cols = ["key", "n", "mae_model", "mae_baseline", "mae_last_cohort", "skill_vs_baseline", "skill_vs_last_cohort", "band_coverage"]
    show = summary[summary.level.isin(["overall", "months_ahead"])][cols]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(show.to_string(index=False))

    step(4, "Promotion gate: beat the naive baseline, 25-75 band coverage within 35-75%")
    ok = promote_if_passing(a.registry, version, out / "backtest_summary.csv")
    print(f"      {'PASSED - ' + version + ' is now CURRENT' if ok else 'FAILED - CURRENT unchanged'}")
    if not ok:
        return 1

    step(5, "Sample scenario forecasts through the same service the API uses")
    from services.api.schemas import ScenarioRequest
    from services.api.service import ForecastService

    svc = ForecastService(a.registry)
    scenarios = [
        ScenarioRequest(game_id="SG01", cohort_month="2025-10", forecast_months=12, total_cac=150_000),
        ScenarioRequest(game_id="SG01", cohort_month="2025-10", forecast_months=12, total_cac=600_000),
        ScenarioRequest(game_id="SG01", cohort_month="2026-04", forecast_months=12, total_cac=150_000),
        ScenarioRequest(game_id="SG01", cohort_month="2025-10", forecast_months=12,
                        channel_spends={"MC_Google": 90_000, "MC_TicToc": 40_000, "MC_Organic": 20_000}),
    ]
    resp = svc.forecast(scenarios)
    print(f"      model {resp.model_version}, data as of {resp.data_as_of}")
    print(f"      {'scenario':<44} {'12-mo cum. ROAS  bad / expected / good':>40}")
    labels = ["Oct-25 cohort, $150k", "Oct-25 cohort, $600k (above training range)",
              "Apr-26 cohort, $150k (seasonality)", "Oct-25, $150k split across channels"]
    for lab, r in zip(labels, resp.results, strict=False):
        m = r.months[-1]
        print(f"      {lab:<44} {m.cumulative_bad:>12.3f} / {m.cumulative_expected:.3f} / {m.cumulative_good:.3f}")

    print(f"\nDone in {time.time() - t0:.0f}s. Next:")
    print("  pytest                                  # 31 tests")
    print(f"  ROAS_REGISTRY={a.registry} ROAS_API_KEYS=local-demo-key uvicorn services.api.main:app --port 8080")
    print(f"  ROAS_REGISTRY={a.registry} ROAS_REPORTS={a.registry}/reports streamlit run services/app/Home.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
