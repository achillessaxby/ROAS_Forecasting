"""Synthetic cohort data that follows DATA_SCHEMA.md - no client data anywhere.

The generator has *known* structure, so tests can check that the pipeline recovers it:
  * month-of-year seasonality on both acquisition and transaction month
  * channel quality (organic and Google cohorts monetise better than TikTok)
  * diminishing returns: ROAS ~ (spend / typical spend) ** -alpha
  * geometric decay of monthly ROAS with cohort age
Dates are written day-first (dd/mm/yyyy) and some channel columns use the raw spellings seen
in production files (``MC_adwords``, ``MC_Rewarded Ads``) to exercise the cleaning code.

    python -m roas.synthetic --out data/synthetic --latest 2025-09-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CHANNELS = ["MC_Google", "MC_Facebook", "MC_Applovin", "MC_Moloco", "MC_Unity", "MC_Tiktok",
                "MC_Organic", "MC_adwords", "MC_Rewarded Ads"]
QUALITY = {"MC_Google": 1.10, "MC_adwords": 1.10, "MC_Facebook": 1.00, "MC_Applovin": 0.95,
           "MC_Moloco": 0.95, "MC_Unity": 0.90, "MC_Tiktok": 0.80, "MC_Organic": 1.40,
           "MC_Rewarded Ads": 0.85}

GAMES = [  # id, category, typical spend, month-1 ROAS, monthly decay, alpha, season amplitude
    ("SG01", "Puzzle", 150_000, 0.30, 0.78, 0.25, 0.15),
    ("SG02", "Puzzle", 60_000, 0.26, 0.80, 0.15, 0.10),
    ("SG03", "Casual", 400_000, 0.22, 0.74, 0.35, 0.20),
    ("SG04", "Casual", 90_000, 0.28, 0.76, 0.20, 0.12),
    ("SG05", "RPG", 900_000, 0.18, 0.85, 0.30, 0.08),
    ("SG06", "RPG", 250_000, 0.20, 0.83, 0.10, 0.10),
]


def _season(month: int, amp: float, peak: int = 12) -> float:
    return 1.0 + amp * np.cos(2 * np.pi * (month - peak) / 12)


def generate(out_dir: str | Path, latest: str = "2025-09-01", first_cohort: str = "2022-07-01",
             max_age: int = 24, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest = pd.Timestamp(latest).replace(day=1)
    cohorts = pd.date_range(first_cohort, latest - pd.DateOffset(months=1), freq="MS")
    meta = []
    for gid, cat, scale, r0, decay, alpha, amp in GAMES:
        base_mix = rng.dirichlet(np.ones(len(RAW_CHANNELS)) * 2)
        rows = []
        for i, c in enumerate(cohorts):
            spend = scale * (1 + 0.01 * i) * rng.lognormal(0, 0.35)
            mix = rng.dirichlet(base_mix * 40 + 0.5)
            quality = float(np.dot(mix, [QUALITY[k] for k in RAW_CHANNELS]))
            level = r0 * _season(c.month, amp) * quality * (spend / scale) ** (-alpha)
            age = min(max_age, (latest.to_period("M") - c.to_period("M")).n)
            cum = 0.0
            for m in range(1, age + 1):
                tp = c + pd.DateOffset(months=m - 1)
                roas = level * decay ** (m - 1) * _season(tp.month, amp / 2) * rng.lognormal(0, 0.12)
                cum += roas
                row = {
                    "game_id": gid, "cohort": c.strftime("%d/%m/%Y"), "transaction_period": tp.strftime("%d/%m/%Y"),
                    "elapsed_months": m, "total_cac": round(spend, 2), "client_actual_spend": round(spend, 2),
                    "transaction_period_ltv_cac": roas, "cumulative_ltv_cac": cum,
                    "net_revenue": roas * spend, "users": int(spend / 3 * decay ** (m - 1)),
                    "day_7_payer_retention": float(np.clip(rng.normal(0.4, 0.05), 0, 1)),
                }
                for k, share in zip(RAW_CHANNELS, mix, strict=False):
                    row[k] = round(spend * share, 2)
                rows.append(row)
        df = pd.DataFrame(rows)
        if gid == "SG02":  # mimic the duplicated column seen in some production files
            df["elapsed_months.1"] = df["elapsed_months"]
        df.to_csv(out / f"{gid}_Dataset_synthetic.csv", index=False)
        meta.append({"game_id": gid, "IAA_pct": float(rng.uniform(10, 60)), "IAP_pct": 0.0,
                     "Category": cat, "Genre": cat, "true_alpha": alpha})
    m = pd.DataFrame(meta)
    m["IAP_pct"] = 100 - m["IAA_pct"]
    m.to_csv(out / "game_metrics.csv", index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="roas.synthetic")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--latest", default="2025-09-01")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    print(generate(a.out, a.latest, seed=a.seed))


if __name__ == "__main__":
    main()
