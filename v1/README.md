# v1: Cohort ROAS Forecasting Platform (production, 2025)

The production system I built and ran in 2025 for a growth-capital provider that funds mobile-game user acquisition.
It forecasts, per game and acquisition cohort, monthly net ROAS and revenue with a bad / expected / good range, and
adjusts the forecast for planned spend.

> **View-only.** This folder is shared for code review. It ran on client data and Google Cloud resources that are not
> included, so it cannot be run as-is. To run the approach end to end on your machine, use [`../v2`](../v2).

**Read the full technical reference: [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md).** It covers the flow, data
contract, models, accuracy metrics, Validation App, deployment, architecture outlook and review notes.

## Flow

```
① Training notebooks           ② Forecast API                 ③ Validation App
m1m2_notebook/          ──►    fastapi-v2/app/          ──►   M2_Validation_app/
M1 + M2 quantile XGBoost       FastAPI on Cloud Run           Streamlit on Cloud Run
per game → final_M1/M2.csv     POST /predict                  back-test tables, forecasts, exports
        ▲                              ▲
        └─ per-game cohort data        └─ category_alpha.csv (fastapi-v2/category_alpha/)
```

## Folder map

| Folder | Contents |
|---|---|
| `m1m2_notebook/Models - Training/` | Batch (`m1_all`, `m2_all`) and per-game training notebooks, feature configs, the seasonality experiment |
| `m1m2_notebook/Models - Prediction*/` | Consolidation into the prediction store, error tables and confidence scores |
| `fastapi-v2/app/` | Forecast API (FastAPI) with Dockerfile |
| `fastapi-v2/category_alpha/` | Diminishing-returns coefficient (α) pipeline and projection experiments |
| `M2_Validation_app/` | Validation App (Streamlit, six tools), modular M2 framework (`M2/`), offline validation-cache generator, Dockerfile, `deploy.sh` |
| `docs/` | `TECHNICAL_OVERVIEW.md` |

## What was changed for review

- **Removed:** client data, trained models, prediction outputs and notebook outputs; workstation paths, user names and
  client infrastructure names; superseded copies (an older app version, a previous API version, run copies of
  notebooks, alternate deploy scripts, scratch scripts and dev notes).
- **Fixed in place after deployment:** the bugs a later code review found (date cutoff, random time-series split,
  TikTok channel mapping, `/refresh_models`, public deployment, silent zero predictions). The structure and design are
  unchanged. Details are in [§13 of the technical overview](docs/TECHNICAL_OVERVIEW.md#13-review-notes).
