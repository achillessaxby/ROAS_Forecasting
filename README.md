# ROAS Forecasting Showcase

Cohort-level revenue and ROAS forecasting for a growth-capital provider that funds mobile-game user acquisition:
*"If we fund $X of UA spend for game G in month M, what will that cohort return over the next N months, and
within what range?"*

| Folder | What it is | How to use it |
|---|---|---|
| [`v1/`](v1) | **Production system (2025)**: training notebooks → FastAPI forecast service → Streamlit Validation App on Google Cloud Run. Bugs found in a later review are fixed in place; the structure is unchanged. | **View-only**: it needs the client's data and GCP. Start with [v1/README.md](v1/README.md), then [v1/docs/TECHNICAL_OVERVIEW.md](v1/docs/TECHNICAL_OVERVIEW.md). |
| [`v2/`](v2) | **Local production-style sample**: the same flow on synthetic data, with a versioned model registry, an automated back-test gate, an authenticated API and the Analytics Suite, in Docker. | **Runnable**: `cd v2 && cp .env.example .env && docker compose up --build`. Full steps in [v2/README.md](v2/README.md). |

## Quick start (v2)

```bash
cd v2
cp .env.example .env
docker compose up --build          # pipeline → API (localhost:8080) + app (localhost:8501)
python3 scripts/smoke_test.py      # in a second terminal
```

Without Docker: `pip install -e ".[api,app,xgboost,dev]" && python run_demo.py && pytest` (see [v2/README.md](v2/README.md)).

## Author's role

I was the sole builder of v1, end to end:

- **Modelling:** designed and trained the per-game M1 (spend-driven) and M2 (history-driven) XGBoost quantile models
  that produce the bad / expected / good ROAS bands, plus the category-alpha spend scaling.
- **Forecast API:** built and deployed the FastAPI service on Google Cloud Run that serves those forecasts.
- **Validation App:** built and deployed the Streamlit app the team used to compare forecasts with actuals.
- **Operations:** owned the deployment, model refreshes and data pipeline into Cloud Storage.

v2 is my own follow-up: the same approach rebuilt as a runnable local sample on synthetic data, with the
review findings from v1 fixed and covered by tests.

The v1 code here holds no credentials, keys or service URLs, so it cannot reach the client's cloud resources.

## Notice

No client data, trained models or outputs are included. v2 uses synthetic data only. Shared with the client's
permission for review; not licensed for reuse.
