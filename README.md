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
python scripts/smoke_test.py       # in a second terminal
```

Without Docker: `pip install -e ".[api,app,xgboost,dev]" && python run_demo.py && pytest` (see [v2/README.md](v2/README.md)).

## Author's role

> _To complete: what I designed, built and owned in v1 (and with whom), how it was used, and what v2 demonstrates._

## Notice

No client data, trained models or outputs are included. v2 uses synthetic data only. Shared with the client's
permission for review; not licensed for reuse.
