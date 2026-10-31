"""Forecast API v2 (FastAPI).

Security (finding C6): every forecasting and admin route requires an ``X-API-Key`` header
matching one of the keys in ``ROAS_API_KEYS`` (inject from Secret Manager on Cloud Run).
The app refuses to start without keys unless ``ROAS_ALLOW_NO_AUTH=1`` is set explicitly for
local development. Deploy with ``--no-allow-unauthenticated`` as well (see deploy/).

Run locally:
    ROAS_REGISTRY=registry ROAS_ALLOW_NO_AUTH=1 uvicorn services.api.main:app --port 8080
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .schemas import ForecastRequest, ForecastResponse
from .service import ForecastService, UnknownGame


class JsonFormatter(logging.Formatter):
    """One JSON object per line - parsed natively by Cloud Logging (finding Q4)."""

    def format(self, record):
        payload = {"severity": record.levelname, "message": record.getMessage(), "logger": record.name}
        for k in ("model_version", "path", "status", "latency_ms", "n_scenarios"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload)


def _configure_logging():
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(registry_root: str | None = None) -> FastAPI:
    _configure_logging()
    log = logging.getLogger("roas.api")
    keys = [k for k in os.environ.get("ROAS_API_KEYS", "").split(",") if k]
    if not keys and os.environ.get("ROAS_ALLOW_NO_AUTH") != "1":
        raise RuntimeError("ROAS_API_KEYS is not set; refusing to start without authentication")

    service = ForecastService(registry_root or os.environ.get("ROAS_REGISTRY", "registry"))
    app = FastAPI(title="ROAS Forecast API", version="2.0.0")

    def require_key(key: str | None = Security(api_key_header)):
        if not keys:
            return
        if not key or not any(secrets.compare_digest(key, k) for k in keys):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        log.info("request", extra={"path": request.url.path, "status": resp.status_code,
                                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return resp

    @app.get("/health")
    def health():
        lv = service.loaded
        return {"status": "ok", "model_version": lv.version, "data_as_of": lv.data_as_of, "games": len(lv.models)}

    @app.post("/v2/forecast", response_model=ForecastResponse, dependencies=[Depends(require_key)])
    def forecast(body: ForecastRequest):
        try:
            resp = service.forecast(body.scenarios)
        except UnknownGame as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        log.info("forecast", extra={"model_version": resp.model_version, "n_scenarios": len(body.scenarios)})
        return resp

    @app.post("/v2/admin/reload", dependencies=[Depends(require_key)])
    def reload():
        return service.reload()

    app.state.service = service
    return app


app = create_app() if os.environ.get("ROAS_REGISTRY") else None  # uvicorn entry point
