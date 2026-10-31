"""Framework-free forecasting service used by the HTTP layer (and directly by tests).

Models and manifest are loaded once (v1 re-read CSVs from disk on every request, Q6);
``reload()`` swaps in whatever version the registry's CURRENT pointer names.
"""
from __future__ import annotations

import logging
import threading

import pandas as pd

from roas.config import DEFAULT_SETTINGS, Settings
from roas.forecast import forecast_scenario, normalise_mix
from roas.registry import LoadedVersion, Registry

from .schemas import ForecastResponse, MonthForecast, ScenarioRequest, ScenarioResult

log = logging.getLogger("roas.api")


class UnknownGame(ValueError):
    pass


class ForecastService:
    def __init__(self, registry_root: str, settings: Settings = DEFAULT_SETTINGS):
        self.registry = Registry(registry_root)
        self.settings = settings
        self._lock = threading.Lock()
        self.loaded: LoadedVersion = self.registry.load()

    def reload(self) -> dict:
        new = self.registry.load()
        with self._lock:
            self.loaded = new
        log.info("reloaded registry", extra={"model_version": new.version})
        return {"model_version": new.version, "data_as_of": new.data_as_of, "games": sorted(new.models)}

    def forecast_one(self, req: ScenarioRequest, lv: LoadedVersion) -> ScenarioResult:
        if req.game_id not in lv.models:
            raise UnknownGame(f"game {req.game_id!r} is not in model version {lv.version}")
        g = lv.game(req.game_id)
        total, shares = normalise_mix(req.channel_spends, g["default_shares"], req.total_cac)
        warnings: list[str] = []
        if not req.channel_spends:
            warnings.append("no channel_spends given: used the game's recent average channel mix")
        if req.forecast_months > g["max_elapsed_observed"]:
            warnings.append(f"months beyond {g['max_elapsed_observed']} exceed the cohort ages seen in training")
        cohort = pd.Timestamp(req.cohort_month + "-01")
        gap = (cohort.to_period("M") - pd.Timestamp(lv.data_as_of).to_period("M")).n
        if gap > 12:
            warnings.append(f"cohort month is {gap} months after the training data; seasonality only")
        in_range = g["spend_min"] <= total <= g["spend_max"]
        if not in_range:
            warnings.append("spend outside the training range: diminishing-returns adjustment applied")
        fc = forecast_scenario(lv.models[req.game_id], cohort, total, shares, req.forecast_months,
                               spend_range=(g["spend_min"], g["spend_max"]), alpha=g["alpha"], settings=self.settings)
        months = [MonthForecast(**{k: r[k] for k in MonthForecast.model_fields}) for r in fc.to_dict("records")]
        return ScenarioResult(game_id=req.game_id, cohort_month=req.cohort_month, total_cac=total,
                              channel_shares={k: round(v, 4) for k, v in shares.items() if v > 0},
                              spend_in_training_range=in_range,
                              spend_adjustment_factor=round(fc.attrs["spend_factor"], 4),
                              months=months, warnings=warnings)

    def forecast(self, scenarios: list[ScenarioRequest]) -> ForecastResponse:
        lv = self.loaded  # one consistent version for the whole batch
        results = [self.forecast_one(s, lv) for s in scenarios]
        return ForecastResponse(model_version=lv.version, data_as_of=lv.data_as_of, results=results)
