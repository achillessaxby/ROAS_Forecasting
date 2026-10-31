"""Request / response contract for the Forecast API v2."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from roas.config import CANONICAL_CHANNELS, DEFAULT_SETTINGS
from roas.data import canonical_channel


class ScenarioRequest(BaseModel):
    game_id: str
    cohort_month: str = Field(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="Acquisition month, YYYY-MM")
    forecast_months: int = Field(12, ge=1, le=DEFAULT_SETTINGS.max_horizon)
    total_cac: float | None = Field(None, gt=0, description="Total UA spend; optional if channel_spends given")
    channel_spends: dict[str, float] | None = Field(
        None, description=f"Spend per channel. Accepted names: {', '.join(CANONICAL_CHANNELS)} (common aliases such as MC_TicToc are mapped)."
    )

    @field_validator("channel_spends")
    @classmethod
    def canonicalise(cls, v):
        """Finding C5: unknown channels are rejected (422), never silently dropped."""
        if v is None:
            return v
        out: dict[str, float] = {}
        unknown = []
        for k, amount in v.items():
            if amount is None or amount < 0:
                raise ValueError(f"spend for {k} must be >= 0")
            c = canonical_channel(k)
            if c is None:
                unknown.append(k)
            else:
                out[c] = out.get(c, 0.0) + float(amount)
        if unknown:
            raise ValueError(f"unknown channel(s) {unknown}; allowed: {CANONICAL_CHANNELS}")
        return out

    @model_validator(mode="after")
    def need_spend(self):
        if self.total_cac is None and not self.channel_spends:
            raise ValueError("provide total_cac or channel_spends")
        return self


class ForecastRequest(BaseModel):
    scenarios: list[ScenarioRequest] = Field(..., min_length=1, max_length=50)


class MonthForecast(BaseModel):
    elapsed_months: int
    calendar_month: str
    roas_bad: float
    roas_expected: float
    roas_good: float
    cumulative_bad: float
    cumulative_expected: float
    cumulative_good: float
    revenue_bad: float
    revenue_expected: float
    revenue_good: float


class ScenarioResult(BaseModel):
    game_id: str
    cohort_month: str
    total_cac: float
    channel_shares: dict[str, float]
    spend_in_training_range: bool
    spend_adjustment_factor: float
    months: list[MonthForecast]
    warnings: list[str] = []


class ForecastResponse(BaseModel):
    model_version: str
    data_as_of: str
    results: list[ScenarioResult]
