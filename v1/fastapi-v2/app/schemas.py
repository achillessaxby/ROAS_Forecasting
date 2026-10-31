# app/schemas.py
from pydantic import BaseModel, Field, validator, field_serializer
from typing import Optional, Dict, List
from datetime import date, datetime

class ForecastRequest(BaseModel):
    game_id: str
    use_market_channels: bool = False
    cohort_month: date = Field(..., description="Start date for the cohort")
    forecast_months: int = Field(..., gt=0, le=60, description="Number of months to forecast")
    # If not using market channels, total_cac is required
    # If using market channels, channel_spends is optional (will use nearest cohort's distribution if not provided)
    total_cac: Optional[float] = Field(None, description="Total spend amount (required when not using segmentation)")
    channel_spends: Optional[Dict[str, float]] = Field(None, description="Marketing channel spends for the 8 standard channels (MC_Google, MC_Facebook, MC_Applovin, MC_Moloco, MC_Unity, MC_Ironsource, MC_Tiktok, MC_Organic); unknown channels are rejected")

    @validator("cohort_month", pre=True)
    def parse_cohort_month(cls, value):
        if isinstance(value, str):
            try:
                # Assume yyyy-mm input and add "-01" for the day
                return datetime.strptime(value + "-01", "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("cohort_month must be in yyyy-mm format")
        return value

class MonthlyForecast(BaseModel):
    elapsed_months: int
    current_month: str
    # Only keeping expected predictions, removing bad and good
    transactional_expected: float
    cumulative_expected: float
    absolute_revenue_expected: float

class ForecastResult(BaseModel):
    game_id: str
    cohort_month: date
    forecasts: List[MonthlyForecast]

    @field_serializer('cohort_month')
    def serialize_cohort_month(self, value: date) -> str:
        return value.strftime('%Y-%m')


class BulkForecastRequest(BaseModel):
    requests: List[ForecastRequest]

class BulkForecastResponse(BaseModel):
    status_code: int
    runtime_seconds: float
    results: List[ForecastResult]