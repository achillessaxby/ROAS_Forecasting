"""C4 - forecasts respond to the requested scenario."""
import numpy as np
import pytest

from roas.forecast import normalise_mix
from services.api.schemas import ScenarioRequest
from services.api.service import ForecastService


@pytest.fixture(scope="module")
def svc(registry_root):
    return ForecastService(str(registry_root))


def cum12(svc, **kw):
    r = svc.forecast([ScenarioRequest(game_id="SG01", forecast_months=12, **kw)]).results[0]
    return r, r.months[-1].cumulative_expected


def test_cohort_month_changes_forecast(svc):
    _, jan = cum12(svc, cohort_month="2025-01", total_cac=150_000)
    _, jul = cum12(svc, cohort_month="2025-07", total_cac=150_000)
    assert jan != pytest.approx(jul, rel=1e-3)
    assert jan > jul  # synthetic seasonality peaks in December


def test_spend_above_range_lowers_roas(svc):
    lv = svc.loaded
    hi = lv.game("SG01")["spend_max"]
    r_in, inside = cum12(svc, cohort_month="2025-07", total_cac=hi * 0.9)
    r_out, outside = cum12(svc, cohort_month="2025-07", total_cac=hi * 3)
    assert r_in.spend_in_training_range and not r_out.spend_in_training_range
    assert outside < inside and r_out.spend_adjustment_factor < 1


def test_bands_ordered_and_cumulative_monotone(svc):
    r, _ = cum12(svc, cohort_month="2025-07", total_cac=150_000)
    for m in r.months:
        assert m.roas_bad <= m.roas_expected <= m.roas_good
    cum = [m.cumulative_expected for m in r.months]
    assert np.all(np.diff(cum) >= 0)


def test_channel_mix_is_used(svc):
    r, _ = cum12(svc, cohort_month="2025-07", channel_spends={"MC_Organic": 100_000, "MC_Tiktok": 50_000})
    assert r.channel_shares["MC_Organic"] == pytest.approx(2 / 3, abs=1e-3)
    assert r.total_cac == 150_000


def test_total_must_match_channels():
    with pytest.raises(ValueError):
        normalise_mix({"MC_Google": 100.0}, {}, total_cac=500.0)
