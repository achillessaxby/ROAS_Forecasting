"""What-if scenario forecasts using the same service as the API."""
import os

import pandas as pd
import streamlit as st

from roas.config import CANONICAL_CHANNELS
from services.api.schemas import ScenarioRequest
from services.api.service import ForecastService


@st.cache_resource
def service():
    return ForecastService(os.environ.get("ROAS_REGISTRY", "registry"))


svc = service()
st.title("Scenario forecast")
games = sorted(svc.loaded.models)
game = st.selectbox("Game", games)
cohort = st.text_input("Cohort month (YYYY-MM)", "2025-10")
months = st.slider("Months", 1, 24, 12)
use_mix = st.checkbox("Specify channel spends")
if use_mix:
    spends = {c: st.number_input(c, min_value=0.0, value=0.0, step=1000.0) for c in CANONICAL_CHANNELS}
    req = ScenarioRequest(game_id=game, cohort_month=cohort, forecast_months=months,
                          channel_spends={k: v for k, v in spends.items() if v > 0} or None,
                          total_cac=None if any(spends.values()) else 1.0)
else:
    total = st.number_input("Total spend", min_value=1.0, value=100000.0, step=10000.0)
    req = ScenarioRequest(game_id=game, cohort_month=cohort, forecast_months=months, total_cac=total)
res = svc.forecast([req]).results[0]
for w in res.warnings:
    st.info(w)
df = pd.DataFrame([m.model_dump() for m in res.months]).set_index("elapsed_months")
st.line_chart(df[["cumulative_bad", "cumulative_expected", "cumulative_good"]])
st.dataframe(df)
st.caption(f"Model {svc.loaded.version} - data as of {svc.loaded.data_as_of}")
