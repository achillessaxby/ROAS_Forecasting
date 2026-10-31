"""ROAS Analytics Suite v2 - Streamlit multipage app (replaces v1's exec()-based router, Q3).

    ROAS_REGISTRY=registry ROAS_REPORTS=reports streamlit run services/app/Home.py
"""
import os

import streamlit as st

from roas.registry import Registry

st.set_page_config(page_title="ROAS Analytics Suite", layout="wide")
st.title("ROAS Analytics Suite")
reg = Registry(os.environ.get("ROAS_REGISTRY", "registry"))
try:
    lv = reg.load()
    st.metric("Live model version", lv.version)
    st.write(f"Data as of **{lv.data_as_of}** - {len(lv.models)} games - data hash `{lv.manifest['data_hash']}`")
    st.dataframe(
        [{"game": g, "category": v["category"], "cohorts": v["n_cohorts"], "alpha": round(v["alpha"], 3),
          "spend range": f"{v['spend_min']:,.0f} - {v['spend_max']:,.0f}"} for g, v in lv.manifest["games"].items()],
    )
except FileNotFoundError as e:
    st.error(f"No promoted model version found: {e}")
st.caption("Pages: Back-test (out-of-sample accuracy vs baseline) - Scenario (what-if forecasts).")
