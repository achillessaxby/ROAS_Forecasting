"""Walk-forward back-test results: error vs naive baseline, band coverage, by horizon and case."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Back-test (walk-forward, out-of-sample)")
rep = Path(os.environ.get("ROAS_REPORTS", "reports"))
if not (rep / "backtest_summary.csv").exists():
    st.warning("Run `make backtest` first.")
    st.stop()
summary = pd.read_csv(rep / "backtest_summary.csv")
preds = pd.read_csv(rep / "backtest_predictions.csv")
st.caption("Case k = forecast made (k+5) months before the latest data month; models re-trained at each origin.")
overall = summary[summary.level == "overall"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("MAE (expected)", f"{overall.mae_model:.4f}", f"{-overall.skill_vs_baseline:.0%} vs baseline", delta_color="inverse")
c2.metric("25-75 band coverage", f"{overall.band_coverage:.0%}", "target ~50%", delta_color="off")
c3.metric("Predictions scored", f"{int(overall.n):,}")
st.subheader("By months ahead")
st.bar_chart(summary[summary.level == "months_ahead"].set_index("key")[["mae_model", "mae_baseline"]])
st.subheader("By case")
st.dataframe(summary[summary.level == "case"])
st.subheader("Forecast vs actual")
game = st.selectbox("Game", sorted(preds.game_id.unique()))
cohort = st.selectbox("Cohort", sorted(preds[preds.game_id == game].cohort.unique()))
case = st.selectbox("Case", sorted(preds.case.unique()))
sel = preds[(preds.game_id == game) & (preds.cohort == cohort) & (preds.case == case)].set_index("elapsed_months")
st.line_chart(sel[["actual", "bad", "expected", "good", "baseline"]])
