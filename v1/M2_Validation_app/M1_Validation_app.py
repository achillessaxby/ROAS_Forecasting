import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from m1_validation_helper import run_m1_validation, list_available_games, MAX_MONTHS

st.title("M1 Model Validation Tool")


# Sidebar
st.sidebar.header("M1 Validation Configuration")
games = list_available_games()
view_option = st.sidebar.selectbox("Select View", ["Overview"] + games)
error_method = st.sidebar.radio("Error Calculation", ["Absolute Percentage Error", "Absolute Difference Error"], index=0)

if view_option == "Overview":
    if True:  # Auto-run Overview
        all_overview_data = []
        # for model_type in ["Ridge Regression", "XGBoost", "Piecewise Linear Regression", "Interpolation Models", "Extrapolation"]:
        for model_type in ["Extrapolation"]:
            # st.header(f"Overview for {model_type}")
            all_results = {}
            game_stats_all = {}
            for game in games:
                summary_df, _, game_stats_trans, game_stats_cum = run_m1_validation(game, model_type, error_method)
                if summary_df is not None and not summary_df.empty:
                    all_results[game] = summary_df
                    game_stats_all[game] = {'trans': game_stats_trans, 'cum': game_stats_cum}
            if all_results:
                # Transactional
                overview_rows_trans = []
                for game, df in all_results.items():
                    error_cols_trans = [f"m{i}_error_trans" for i in range(1, MAX_MONTHS + 1)]
                    avg_errors_trans = {col: df[col].mean() for col in error_cols_trans}
                    row = {"game_id": game, **avg_errors_trans}
                    # Transactional summary metrics intentionally omitted from table: rmspe_trans, avg_rmse_trans, mape_trans
                    overview_rows_trans.append(row)
                overview_df_trans = pd.DataFrame(overview_rows_trans)
                st.subheader("Average Monthly Transactional Errors")
                if error_method == "Absolute Percentage Error":
                    st.dataframe(overview_df_trans.style.format({f'm{i}_error_trans': '{:.2f}%' for i in range(1, MAX_MONTHS + 1)}))
                else:
                    st.dataframe(overview_df_trans.style.format({f'm{i}_error_trans': '{:.4f}' for i in range(1, MAX_MONTHS + 1)}))

                # Cumulative
                overview_rows_cum = []
                for game, df in all_results.items():
                    error_cols_cum = [f"m{i}_error_cum" for i in range(1, MAX_MONTHS + 1)]
                    avg_errors_cum = {col: df[col].mean() for col in error_cols_cum}
                    row = {"game_id": game, **avg_errors_cum}
                    # Cumulative summary metrics intentionally omitted from table: rmspe_cum, avg_rmse_cum, mape_cum
                    overview_rows_cum.append(row)
                overview_df_cum = pd.DataFrame(overview_rows_cum)
                st.subheader("Average Monthly Cumulative Errors")
                if error_method == "Absolute Percentage Error":
                    st.dataframe(overview_df_cum.style.format({f'm{i}_error_cum': '{:.2f}%' for i in range(1, MAX_MONTHS + 1)}))
                else:
                    st.dataframe(overview_df_cum.style.format({f'm{i}_error_cum': '{:.4f}' for i in range(1, MAX_MONTHS + 1)}))
else:
    # Auto-run Extrapolation for selected game
    model_type = "Extrapolation"
    if True:  # Auto-run validation
        summary_df, forecast_plot_data, game_stats_trans, game_stats_cum = run_m1_validation(view_option, model_type, error_method)
        if summary_df is None or summary_df.empty:
            st.warning("No validation results available for this selection.")
        else:
            # Ensure monthly columns exist
            for i in range(1, MAX_MONTHS + 1):
                trans_col = f"m{i}_error_trans"
                cum_col = f"m{i}_error_cum"
                if trans_col not in summary_df.columns:
                    summary_df[trans_col] = np.nan
                if cum_col not in summary_df.columns:
                    summary_df[cum_col] = np.nan

            month_labels = [f"M{i}" for i in range(1, MAX_MONTHS + 1)]

            # Iterate cohorts like M2 and show per-cohort blocks
            try:
                cohorts_sorted = sorted(summary_df['cohort'].unique())
            except Exception:
                cohorts_sorted = summary_df['cohort'].unique()

            for cohort_str in cohorts_sorted:
                st.subheader(f"Cohort: {cohort_str}")
                row = summary_df[summary_df['cohort'] == cohort_str].iloc[0]

                # Error Table (Transactional)
                trans_vals = [row.get(f"m{i}_error_trans", np.nan) for i in range(1, MAX_MONTHS + 1)]
                trans_table = pd.DataFrame([trans_vals], columns=month_labels)
                st.markdown("Error Table (LTV/CAC)")
                if error_method == "Absolute Percentage Error":
                    st.dataframe(trans_table.style.format({label: '{:.2f}%' for label in month_labels}))
                else:
                    st.dataframe(trans_table.style.format({label: '{:.4f}' for label in month_labels}))

                # Cumulative Error Table
                cum_vals = [row.get(f"m{i}_error_cum", np.nan) for i in range(1, MAX_MONTHS + 1)]
                cum_table = pd.DataFrame([cum_vals], columns=month_labels)
                st.markdown("Cumulative Error Table (LTV/CAC)")
                if error_method == "Absolute Percentage Error":
                    st.dataframe(cum_table.style.format({label: '{:.2f}%' for label in month_labels}))
                else:
                    st.dataframe(cum_table.style.format({label: '{:.4f}' for label in month_labels}))

                # Actual vs Predicted table (Transactional)
                if cohort_str in forecast_plot_data:
                    data_dict = forecast_plot_data[cohort_str]
                    forecast_df = data_dict['forecast']
                    actual_df = data_dict['actual']
                    rows = []
                    for idx, f_row in enumerate(forecast_df.itertuples(), start=1):
                        forecast_month = f_row.forecast_month
                        pred_trans = f_row.predicted_transactional
                        act_mask = actual_df['actual_date'].dt.strftime('%Y-%m') == forecast_month
                        actual_trans = actual_df.loc[act_mask, 'transaction_period_ltv_cac'].iloc[0] if act_mask.any() else np.nan
                        # Display as percentages
                        actual_pct = (actual_trans * 100) if pd.notnull(actual_trans) else np.nan
                        pred_pct = (pred_trans * 100) if pd.notnull(pred_trans) else np.nan
                        rows.append({
                            'Month': f"M{idx}",
                            'Actual': actual_pct,
                            'Predicted': pred_pct,
                        })
                    avp_df = pd.DataFrame(rows)
                    st.markdown(f"Actual vs Predicted Table for Cohort {cohort_str} (LTV/CAC)")
                    st.dataframe(avp_df.style.format({'Actual': '{:.2f}%', 'Predicted': '{:.2f}%'}))

                # Error statistics per cohort (Transactional)
                trans_errors = pd.Series(trans_vals).dropna()
                if not trans_errors.empty:
                    trans_stats = {
                        'Mean Error': trans_errors.mean(),
                        'Min Error': trans_errors.min(),
                        'Max Error': trans_errors.max(),
                        'SD Error': trans_errors.std(ddof=0),
                        'RMSE': row.get('cohort_rmse_trans', np.nan)
                    }
                    trans_stats_df = pd.DataFrame([trans_stats])
                    st.markdown(f"Error Statistics for Cohort {cohort_str}")
                    if error_method == "Absolute Percentage Error":
                        st.dataframe(trans_stats_df.style.format({'Mean Error': '{:.2f}%', 'Min Error': '{:.2f}%', 'Max Error': '{:.2f}%', 'SD Error': '{:.2f}%', 'RMSE': '{:.2f}%'}))
                    else:
                        st.dataframe(trans_stats_df.style.format({'Mean Error': '{:.4f}', 'Min Error': '{:.4f}', 'Max Error': '{:.4f}', 'SD Error': '{:.4f}', 'RMSE': '{:.4f}'}))

                # Error statistics per cohort (Cumulative)
                cum_errors = pd.Series(cum_vals).dropna()
                if not cum_errors.empty:
                    cum_stats = {
                        'Mean Error': cum_errors.mean(),
                        'Min Error': cum_errors.min(),
                        'Max Error': cum_errors.max(),
                        'SD Error': cum_errors.std(ddof=0),
                        'RMSE': row.get('cohort_rmse_cum', np.nan)
                    }
                    cum_stats_df = pd.DataFrame([cum_stats])
                    st.markdown(f"Cumulative Error Statistics for Cohort {cohort_str}")
                    if error_method == "Absolute Percentage Error":
                        st.dataframe(cum_stats_df.style.format({'Mean Error': '{:.2f}%', 'Min Error': '{:.2f}%', 'Max Error': '{:.2f}%', 'SD Error': '{:.2f}%', 'RMSE': '{:.2f}%'}))
                    else:
                        st.dataframe(cum_stats_df.style.format({'Mean Error': '{:.4f}', 'Min Error': '{:.4f}', 'Max Error': '{:.4f}', 'SD Error': '{:.4f}', 'RMSE': '{:.4f}'}))


