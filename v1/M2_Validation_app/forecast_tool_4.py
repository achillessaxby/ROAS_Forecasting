import streamlit as st
import pandas as pd
import numpy as np
import os
import cloudpickle
import json
from datetime import datetime
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from inference import available_game_ids
from google.cloud import storage
import io

EPSILON = 1e-6

if os.environ.get("PAGE_CONFIG_SET") != "1":
    st.set_page_config(page_title='M1 6 Months Forecasting Tool', layout='wide')
st.title('M1 6 Months Forecasting Tool')

# Global variables for extrapolation helper data
game_metrics_df = None
category_averages_df = None
game_to_category = {}

# GCS configuration
BUCKET_NAME = "m1m2_notebook"
DATA_PATH = "Models - Training/Latest Data/Segregated/"
HELPER_PATH = ""  # helper CSVs location if different, leave empty if bucket root

@st.cache_resource
def get_gcs_client():
    try:
        return storage.Client()
    except Exception:
        return None

storage_client = get_gcs_client()

def read_csv_from_gcs(bucket_name, blob_name):
    try:
        if storage_client is None:
            return None
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None

def qualifies_game(game_id):
    file_suffix = "_Dataset_12_2024.csv"
    blob_name = f"{DATA_PATH}{game_id}{file_suffix}"
    try:
        df = read_csv_from_gcs(BUCKET_NAME, blob_name)
        if df is None:
            return False
        if 'cohort' not in df.columns:
            return False
        df['cohort'] = pd.to_datetime(df['cohort'], format='%d/%m/%Y', errors='coerce')
        distinct = df['cohort'].dropna().nunique()
        return distinct >= 8
    except Exception:
        return False

all_game_list = available_game_ids

# Show all available games; individual runs will handle missing data gracefully
filtered_game_list = all_game_list

# Sidebar configuration
st.sidebar.header("📊 Configuration")
st.sidebar.subheader("Error Calculation Theme")
error_method = st.sidebar.radio(
    "Choose Error Calculation Method:",
    ["Absolute Percentage Error", "Absolute Difference Error"],
    index=0
)

# Show active configuration
if error_method == "Absolute Percentage Error":
    st.sidebar.success("🎯 Active: Absolute Percentage Error")
    st.sidebar.info("📈 Using: percentage error columns")
else:
    st.sidebar.success("🎯 Active: Absolute Difference Error") 
    st.sidebar.info("📊 Using: absolute error columns")

st.sidebar.subheader("Select Analysis Type:")
view_option = st.sidebar.selectbox("", ["Overview"] + filtered_game_list)
model_type = "Extrapolation"

def predict_interpolation_model1(model_package, X_new):
    x_sorted = model_package['x_sorted']
    y_sorted = model_package['y_sorted']
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
    total_cac_values = X_new['total_cac'].astype(float).values
    predictions = interp_func(total_cac_values)
    predictions = np.maximum(predictions, EPSILON)
    return predictions[0]

def predict_interpolation_model23(model_package, X_new):
    scaler = model_package['scaler']
    weights = model_package['weights']
    features = model_package['features']
    x_sorted = model_package['x_sorted']
    y_sorted = model_package['y_sorted']
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    for feat in features:
        if feat not in X_new.columns:
            if feat.startswith('MC_'):
                X_new[feat] = 0.0
            else:
                raise ValueError(f"Required feature {feat} missing in prediction data")
    X_subset = X_new[features].copy()
    for col in X_subset.columns:
        X_subset[col] = pd.to_numeric(X_subset[col], errors='coerce')
    X_scaled = scaler.transform(X_subset)
    composite_feature = np.sum(X_scaled * np.array([weights[feat] for feat in features]), axis=1)
    predictions = interp_func(composite_feature)
    predictions = np.maximum(predictions, EPSILON)
    return predictions[0]

def get_prediction(model, X_new, model_type):
    if model_type == "Interpolation Models":
        if 'features' in model:
            return predict_interpolation_model23(model, X_new)
        elif 'x_sorted' in model and 'y_sorted' in model:
            return predict_interpolation_model1(model, X_new)
        else:
            raise ValueError("Unknown model structure for Interpolation Models")
    elif model_type == "Piecewise Linear Regression":
        if isinstance(model, dict):
            pwlf_model = model['pwlf_model']
            scaler = model['scaler']
            weights = model['weights']
            features = model['features']
            for feat in features:
                if feat not in X_new.columns:
                    if feat.startswith('MC_'):
                        X_new[feat] = 0.0
                    else:
                        raise ValueError(f"Required feature {feat} missing in prediction data")
            X_subset = X_new[features].copy()
            for col in X_subset.columns:
                X_subset[col] = pd.to_numeric(X_subset[col], errors='coerce')
            X_scaled = scaler.transform(X_subset)
            composite_feature = sum(X_scaled[0][i] * weights[feat] for i, feat in enumerate(features))
            return pwlf_model.predict([composite_feature])[0]
        else:
            x = X_new['total_cac'].astype(float).values
            return model.predict(x)[0]
    else:
        try:
            if hasattr(model, 'forecast') and callable(model.forecast):
                if hasattr(model, 'model') and getattr(model.model.data, 'orig_exog', None) is not None:
                    pred_log = model.forecast(steps=1, exog=X_new.values)[0]
                else:
                    pred_log = model.forecast(steps=1)[0]
                pred = np.exp(pred_log)
                return np.clip(pred, a_min=EPSILON, a_max=None)
            else:
                return model.predict(X_new)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return np.nan

def forecast_cascaded_from_actual(total_cac, init_lag1, init_lag2, init_cum, forecast_start_date, months, model1, model2, model3, model_type):
    results = []
    lag1 = init_lag1
    lag2 = init_lag2
    cumulative = init_cum
    for m in range(1, months+1):
        current_date = forecast_start_date + pd.DateOffset(months=m-1)
        current_month = current_date.month
        if pd.isna(lag1):
            X_new = pd.DataFrame([[float(total_cac)]], columns=['total_cac'])
            pred = get_prediction(model1, X_new, model_type)
        elif pd.isna(lag2):
            X_new = pd.DataFrame([{'total_cac': float(total_cac), 'ltv_lag_1': float(lag1)}])
            pred = get_prediction(model2, X_new, model_type)
        else:
            if model_type in ["Piecewise Linear Regression", "Interpolation Models"]:
                features = model3['features']
                X_new = pd.DataFrame(columns=features)
                X_new.loc[0, 'total_cac'] = float(total_cac)
                X_new.loc[0, 'ltv_lag_1'] = float(lag1)
                X_new.loc[0, 'ltv_lag_2'] = float(lag2)
                X_new.loc[0, 'current_month'] = int(current_month)
                for feat in features:
                    if feat.startswith('MC_') and pd.isna(X_new.loc[0, feat]):
                        X_new.loc[0, feat] = 0.0
                pred = get_prediction(model3, X_new, model_type)
            else:
                X_new = pd.DataFrame([{'total_cac': float(total_cac), 'ltv_lag_1': float(lag1), 'ltv_lag_2': float(lag2)}])
                pred = get_prediction(model3, X_new, model_type)
        cumulative += pred
        results.append({
            'forecast_month': current_date.strftime('%Y-%m'),
            'predicted_transactional': pred,
            'predicted_cumulative': cumulative
        })
        lag2 = lag1
        lag1 = pred
    return pd.DataFrame(results)

def forecast_xgb(model, total_cac, cohort_month, start_date, months, mc_features):
    results = []
    lag1 = 0
    lag2 = 0
    lag3 = 0
    cumulative = 0
    for m in range(1, months+1):
        current_date = start_date + pd.DateOffset(months=m-1)
        current_month = current_date.month
        elapsed_months = m
        X_new = pd.DataFrame([{
            'total_cac': float(total_cac),
            'cohort_month': int(cohort_month),
            'current_month': int(current_month),
            'ltv_lag_1': float(lag1),
            'ltv_lag_2': float(lag2),
            'ltv_lag_3': float(lag3),
            'elapsed_months': int(elapsed_months)
        }])
        for mc in mc_features:
            X_new[mc] = 0.0
        pred = model.predict(X_new)[0]
        cumulative += pred
        results.append({
            'forecast_month': current_date.strftime('%Y-%m'),
            'predicted_transactional': pred,
            'predicted_cumulative': cumulative
        })
        lag3 = lag2
        lag2 = lag1
        lag1 = pred
    return pd.DataFrame(results)

def get_base_val(row):
    if pd.isnull(row['total_cac']) or row['total_cac'] == 0:
        return row['client_actual_spend'] if ('client_actual_spend' in row and pd.notnull(row['client_actual_spend'])) else np.nan
    else:
        return row['total_cac']

def load_helper_data():
    global game_metrics_df, category_averages_df, game_to_category
    try:
        # Avoid reloading repeatedly
        if game_metrics_df is not None and category_averages_df is not None and game_to_category:
            return True

        # Read helper CSVs from the project workspace (as requested)
        gm_df = pd.read_csv('game_metrics_output.csv')
        ca_df = pd.read_csv('category_averages_output.csv')

        # Basic schema checks
        required_gm_cols = {'game_id', 'Category', 'IAA_pct', 'IAP_pct'}
        required_ca_cols = {'Category', 'IAA_pct', 'IAP_pct'}
        if not required_gm_cols.issubset(set(gm_df.columns)):
            st.error("game_metrics_output.csv is missing required columns: game_id, Category, IAA_pct, IAP_pct")
            return False
        if not required_ca_cols.issubset(set(ca_df.columns)):
            st.error("category_averages_output.csv is missing required columns: Category, IAA_pct, IAP_pct")
            return False

        game_metrics_df = gm_df
        category_averages_df = ca_df
        game_to_category = dict(zip(game_metrics_df['game_id'], game_metrics_df['Category']))
        return True
    except Exception as e:
        st.error(f"Error loading helper data: {e}")
        return False

def get_gamma(total_cac: float) -> float:
    if total_cac < 300000:
        return 1.0
    elif total_cac < 1000000:
        return 1.4
    else:
        return 1.7

def compute_alpha_prime(game_id: str) -> float:
    default_alpha = 0.05
    try:
        # If helpers weren't loaded, fall back silently
        if game_metrics_df is None or category_averages_df is None or not game_to_category:
            return default_alpha
        category = game_to_category.get(game_id)
        if not category:
            return default_alpha
        game_row = game_metrics_df[game_metrics_df['game_id'] == game_id]
        if game_row.empty:
            return default_alpha
        category_row = category_averages_df[category_averages_df['Category'] == category]
        if category_row.empty:
            return default_alpha
        game_IAA = game_row['IAA_pct'].values[0] / 100
        game_IAP = game_row['IAP_pct'].values[0] / 100
        category_IAA = category_row['IAA_pct'].values[0] / 100
        category_IAP = category_row['IAP_pct'].values[0] / 100
        delta_IAA = game_IAA - category_IAA
        delta_IAP = game_IAP - category_IAP
        category_alpha = 0.05
        alpha_prime = category_alpha * (game_IAA * (1 + delta_IAA) + game_IAP * (1 + delta_IAP))
        return max(0.0, alpha_prime)
    except Exception:
        return default_alpha

def run_cohort_forecasts(game_id, model_type, error_method="Absolute Percentage Error"):
    all_signed_error_pct_trans = []
    all_signed_error_pct_cum = []
    if model_type == "Extrapolation":
        if not load_helper_data():
            return None, None, None, None
        
        file_suffix = "_Dataset_12_2024.csv"
        blob_name = f"{DATA_PATH}{game_id}{file_suffix}"
        df_game = read_csv_from_gcs(BUCKET_NAME, blob_name)
        if df_game is None:
            return None, None, None, None
        if 'total_cac' not in df_game.columns:
            st.error(f"Column 'total_cac' missing in data for game {game_id}")
            return None, None, None, None
        
        df_game['transaction_period_date'] = pd.to_datetime(df_game['transaction_period'], errors='coerce', infer_datetime_format=True)
        df_game['cohort'] = pd.to_datetime(df_game['cohort'], format='%d/%m/%Y', errors='coerce')
        df_game = df_game.dropna(subset=['elapsed_months'])
        df_game['actual_date'] = df_game.apply(lambda row: row['cohort'] + pd.DateOffset(months=int(row['elapsed_months']) - 1), axis=1)
        
        all_cohorts = np.sort(df_game['cohort'].dropna().unique())
        if len(all_cohorts) < 8:
            return None, None, None, None
        valid_cohorts = [co for co in all_cohorts if df_game[df_game['cohort'] == co]['transaction_period_ltv_cac'].gt(0).any()]
        if len(valid_cohorts) < 7:
            return None, None, None, None
        
        base_cohort = valid_cohorts[-7]
        predicted_cohorts = list(valid_cohorts[-6:])
        
        base_df = df_game[df_game['cohort'] == base_cohort].sort_values('elapsed_months')
        if 'client_actual_spend' in base_df.columns:
            base_df['base_input'] = base_df.apply(get_base_val, axis=1)
        else:
            base_df['base_input'] = base_df['total_cac']
        
        try:
            base_cac = base_df['base_input'].dropna().iloc[0]
        except IndexError:
            st.warning(f"No valid 'base_input' for base cohort {base_cohort} in game {game_id}")
            return None, None, None, None
        
        base_predictions = {}
        for m in range(1, 7):
            m_df = base_df[base_df['elapsed_months'] == m]
            base_predictions[m] = m_df['transaction_period_ltv_cac'].iloc[0] if not m_df.empty else 0.0
        
        alpha_prime = compute_alpha_prime(game_id)
        
        summary_rows = []
        forecast_plot_data = {}
        all_diff_trans = []
        all_diff_cum = []
        diff_trans_per_month = {i: [] for i in range(1,7)}
        diff_cum_per_month = {i: [] for i in range(1,7)}
        
        for cohort_date in predicted_cohorts:
            cohort_str = pd.Timestamp(cohort_date).strftime('%Y-%m-%d')
            cohort_df = df_game[df_game['cohort'] == cohort_date].copy().sort_values(by='actual_date')
            if 'client_actual_spend' in cohort_df.columns:
                cohort_df['base_input'] = cohort_df.apply(get_base_val, axis=1)
            else:
                cohort_df['base_input'] = cohort_df['total_cac']
            
            try:
                total_cac_val = cohort_df['base_input'].dropna().iloc[0]
            except IndexError:
                continue
            if total_cac_val <= 0:
                continue
            
            inc = (total_cac_val - base_cac) / base_cac if base_cac > 0 else 0
            gamma = get_gamma(total_cac_val)
            adjustment_factor = 1 / (1 + alpha_prime * gamma * inc)
            
            predicted_transactional = [base_predictions.get(m, 0.0) * adjustment_factor for m in range(1, 7)]
            predicted_cumulative = np.cumsum(predicted_transactional)
            
            forecast_start_date = cohort_date
            forecast_df = pd.DataFrame({
                'forecast_month': [(forecast_start_date + pd.DateOffset(months=m-1)).strftime('%Y-%m') for m in range(1, 7)],
                'predicted_transactional': predicted_transactional,
                'predicted_cumulative': predicted_cumulative
            })
            
            actual_df = cohort_df[['elapsed_months', 'actual_date', 'transaction_period_ltv_cac']].drop_duplicates().sort_values(by='actual_date')
            actual_df['actual_cumulative'] = actual_df['transaction_period_ltv_cac'].cumsum()
            
            row = {"game_id": game_id, "cohort": cohort_str, "spends": total_cac_val}
            cohort_diff_trans = []
            cohort_diff_cum = []
            for idx, f_row in enumerate(forecast_df.itertuples(), start=1):
                forecast_month = f_row.forecast_month
                pred_trans = f_row.predicted_transactional
                pred_cum = f_row.predicted_cumulative
                act_mask = actual_df['actual_date'].dt.strftime('%Y-%m') == forecast_month
                if act_mask.any():
                    actual_trans = actual_df.loc[act_mask, 'transaction_period_ltv_cac'].iloc[0]
                    actual_cum = actual_df.loc[act_mask, 'actual_cumulative'].iloc[0]
                    diff_trans = pred_trans - actual_trans if pd.notnull(actual_trans) else np.nan
                    diff_cum = pred_cum - actual_cum if pd.notnull(actual_cum) else np.nan
                    if error_method == "Absolute Percentage Error":
                        if pd.notnull(diff_trans) and actual_trans != 0:
                            signed_error_pct_trans = (diff_trans / actual_trans) * 100
                            all_signed_error_pct_trans.append(signed_error_pct_trans)
                            abs_error_pct_trans = abs(signed_error_pct_trans)
                            row[f"m{idx}_error_trans"] = abs_error_pct_trans
                        else:
                            row[f"m{idx}_error_trans"] = np.nan
                    else:  # Absolute Difference Error
                        if pd.notnull(diff_trans):
                            abs_error_abs_trans = abs(diff_trans)
                            all_signed_error_pct_trans.append(diff_trans)  # Store raw difference for stats
                            row[f"m{idx}_error_trans"] = abs_error_abs_trans
                        else:
                            row[f"m{idx}_error_trans"] = np.nan
                    if error_method == "Absolute Percentage Error":
                        if pd.notnull(diff_cum) and actual_cum != 0:
                            signed_error_pct_cum = (diff_cum / actual_cum) * 100
                            all_signed_error_pct_cum.append(signed_error_pct_cum)
                            abs_error_pct_cum = abs(signed_error_pct_cum)
                            row[f"m{idx}_error_cum"] = abs_error_pct_cum
                        else:
                            row[f"m{idx}_error_cum"] = np.nan
                    else:  # Absolute Difference Error
                        if pd.notnull(diff_cum):
                            abs_error_abs_cum = abs(diff_cum)
                            all_signed_error_pct_cum.append(diff_cum)  # Store raw difference for stats
                            row[f"m{idx}_error_cum"] = abs_error_abs_cum
                        else:
                            row[f"m{idx}_error_cum"] = np.nan
                    if pd.notnull(diff_trans):
                        cohort_diff_trans.append(diff_trans)
                        all_diff_trans.append(diff_trans)
                        diff_trans_per_month[idx].append(diff_trans)
                    if pd.notnull(diff_cum):
                        cohort_diff_cum.append(diff_cum)
                        all_diff_cum.append(diff_cum)
                        diff_cum_per_month[idx].append(diff_cum)
                else:
                    row[f"m{idx}_error_trans"] = np.nan
                    row[f"m{idx}_error_cum"] = np.nan
            row["cohort_rmse_trans"] = np.sqrt(np.mean(np.square(cohort_diff_trans))) * 100 if cohort_diff_trans else np.nan
            row["cohort_rmse_cum"] = np.sqrt(np.mean(np.square(cohort_diff_cum))) * 100 if cohort_diff_cum else np.nan
            summary_rows.append(row)
            forecast_plot_data[cohort_str] = {'forecast': forecast_df, 'actual': actual_df}
        
        summary_df = pd.DataFrame(summary_rows)
        
        game_stats_trans = {}
        for m in range(1, 7):
            if diff_trans_per_month[m]:
                game_stats_trans[f"rmse_m{m}_trans"] = np.sqrt(np.mean(np.square(diff_trans_per_month[m]))) * 100
            else:
                game_stats_trans[f"rmse_m{m}_trans"] = np.nan
        if all_signed_error_pct_trans:
            if error_method == "Absolute Percentage Error":
                game_stats_trans["rmspe_trans"] = np.sqrt(np.mean(np.square(all_signed_error_pct_trans)))
                abs_error_pct_trans = [abs(e) for e in all_signed_error_pct_trans]
                game_stats_trans["mape_trans"] = np.mean(abs_error_pct_trans)
                game_stats_trans["min_error_pct_trans"] = min(abs_error_pct_trans)
                game_stats_trans["max_error_pct_trans"] = max(abs_error_pct_trans)
                game_stats_trans["mean_error_pct_trans"] = game_stats_trans["mape_trans"]
                game_stats_trans["sd_error_pct_trans"] = np.std(abs_error_pct_trans)
            else:  # Absolute Difference Error
                game_stats_trans["rmspe_trans"] = np.sqrt(np.mean(np.square(all_signed_error_pct_trans)))
                abs_error_abs_trans = [abs(e) for e in all_signed_error_pct_trans]
                game_stats_trans["mape_trans"] = np.mean(abs_error_abs_trans)
                game_stats_trans["min_error_pct_trans"] = min(abs_error_abs_trans)
                game_stats_trans["max_error_pct_trans"] = max(abs_error_abs_trans)
                game_stats_trans["mean_error_pct_trans"] = game_stats_trans["mape_trans"]
                game_stats_trans["sd_error_pct_trans"] = np.std(abs_error_abs_trans)
            rmse_values_trans = [game_stats_trans[f"rmse_m{m}_trans"] for m in range(1, 7) if pd.notnull(game_stats_trans[f"rmse_m{m}_trans"])]
            game_stats_trans["avg_rmse_trans"] = np.mean(rmse_values_trans) if rmse_values_trans else np.nan
        else:
            game_stats_trans["rmspe_trans"] = np.nan
            game_stats_trans["mape_trans"] = np.nan
            game_stats_trans["min_error_pct_trans"] = np.nan
            game_stats_trans["max_error_pct_trans"] = np.nan
            game_stats_trans["mean_error_pct_trans"] = np.nan
            game_stats_trans["sd_error_pct_trans"] = np.nan
            game_stats_trans["avg_rmse_trans"] = np.nan
        
        game_stats_cum = {}
        for m in range(1, 7):
            if diff_cum_per_month[m]:
                game_stats_cum[f"rmse_m{m}_cum"] = np.sqrt(np.mean(np.square(diff_cum_per_month[m]))) * 100
            else:
                game_stats_cum[f"rmse_m{m}_cum"] = np.nan
        if all_signed_error_pct_cum:
            if error_method == "Absolute Percentage Error":
                game_stats_cum["rmspe_cum"] = np.sqrt(np.mean(np.square(all_signed_error_pct_cum)))
                abs_error_pct_cum = [abs(e) for e in all_signed_error_pct_cum]
                game_stats_cum["mape_cum"] = np.mean(abs_error_pct_cum)
                game_stats_cum["min_error_pct_cum"] = min(abs_error_pct_cum)
                game_stats_cum["max_error_pct_cum"] = max(abs_error_pct_cum)
                game_stats_cum["mean_error_pct_cum"] = game_stats_cum["mape_cum"]
                game_stats_cum["sd_error_pct_cum"] = np.std(abs_error_pct_cum)
            else:  # Absolute Difference Error
                game_stats_cum["rmspe_cum"] = np.sqrt(np.mean(np.square(all_signed_error_pct_cum)))
                abs_error_abs_cum = [abs(e) for e in all_signed_error_pct_cum]
                game_stats_cum["mape_cum"] = np.mean(abs_error_abs_cum)
                game_stats_cum["min_error_pct_cum"] = min(abs_error_abs_cum)
                game_stats_cum["max_error_pct_cum"] = max(abs_error_abs_cum)
                game_stats_cum["mean_error_pct_cum"] = game_stats_cum["mape_cum"]
                game_stats_cum["sd_error_pct_cum"] = np.std(abs_error_abs_cum)
            rmse_values_cum = [game_stats_cum[f"rmse_m{m}_cum"] for m in range(1, 7) if pd.notnull(game_stats_cum[f"rmse_m{m}_cum"])]
            game_stats_cum["avg_rmse_cum"] = np.mean(rmse_values_cum) if rmse_values_cum else np.nan
        else:
            game_stats_cum["rmspe_cum"] = np.nan
            game_stats_cum["mape_cum"] = np.nan
            game_stats_cum["min_error_pct_cum"] = np.nan
            game_stats_cum["max_error_pct_cum"] = np.nan
            game_stats_cum["mean_error_pct_cum"] = np.nan
            game_stats_cum["sd_error_pct_cum"] = np.nan
            game_stats_cum["avg_rmse_cum"] = np.nan
        
        return summary_df, forecast_plot_data, game_stats_trans, game_stats_cum
    else:
        folder = {
        "Ridge Regression": "models",
        "XGBoost": "models_xgb",
        "Piecewise Linear Regression": "models_pwlr",
        "Interpolation Models": "models_pwli"
    }[model_type]
    if model_type == "XGBoost":
        model_file = os.path.join(folder, f"{game_id}_xgb_0.5.pkl")
        if not os.path.exists(model_file):
            return None, None, None, None
        with open(model_file, 'rb') as f:
            model = cloudpickle.load(f)
    else:
        suffix = "_best.pkl" if model_type == "Ridge Regression" else "_model.pkl"
        model_files = {
            'model1': os.path.join(folder, f"{game_id}_model1{suffix}"),
            'model2': os.path.join(folder, f"{game_id}_model2{suffix}"),
            'model3': os.path.join(folder, f"{game_id}_model3{suffix}")
        }
        models = {}
        for key, filepath in model_files.items():
            if not os.path.exists(filepath):
                return None, None, None, None
            with open(filepath, 'rb') as f:
                models[key] = cloudpickle.load(f)
        
    file_suffix = "_Dataset_12_2024.csv"
    blob_name = f"{DATA_PATH}{game_id}{file_suffix}"
    df_game = read_csv_from_gcs(BUCKET_NAME, blob_name)
    if df_game is None:
        return None, None, None, None
    df_game['transaction_period_date'] = pd.to_datetime(df_game['transaction_period'], errors='coerce', infer_datetime_format=True)
    df_game['cohort'] = pd.to_datetime(df_game['cohort'], format='%d/%m/%Y', errors='coerce')
    df_game = df_game.dropna(subset=['elapsed_months'])
    df_game['actual_date'] = df_game.apply(lambda row: row['cohort'] + pd.DateOffset(months=int(row['elapsed_months']) - 1), axis=1)
    
    all_cohorts = np.sort(df_game['cohort'].dropna().unique())
    if len(all_cohorts) < 8:
        return None, None, None, None
    valid_cohorts = [co for co in all_cohorts if df_game[df_game['cohort'] == co]['transaction_period_ltv_cac'].gt(0).any()]
    if len(valid_cohorts) < 6:
        return None, None, None, None
    predicted_cohorts = list(valid_cohorts[-6:])
    
    summary_rows = []
    forecast_plot_data = {}
    all_diff_trans = []
    all_diff_cum = []
    diff_trans_per_month = {i: [] for i in range(1,7)}
    diff_cum_per_month = {i: [] for i in range(1,7)}
    mc_features = [col for col in df_game.columns if col.startswith('MC_')]
    
    for cohort_date in predicted_cohorts:
        cohort_str = pd.Timestamp(cohort_date).strftime('%Y-%m-%d')
        cohort_df = df_game[df_game['cohort'] == cohort_date].copy().sort_values(by='actual_date')
        if 'client_actual_spend' in cohort_df.columns:
            cohort_df['base_input'] = cohort_df.apply(get_base_val, axis=1)
        else:
            cohort_df['base_input'] = cohort_df['total_cac']
        try:
            total_cac_val = cohort_df['base_input'].dropna().iloc[0]
        except IndexError:
            continue
        if total_cac_val <= 0:
            continue
        
        actual_df = cohort_df[['elapsed_months', 'actual_date', 'transaction_period_ltv_cac']].drop_duplicates().sort_values(by='actual_date')
        actual_df['actual_cumulative'] = actual_df['transaction_period_ltv_cac'].cumsum()
        unique_dates = sorted(actual_df['actual_date'].unique())
        forecast_length = min(len(unique_dates), 6)
        forecast_start_date = cohort_date
        cohort_month = pd.to_datetime(cohort_date).month
        
        if model_type == "XGBoost":
            forecast_df = forecast_xgb(model, total_cac_val, cohort_month, forecast_start_date, forecast_length, mc_features)
        else:
            forecast_df = forecast_cascaded_from_actual(total_cac_val, np.nan, np.nan, 0.0, forecast_start_date, forecast_length, models['model1'], models['model2'], models['model3'], model_type)
        
        row = {"game_id": game_id, "cohort": cohort_str, "spends": total_cac_val}
        cohort_diff_trans = []
        cohort_diff_cum = []
        
        for idx, f_row in enumerate(forecast_df.itertuples(), start=1):
            forecast_month = f_row.forecast_month
            pred_trans = f_row.predicted_transactional
            pred_cum = f_row.predicted_cumulative
            act_mask = actual_df['actual_date'].dt.strftime('%Y-%m') == forecast_month
            if act_mask.any():
                actual_trans = actual_df.loc[act_mask, 'transaction_period_ltv_cac'].iloc[0]
                actual_cum = actual_df.loc[act_mask, 'actual_cumulative'].iloc[0]
                diff_trans = pred_trans - actual_trans if pd.notnull(actual_trans) else np.nan
                diff_cum = pred_cum - actual_cum if pd.notnull(actual_cum) else np.nan
                if error_method == "Absolute Percentage Error":
                    if pd.notnull(diff_trans) and actual_trans != 0:
                        signed_error_pct_trans = (diff_trans / actual_trans) * 100
                        all_signed_error_pct_trans.append(signed_error_pct_trans)
                        abs_error_pct_trans = abs(signed_error_pct_trans)
                        row[f"m{idx}_error_trans"] = abs_error_pct_trans
                    else:
                        row[f"m{idx}_error_trans"] = np.nan
                else:  # Absolute Difference Error
                    if pd.notnull(diff_trans):
                        abs_error_abs_trans = abs(diff_trans)
                        all_signed_error_pct_trans.append(diff_trans)  # Store raw difference for stats
                        row[f"m{idx}_error_trans"] = abs_error_abs_trans
                    else:
                        row[f"m{idx}_error_trans"] = np.nan
                
                if error_method == "Absolute Percentage Error":
                    if pd.notnull(diff_cum) and actual_cum != 0:
                        signed_error_pct_cum = (diff_cum / actual_cum) * 100
                        all_signed_error_pct_cum.append(signed_error_pct_cum)
                        abs_error_pct_cum = abs(signed_error_pct_cum)
                        row[f"m{idx}_error_cum"] = abs_error_pct_cum
                    else:
                        row[f"m{idx}_error_cum"] = np.nan
                else:  # Absolute Difference Error
                    if pd.notnull(diff_cum):
                        abs_error_abs_cum = abs(diff_cum)
                        all_signed_error_pct_cum.append(diff_cum)  # Store raw difference for stats
                        row[f"m{idx}_error_cum"] = abs_error_abs_cum
                    else:
                        row[f"m{idx}_error_cum"] = np.nan
                if pd.notnull(diff_trans):
                    cohort_diff_trans.append(diff_trans)
                    all_diff_trans.append(diff_trans)
                    diff_trans_per_month[idx].append(diff_trans)
                if pd.notnull(diff_cum):
                    cohort_diff_cum.append(diff_cum)
                    all_diff_cum.append(diff_cum)
                    diff_cum_per_month[idx].append(diff_cum)
            else:
                row[f"m{idx}_error_trans"] = np.nan
                row[f"m{idx}_error_cum"] = np.nan
        row["cohort_rmse_trans"] = np.sqrt(np.mean(np.square(cohort_diff_trans))) * 100 if cohort_diff_trans else np.nan
        row["cohort_rmse_cum"] = np.sqrt(np.mean(np.square(cohort_diff_cum))) * 100 if cohort_diff_cum else np.nan
        summary_rows.append(row)
        forecast_plot_data[cohort_str] = {'forecast': forecast_df, 'actual': actual_df}
    
    summary_df = pd.DataFrame(summary_rows)
    
    game_stats_trans = {}
    for m in range(1, 7):
        if diff_trans_per_month[m]:
            game_stats_trans[f"rmse_m{m}_trans"] = np.sqrt(np.mean(np.square(diff_trans_per_month[m]))) * 100
        else:
            game_stats_trans[f"rmse_m{m}_trans"] = np.nan
    if all_signed_error_pct_trans:
        game_stats_trans["rmspe_trans"] = np.sqrt(np.mean(np.square(all_signed_error_pct_trans)))
        abs_error_pct_trans = [abs(e) for e in all_signed_error_pct_trans]
        game_stats_trans["mape_trans"] = np.mean(abs_error_pct_trans)
        game_stats_trans["min_error_pct_trans"] = min(abs_error_pct_trans)
        game_stats_trans["max_error_pct_trans"] = max(abs_error_pct_trans)
        game_stats_trans["mean_error_pct_trans"] = game_stats_trans["mape_trans"]
        game_stats_trans["sd_error_pct_trans"] = np.std(abs_error_pct_trans)
        rmse_values_trans = [game_stats_trans[f"rmse_m{m}_trans"] for m in range(1, 7) if pd.notnull(game_stats_trans[f"rmse_m{m}_trans"])]
        game_stats_trans["avg_rmse_trans"] = np.mean(rmse_values_trans) if rmse_values_trans else np.nan
    else:
        game_stats_trans["rmspe_trans"] = np.nan
        game_stats_trans["mape_trans"] = np.nan
        game_stats_trans["min_error_pct_trans"] = np.nan
        game_stats_trans["max_error_pct_trans"] = np.nan
        game_stats_trans["mean_error_pct_trans"] = np.nan
        game_stats_trans["sd_error_pct_trans"] = np.nan
        game_stats_trans["avg_rmse_trans"] = np.nan
    
    game_stats_cum = {}
    for m in range(1, 7):
        if diff_cum_per_month[m]:
            game_stats_cum[f"rmse_m{m}_cum"] = np.sqrt(np.mean(np.square(diff_cum_per_month[m]))) * 100
        else:
            game_stats_cum[f"rmse_m{m}_cum"] = np.nan
    if all_signed_error_pct_cum:
        game_stats_cum["rmspe_cum"] = np.sqrt(np.mean(np.square(all_signed_error_pct_cum)))
        abs_error_pct_cum = [abs(e) for e in all_signed_error_pct_cum]
        game_stats_cum["mape_cum"] = np.mean(abs_error_pct_cum)
        game_stats_cum["min_error_pct_cum"] = min(abs_error_pct_cum)
        game_stats_cum["max_error_pct_cum"] = max(abs_error_pct_cum)
        game_stats_cum["mean_error_pct_cum"] = game_stats_cum["mape_cum"]
        game_stats_cum["sd_error_pct_cum"] = np.std(abs_error_pct_cum)
        rmse_values_cum = [game_stats_cum[f"rmse_m{m}_cum"] for m in range(1, 7) if pd.notnull(game_stats_cum[f"rmse_m{m}_cum"])]
        game_stats_cum["avg_rmse_cum"] = np.mean(rmse_values_cum) if rmse_values_cum else np.nan
    else:
        game_stats_cum["rmspe_cum"] = np.nan
        game_stats_cum["mape_cum"] = np.nan
        game_stats_cum["min_error_pct_cum"] = np.nan
        game_stats_cum["max_error_pct_cum"] = np.nan
        game_stats_cum["mean_error_pct_cum"] = np.nan
        game_stats_cum["sd_error_pct_cum"] = np.nan
        game_stats_cum["avg_rmse_cum"] = np.nan
    
    return summary_df, forecast_plot_data, game_stats_trans, game_stats_cum

if view_option == "Overview":
        all_overview_data = []
        for model_type in ["Extrapolation"]:
            all_results = {}
            game_stats_all = {}
            for game in filtered_game_list:
                summary_df, _, game_stats_trans, game_stats_cum = run_cohort_forecasts(game, model_type, error_method)
                if summary_df is not None and not summary_df.empty:
                    all_results[game] = summary_df
                    game_stats_all[game] = {'trans': game_stats_trans, 'cum': game_stats_cum}
            if all_results:
                # Transactional Errors Overview
                overview_rows_trans = []
                for game, df in all_results.items():
                    error_cols_trans = [f"m{i}_error_trans" for i in range(1, 7)]
                    avg_errors_trans = {col: df[col].mean() for col in error_cols_trans}
                    row = {
                        "game_id": game,
                        "rmspe_trans": game_stats_all[game]['trans']['rmspe_trans'],
                        "avg_rmse_trans": game_stats_all[game]['trans']['avg_rmse_trans'],
                        "mape_trans": game_stats_all[game]['trans']['mape_trans'],
                        "min_error_pct_trans": game_stats_all[game]['trans']['min_error_pct_trans'],
                        "max_error_pct_trans": game_stats_all[game]['trans']['max_error_pct_trans'],
                        "mean_error_pct_trans": game_stats_all[game]['trans']['mean_error_pct_trans'],
                        "sd_error_pct_trans": game_stats_all[game]['trans']['sd_error_pct_trans']
                    }
                    row.update(avg_errors_trans)
                    overview_rows_trans.append(row)
                overview_df_trans = pd.DataFrame(overview_rows_trans)
                overall_row_trans = {'game_id': 'ALL_GAMES'}
                for col in overview_df_trans.columns:
                    if col != 'game_id':
                        overall_row_trans[col] = overview_df_trans[col].mean()
                overview_df_trans = pd.concat([pd.DataFrame([overall_row_trans]), overview_df_trans], ignore_index=True)
                
                # Split into Monthly Errors and Statistics Tables
                monthly_errors_trans = overview_df_trans[['game_id'] + [f'm{i}_error_trans' for i in range(1,7)]]
                if error_method == "Absolute Percentage Error":
                    st.subheader("Average Monthly Transactional Errors (%)")
                    st.dataframe(monthly_errors_trans.style.format({f'm{i}_error_trans': '{:.2f}%' for i in range(1,7)}))
                else:
                    st.subheader("Average Monthly Transactional Errors (Absolute)")
                    st.dataframe(monthly_errors_trans.style.format({f'm{i}_error_trans': '{:.4f}' for i in range(1,7)}))

                stats_trans = overview_df_trans[['game_id', 'rmspe_trans', 'avg_rmse_trans', 'mape_trans', 'min_error_pct_trans', 'max_error_pct_trans', 'mean_error_pct_trans', 'sd_error_pct_trans']]
                if error_method == "Absolute Percentage Error":
                    st.subheader("Transactional Error Statistics (%)")
                    st.dataframe(stats_trans.style.format({col: '{:.2f}%' for col in stats_trans.columns if col != 'game_id'}))
                else:
                    st.subheader("Transactional Error Statistics (Absolute)")
                    st.dataframe(stats_trans.style.format({col: '{:.4f}' for col in stats_trans.columns if col != 'game_id'}))
                
                # Transactional Heatmap
                if not overview_df_trans.empty:
                    heatmap_data_trans = overview_df_trans[overview_df_trans['game_id'] != 'ALL_GAMES'].set_index('game_id')[[f"m{i}_error_trans" for i in range(1, 7)]]
                    x_labels = [f"m{i}_error_trans" for i in range(1, 7)]
                    
                    fig_heatmap_trans = go.Figure(data=go.Heatmap(
                        z=heatmap_data_trans.values,
                        x=x_labels,
                        y=heatmap_data_trans.index,
                        text=heatmap_data_trans.values,
                        texttemplate="%{text:.2f}%",
                        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                        colorbar=dict(title="Error (%)")
                    ))
                    fig_heatmap_trans.update_layout(
                        title=f"Heatmap of Transactional Forecast Errors (%) - {model_type}",
                        xaxis_title="Forecast Month",
                        yaxis_title="Game",
                        height=600
                    )
                    st.plotly_chart(fig_heatmap_trans, use_container_width=True)
                
                # Cumulative Errors Overview
                overview_rows_cum = []
                for game, df in all_results.items():
                    error_cols_cum = [f"m{i}_error_cum" for i in range(1, 7)]
                    avg_errors_cum = {col: df[col].mean() for col in error_cols_cum}
                    row = {
                        "game_id": game,
                        "rmspe_cum": game_stats_all[game]['cum']['rmspe_cum'],
                        "avg_rmse_cum": game_stats_all[game]['cum']['avg_rmse_cum'],
                        "mape_cum": game_stats_all[game]['cum']['mape_cum'],
                        "min_error_pct_cum": game_stats_all[game]['cum']['min_error_pct_cum'],
                        "max_error_pct_cum": game_stats_all[game]['cum']['max_error_pct_cum'],
                        "mean_error_pct_cum": game_stats_all[game]['cum']['mean_error_pct_cum'],
                        "sd_error_pct_cum": game_stats_all[game]['cum']['sd_error_pct_cum']
                    }
                    row.update(avg_errors_cum)
                    overview_rows_cum.append(row)
                overview_df_cum = pd.DataFrame(overview_rows_cum)
                overall_row_cum = {'game_id': 'ALL_GAMES'}
                for col in overview_df_cum.columns:
                    if col != 'game_id':
                        overall_row_cum[col] = overview_df_cum[col].mean()
                overview_df_cum = pd.concat([pd.DataFrame([overall_row_cum]), overview_df_cum], ignore_index=True)
                
                # Split into Monthly Errors and Statistics Tables
                monthly_errors_cum = overview_df_cum[['game_id'] + [f'm{i}_error_cum' for i in range(1,7)]]
                if error_method == "Absolute Percentage Error":
                    st.subheader("Average Monthly Cumulative Errors (%)")
                    st.dataframe(monthly_errors_cum.style.format({f'm{i}_error_cum': '{:.2f}%' for i in range(1,7)}))
                else:
                    st.subheader("Average Monthly Cumulative Errors (Absolute)")
                    st.dataframe(monthly_errors_cum.style.format({f'm{i}_error_cum': '{:.4f}' for i in range(1,7)}))

                stats_cum = overview_df_cum[['game_id', 'rmspe_cum', 'avg_rmse_cum', 'mape_cum', 'min_error_pct_cum', 'max_error_pct_cum', 'mean_error_pct_cum', 'sd_error_pct_cum']]
                if error_method == "Absolute Percentage Error":
                    st.subheader("Cumulative Error Statistics (%)")
                    st.dataframe(stats_cum.style.format({col: '{:.2f}%' for col in stats_cum.columns if col != 'game_id'}))
                else:
                    st.subheader("Cumulative Error Statistics (Absolute)")
                    st.dataframe(stats_cum.style.format({col: '{:.4f}' for col in stats_cum.columns if col != 'game_id'}))
                
                # Cumulative Heatmap
                if not overview_df_cum.empty:
                    heatmap_data_cum = overview_df_cum[overview_df_cum['game_id'] != 'ALL_GAMES'].set_index('game_id')[[f"m{i}_error_cum" for i in range(1, 7)]]
                    x_labels_cum = [f"m{i}_error_cum" for i in range(1, 7)]
                    
                    fig_heatmap_cum = go.Figure(data=go.Heatmap(
                        z=heatmap_data_cum.values,
                        x=x_labels_cum,
                        y=heatmap_data_cum.index,
                        text=heatmap_data_cum.values,
                        texttemplate="%{text:.2f}%",
                        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                        colorbar=dict(title="Error (%)")
                    ))
                    fig_heatmap_cum.update_layout(
                        title=f"Heatmap of Cumulative Forecast Errors (%) - {model_type}",
                        xaxis_title="Forecast Month",
                        yaxis_title="Game",
                        height=600
                    )
                    st.plotly_chart(fig_heatmap_cum, use_container_width=True)
                
                # Collect data for overall comparison
                for game in all_results:
                    all_overview_data.append({
                        "game_id": game,
                        "model_type": model_type,
                        "rmspe_trans": game_stats_all[game]['trans']['rmspe_trans'],
                        "rmspe_cum": game_stats_all[game]['cum']['rmspe_cum']
                    })
        
        if all_overview_data:
            all_overview_df = pd.DataFrame(all_overview_data)
            
            # Best models based on transactional RMSPE
            # best_models_trans = all_overview_df.loc[all_overview_df.groupby('game_id')['rmspe_trans'].idxmin()]
            # best_models_trans = best_models_trans.sort_values('game_id')
            # st.header("Best Model Performance for Each Game (Based on Transactional RMSPE)")
            # st.dataframe(best_models_trans[['game_id', 'model_type', 'rmspe_trans', 'rmspe_cum']].style.format({
            #     'rmspe_trans': "{:.2f}%",
            #     'rmspe_cum': "{:.2f}%"
            # }))
            
            # Best models based on cumulative RMSPE
            # best_models_cum = all_overview_df.loc[all_overview_df.groupby('game_id')['rmspe_cum'].idxmin()]
            # best_models_cum = best_models_cum.sort_values('game_id')
            # st.header("Best Model Performance for Each Game (Based on Cumulative RMSPE)")
            # st.dataframe(best_models_cum[['game_id', 'model_type', 'rmspe_trans', 'rmspe_cum']].style.format({
            #     'rmspe_trans': "{:.2f}%",
            #     'rmspe_cum': "{:.2f}%"
            # }))
else:
        game_id = view_option
        summary_df, forecast_plot_data, game_stats_trans, game_stats_cum = run_cohort_forecasts(game_id, model_type, error_method)
        if summary_df is not None and not summary_df.empty:
            # Transactional Errors Table with Average Row
            if error_method == "Absolute Percentage Error":
                st.header(f"Transactional Prediction Errors for Game: {game_id} - {model_type} (%)")
                format_dict_trans = {f"m{i}_error_trans": "{:.2f}%" for i in range(1, 7)}
                format_dict_trans["cohort_rmse_trans"] = "{:.2f}%"
            else:
                st.header(f"Transactional Prediction Errors for Game: {game_id} - {model_type} (Absolute)")
                format_dict_trans = {f"m{i}_error_trans": "{:.4f}" for i in range(1, 7)}
                format_dict_trans["cohort_rmse_trans"] = "{:.4f}"
            
            trans_cols = ['cohort', 'spends'] + [f"m{i}_error_trans" for i in range(1, 7)] + ['cohort_rmse_trans']
            avg_row_trans = {
                'cohort': 'Average',
                'spends': np.nan,
                **{f'm{i}_error_trans': summary_df[f'm{i}_error_trans'].mean() for i in range(1, 7)},
                'cohort_rmse_trans': summary_df['cohort_rmse_trans'].mean()
            }
            summary_df_trans_display = pd.concat([summary_df, pd.DataFrame([avg_row_trans])], ignore_index=True)
            st.dataframe(summary_df_trans_display[trans_cols].style.format(format_dict_trans))
            if error_method == "Absolute Percentage Error":
                st.subheader("Transactional Error Statistics (%)")
                stats_format = "{:.2f}%"
            else:
                st.subheader("Transactional Error Statistics (Absolute)")
                stats_format = "{:.4f}"
                
            stats_trans_df = pd.DataFrame([{
                "RMSPE": game_stats_trans['rmspe_trans'],
                "Avg RMSE": game_stats_trans['avg_rmse_trans'],
                "MAPE": game_stats_trans['mape_trans'],
                "Min Error": game_stats_trans['min_error_pct_trans'],
                "Max Error": game_stats_trans['max_error_pct_trans'],
                "Mean Error": game_stats_trans['mean_error_pct_trans'],
                "SD Error": game_stats_trans['sd_error_pct_trans']
            }])
            st.dataframe(stats_trans_df.style.format(stats_format))
            
            # Cumulative Errors Table with Average Row
            if error_method == "Absolute Percentage Error":
                st.header(f"Cumulative Prediction Errors for Game: {game_id} - {model_type} (%)")
                format_dict_cum = {f"m{i}_error_cum": "{:.2f}%" for i in range(1, 7)}
                format_dict_cum["cohort_rmse_cum"] = "{:.2f}%"
            else:
                st.header(f"Cumulative Prediction Errors for Game: {game_id} - {model_type} (Absolute)")
                format_dict_cum = {f"m{i}_error_cum": "{:.4f}" for i in range(1, 7)}
                format_dict_cum["cohort_rmse_cum"] = "{:.4f}"
                
            cum_cols = ['cohort', 'spends'] + [f"m{i}_error_cum" for i in range(1, 7)] + ['cohort_rmse_cum']
            avg_row_cum = {
                'cohort': 'Average',
                'spends': np.nan,
                **{f'm{i}_error_cum': summary_df[f'm{i}_error_cum'].mean() for i in range(1, 7)},
                'cohort_rmse_cum': summary_df['cohort_rmse_cum'].mean()
            }
            summary_df_cum_display = pd.concat([summary_df, pd.DataFrame([avg_row_cum])], ignore_index=True)
            st.dataframe(summary_df_cum_display[cum_cols].style.format(format_dict_cum))
            if error_method == "Absolute Percentage Error":
                st.subheader("Cumulative Error Statistics (%)")
                stats_format_cum = "{:.2f}%"
            else:
                st.subheader("Cumulative Error Statistics (Absolute)")
                stats_format_cum = "{:.4f}"
                
            stats_cum_df = pd.DataFrame([{
                "RMSPE": game_stats_cum['rmspe_cum'],
                "Avg RMSE": game_stats_cum['avg_rmse_cum'],
                "MAPE": game_stats_cum['mape_cum'],
                "Min Error": game_stats_cum['min_error_pct_cum'],
                "Max Error": game_stats_cum['max_error_pct_cum'],
                "Mean Error": game_stats_cum['mean_error_pct_cum'],
                "SD Error": game_stats_cum['sd_error_pct_cum']
            }])
            st.dataframe(stats_cum_df.style.format(stats_format_cum))
            
            # Prediction Plots
            st.subheader("Prediction Curves per Cohort")
            for cohort_str, data_dict in forecast_plot_data.items():
                st.write(f"#### Cohort: {cohort_str}")
                forecast_df = data_dict['forecast']
                actual_df = data_dict['actual']
                forecast_dates = pd.to_datetime(forecast_df['forecast_month'] + "-01")
                actual_sorted = actual_df.sort_values(by='actual_date').copy()
                if not actual_sorted.empty:
                    actual_sorted['actual_cumulative'] = actual_sorted['transaction_period_ltv_cac'].cumsum()
                
                fig_cum = go.Figure()
                if not actual_sorted.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=actual_sorted['actual_date'],
                        y=actual_sorted['actual_cumulative'],
                        mode='lines+markers',
                        name="Actual Cumulative"
                    ))
                fig_cum.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_df['predicted_cumulative'],
                    mode='lines+markers',
                    name="Predicted Cumulative"
                ))
                fig_cum.update_layout(
                    title=f"Cumulative Prediction - Cohort {cohort_str} - {model_type}",
                    xaxis_title="Calendar Month",
                    yaxis_title="Cumulative Value"
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
                fig_trans = go.Figure()
                if not actual_df.empty:
                    actual_monthly = actual_df.copy()
                    actual_monthly['month_str'] = actual_monthly['actual_date'].dt.strftime('%Y-%m')
                    actual_monthly = actual_monthly.drop_duplicates(subset=['month_str'])
                    fig_trans.add_trace(go.Scatter(
                        x=pd.to_datetime(actual_monthly['month_str'] + "-01"),
                        y=actual_monthly['transaction_period_ltv_cac'],
                        mode='lines+markers',
                        name="Actual Transactional"
                    ))
                fig_trans.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_df['predicted_transactional'],
                    mode='lines+markers',
                    name="Predicted Transactional"
                ))
                fig_trans.update_layout(
                    title=f"Transactional Prediction - Cohort {cohort_str} - {model_type}",
                    xaxis_title="Calendar Month",
                    yaxis_title="Transactional Value"
                )
                st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.write(f"No prediction data available for game {game_id}.")