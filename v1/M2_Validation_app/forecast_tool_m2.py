import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import cloudpickle
import json
from datetime import datetime
import plotly.graph_objects as go

EPSILON = 1e-6

st.title('M2 6 Months Forecasting Tool')

# Global variables for M2 configuration
m2_config = None

def load_m2_config():
    """Load M2 configuration from M2/Latest Data/Segregated/config.json"""
    global m2_config
    try:
        with open('M2/Latest Data/Segregated/config.json', 'r') as f:
            m2_config = json.load(f)
        return True
    except Exception as e:
        st.error(f"Error loading M2 config: {e}")
        return False

def get_m2_model_info(game_id, model_type=None):
    """Get model features for M2 game, optionally for a specific model type"""
    if m2_config is None:
        if not load_m2_config():
            return None, None
    
    m2_key = f"M2_{game_id}"
    if m2_key in m2_config:
        config = m2_config[m2_key]
        if model_type:
            # Use specified model type
            return model_type, config.get('selected_features', [])
        else:
            # Use best model type
            return config.get('best_model'), config.get('selected_features', [])
    return None, None

def load_m2_model(game_id, model_type=None):
    """Load M2 model for a game, optionally for a specific model type"""
    try:
        if model_type:
            # Load specific model type
            features = None
            if m2_config:
                m2_key = f"M2_{game_id}"
                if m2_key in m2_config:
                    features = m2_config[m2_key].get('selected_features', [])
            
            model_path = f"M2/new_models/{game_id}_{model_type}_M2.pkl"
            if not os.path.exists(model_path):
                return None, None
            
            # Try fallback loading first
            try:
                from model_fix_utils import load_model_with_fallback
                model, success, error_msg = load_model_with_fallback(model_path)
                if success:
                    if error_msg:  # There was a fallback used
                        st.info(f"⚠️ Model loaded with fallback method for {game_id}: {error_msg}")
                    return model, features
                else:
                    raise Exception(error_msg)
            except ImportError:
                # Fallback utils not available, use normal loading
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
                return model, features
                
        else:
            # Load best model
            best_model_type, features = get_m2_model_info(game_id)
            if best_model_type is None:
                return None, None
            
            model_path = f"M2/new_models/{game_id}_{best_model_type}_M2.pkl"
            if not os.path.exists(model_path):
                return None, None
            
            # Try fallback loading first
            try:
                from model_fix_utils import load_model_with_fallback
                model, success, error_msg = load_model_with_fallback(model_path)
                if success:
                    if error_msg:  # There was a fallback used
                        st.info(f"⚠️ Model loaded with fallback method for {game_id}: {error_msg}")
                    return model, features
                else:
                    raise Exception(error_msg)
            except ImportError:
                # Fallback utils not available, use normal loading
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
                return model, features
            
    except Exception as e:
        error_msg = str(e)
        if "_RemainderColsList" in error_msg or "sklearn" in error_msg.lower():
            st.error(f"""
            🚨 **sklearn Version Compatibility Issue** for {game_id}
            
            **Problem**: The M2 models were trained with a different version of scikit-learn 
            than what's installed in your deployment environment.
            
            **Error**: {error_msg}
            
            **Solutions**:
            1. Run `python detect_env_versions.py` in your development environment
            2. Use the generated `requirements_exact.txt` 
            3. Update Dockerfile to use Python 3.12-slim (matches your development Python 3.12.5)
            4. Rebuild Docker image: `docker build -t your-app .`
            
            **Current Environment**: Python {os.sys.version.split()[0]} in Docker vs Python 3.12.5 in development
            """)
        else:
            st.error(f"Error loading M2 model for {game_id}: {error_msg}")
        return None, None

def get_available_m2_games():
    """Get list of games that have M2 models available"""
    if not load_m2_config():
        return []
    
    m2_games = []
    for key in m2_config.keys():
        if key.startswith('M2_'):
            game_id = key.replace('M2_', '')
            # Check if at least one model file exists
            model_types = ['xgboost', 'catboost', 'lightgbm', 'random_forest']
            for model_type in model_types:
                model_path = f"M2/new_models/{game_id}_{model_type}_M2.pkl"
                if os.path.exists(model_path):
                    m2_games.append(game_id)
                    break
    
    return sorted(m2_games)

def get_available_m2_model_types(game_id):
    """Get list of available M2 model types for a specific game"""
    model_types = ['xgboost', 'catboost', 'lightgbm', 'random_forest']
    available_types = []
    
    for model_type in model_types:
        model_path = f"M2/new_models/{game_id}_{model_type}_M2.pkl"
        if os.path.exists(model_path):
            available_types.append(model_type)
    
    return available_types

def qualifies_m2_game(game_id):
    """Check if game has sufficient data for M2 forecasting"""
    file_suffix = "_Dataset_12_2024.csv"
    file_path = os.path.join("Latest Data", "Segregated", f"{game_id}{file_suffix}")
    if not os.path.exists(file_path):
        return False
    try:
        df = pd.read_csv(file_path, usecols=['cohort'])
        df['cohort'] = pd.to_datetime(df['cohort'], format='%d/%m/%Y', errors='coerce')
        distinct = df['cohort'].dropna().nunique()
        return distinct >= 8
    except Exception:
        return False

def calculate_current_month_m2(cohort_date, elapsed_months):
    """Calculate current month for M2 predictions"""
    cohort_start = pd.to_datetime(cohort_date)
    current_date = cohort_start + pd.DateOffset(months=int(elapsed_months))
    return current_date.month

def prepare_m2_features(cohort_data, elapsed_months, lag_1, lag_2, lag_3, features_list):
    """Prepare features for M2 model prediction"""
    # Get static features from the cohort data (use latest available data)
    max_elapsed = cohort_data['elapsed_months'].max()
    static_row = cohort_data[cohort_data['elapsed_months'] == max_elapsed].iloc[0]
    
    # Start with static features
    features = {}
    for feature in features_list:
        if feature in static_row.index:
            features[feature] = static_row[feature]
        else:
            features[feature] = 0.0  # Default value for missing features
    
    # Set dynamic features
    features['elapsed_months'] = elapsed_months
    
    # Calculate current month
    cohort_date = static_row['cohort']
    features['current_month'] = calculate_current_month_m2(cohort_date, elapsed_months)
    
    # Set lag features
    features['ltv_lag_1'] = lag_1 if pd.notna(lag_1) else 0.0
    features['ltv_lag_2'] = lag_2 if pd.notna(lag_2) else 0.0
    features['ltv_lag_3'] = lag_3 if pd.notna(lag_3) else 0.0
    
    # Handle m1_feature_* - these should come from month 1 data
    month1_data = cohort_data[cohort_data['elapsed_months'] == 1]
    if not month1_data.empty:
        month1_row = month1_data.iloc[0]
        for feature in features_list:
            if feature.startswith('m1_feature_'):
                # Extract the base feature name and get from month 1 data
                base_feature = feature.replace('m1_feature_', '')
                if base_feature in month1_row.index:
                    features[feature] = month1_row[base_feature]
    
    # Ensure all MC_ features are handled
    for feature in features_list:
        if feature.startswith('MC_') and pd.isna(features.get(feature, np.nan)):
            features[feature] = 0.0
    
    # Ensure day_ features are properly handled
    for feature in features_list:
        if feature.startswith('day_') and pd.isna(features.get(feature, np.nan)):
            features[feature] = 0.0
            
    return features

def forecast_m2_single_game(game_id, total_cac, forecast_start_date, months, model, features_list, cohort_data):
    """Forecast using M2 model for a single game"""
    results = []
    
    # Initialize lag variables - M2 starts from month 2, so we need month 1 data for lag_1
    lag_1 = np.nan
    lag_2 = np.nan  
    lag_3 = np.nan
    
    # For M2, we start predictions from month 2 (elapsed_months = 1 in M2's 0-indexed system)
    # But we need to get the actual LTV value for month 1 to use as lag_1
    if not cohort_data.empty:
        month1_data = cohort_data[cohort_data['elapsed_months'] == 1]  # Month 1 in original indexing
        if not month1_data.empty:
            lag_1 = month1_data['transaction_period_ltv_cac'].iloc[0]
    
    cumulative = 0.0
    # Add month 1 cumulative if available
    if pd.notna(lag_1):
        cumulative = lag_1
    
    # M2 forecasts months 2-6 (or however many months requested)
    for m in range(2, months + 2):  # Start from month 2
        current_date = forecast_start_date + pd.DateOffset(months=m-1)
        
        # Prepare features for prediction
        elapsed_months_m2 = m - 1  # M2 uses 0-indexed elapsed_months, so month 2 = elapsed_months 1
        features = prepare_m2_features(cohort_data, elapsed_months_m2, lag_1, lag_2, lag_3, features_list)
        
        # Make prediction
        try:
            # Ensure features are in correct order for model
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                feature_values = [features.get(fname, 0.0) for fname in feature_names]
                X_new = pd.DataFrame([feature_values], columns=feature_names)
            else:
                X_new = pd.DataFrame([features])
            
            pred = model.predict(X_new)[0]
            pred = max(pred, EPSILON)  # Ensure non-negative
        except Exception as e:
            st.error(f"M2 prediction error for {game_id} month {m}: {e}")
            pred = EPSILON
        
        cumulative += pred
        results.append({
            'forecast_month': current_date.strftime('%Y-%m'),
            'predicted_transactional': pred,
            'predicted_cumulative': cumulative
        })
        
        # Update lag variables for next iteration
        lag_3 = lag_2
        lag_2 = lag_1
        lag_1 = pred
    
    return pd.DataFrame(results)

def get_base_val(row):
    if pd.isnull(row['total_cac']) or row['total_cac'] == 0:
        return row['client_actual_spend'] if ('client_actual_spend' in row and pd.notnull(row['client_actual_spend'])) else np.nan
    else:
        return row['total_cac']

def run_m2_cohort_forecasts(game_id, model_type=None, error_method="Absolute Percentage Error"):
    """Run M2 cohort forecasts for a specific game and model type"""
    all_signed_error_pct_trans = []
    all_signed_error_pct_cum = []
    
    # Load M2 model
    model, features_list = load_m2_model(game_id, model_type)
    if model is None:
        if model_type:
            st.error(f"No M2 {model_type} model found for game {game_id}")
        else:
            st.error(f"No M2 model found for game {game_id}")
        return None, None, None, None
        
    file_suffix = "_Dataset_12_2024.csv"
    file_path = os.path.join("Latest Data", "Segregated", f"{game_id}{file_suffix}")
    if not os.path.exists(file_path):
        return None, None, None, None
    
    df_game = pd.read_csv(file_path)
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
    # Shift cohort selection 1 month earlier to ensure we have actual data for all 6 months (m2-m7)
    predicted_cohorts = list(valid_cohorts[-7:-1])  # Use cohorts from -7 to -2 instead of -6 to -1
    
    summary_rows = []
    forecast_plot_data = {}
    all_diff_trans = []
    all_diff_cum = []
    # M2 starts from month 2, so we track errors for months 2-7
    diff_trans_per_month = {i: [] for i in range(2,8)}
    diff_cum_per_month = {i: [] for i in range(2,8)}
    
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
        
        forecast_start_date = cohort_date
        # M2 forecasts 6 months (months 2-7)
        forecast_df = forecast_m2_single_game(game_id, total_cac_val, forecast_start_date, 6, model, features_list, cohort_df)
        
        row = {"game_id": game_id, "cohort": cohort_str, "spends": total_cac_val}
        cohort_diff_trans = []
        cohort_diff_cum = []
        
        # For M2, we start error tracking from month 2 (index 0 in forecast_df corresponds to month 2)
        for idx, f_row in enumerate(forecast_df.itertuples(), start=2):  # Start from month 2
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
    
    # Calculate game statistics for M2 (months 2-7)
    game_stats_trans = {}
    for m in range(2, 8):
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
        rmse_values_trans = [game_stats_trans[f"rmse_m{m}_trans"] for m in range(2, 8) if pd.notnull(game_stats_trans[f"rmse_m{m}_trans"])]
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
    for m in range(2, 8):
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
        rmse_values_cum = [game_stats_cum[f"rmse_m{m}_cum"] for m in range(2, 8) if pd.notnull(game_stats_cum[f"rmse_m{m}_cum"])]
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

# Get available M2 games
available_m2_games = get_available_m2_games()
filtered_m2_games = [g for g in available_m2_games if qualifies_m2_game(g)]

if not filtered_m2_games:
    st.error("No M2 games found with sufficient data. Please check your M2 models and data.")
    st.stop()

# Sidebar configuration
st.sidebar.header("📊 Configuration")
st.sidebar.subheader("Error Calculation Theme")
error_method = st.sidebar.radio(
    "Choose Error Calculation Method:",
    ["Absolute Difference Error", "Absolute Percentage Error"],
    index=1
)

# # Show active configuration
# if error_method == "Absolute Difference Error":
#     st.sidebar.success("🎯 Active: Absolute Difference Error") 
#     st.sidebar.info("� Using: absolute error columns")
# else:
#     st.sidebar.success("🎯 Active: Absolute Percentage Error")
#     st.sidebar.info("� Using: percentage error columns")

st.sidebar.subheader("Select Analysis Type:")
view_option = st.sidebar.selectbox("", ["Overview"] + filtered_m2_games)

# Model type selection - only for individual game view
model_type_mapping = {
    "XGBoost": "xgboost",
    "CatBoost": "catboost", 
    "LightGBM": "lightgbm",
    "Random Forest": "random_forest"
}

if view_option != "Overview":
    # Check available model types for the selected game
    available_model_types = get_available_m2_model_types(view_option)
    available_display_names = [name for name, code in model_type_mapping.items() if code in available_model_types]
    
    if available_display_names:
        selected_model_display = st.sidebar.selectbox("Select M2 Model Type", available_display_names)
        selected_model_type = model_type_mapping[selected_model_display]
    else:
        st.error(f"No M2 models found for game {view_option}")
        st.stop()

if st.sidebar.button("Run M2 Cohort Forecasts"):
    if view_option == "Overview":
        st.header("M2 Models Overview - All Games")
        all_overview_data = []
        
        for model_display, model_code in model_type_mapping.items():
            st.header(f"Overview for M2 {model_display}")
            all_results = {}
            game_stats_all = {}
            
            for game in filtered_m2_games:
                # Check if this game has the selected model type
                available_model_types = get_available_m2_model_types(game)
                if model_code in available_model_types:
                    summary_df, _, game_stats_trans, game_stats_cum = run_m2_cohort_forecasts(game, model_code, error_method)
                    if summary_df is not None and not summary_df.empty:
                        all_results[game] = summary_df
                        game_stats_all[game] = {'trans': game_stats_trans, 'cum': game_stats_cum}
            
            if all_results:
                # Transactional Errors Overview
                overview_rows_trans = []
                for game, df in all_results.items():
                    error_cols_trans = [f"m{i}_error_trans" for i in range(2, 8)]  # M2: months 2-7
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
                monthly_errors_trans = overview_df_trans[['game_id'] + [f'm{i}_error_trans' for i in range(2,8)]]
                if error_method == "Absolute Percentage Error":
                    st.subheader(f"Average Monthly Transactional Errors (%) - M2 {model_display}")
                    st.dataframe(monthly_errors_trans.style.format({f'm{i}_error_trans': '{:.2f}%' for i in range(2,8)}))
                else:
                    st.subheader(f"Average Monthly Transactional Errors - M2 {model_display}")
                    st.dataframe(monthly_errors_trans.style.format({f'm{i}_error_trans': '{:.4f}' for i in range(2,8)}))

                stats_trans = overview_df_trans[['game_id', 'rmspe_trans', 'avg_rmse_trans', 'mape_trans', 'min_error_pct_trans', 'max_error_pct_trans', 'mean_error_pct_trans', 'sd_error_pct_trans']]
                if error_method == "Absolute Percentage Error":
                    st.subheader(f"Transactional Error Statistics (%) - M2 {model_display}")
                    st.dataframe(stats_trans.style.format({col: '{:.2f}%' for col in stats_trans.columns if col != 'game_id'}))
                else:
                    st.subheader(f"Transactional Error Statistics - M2 {model_display}")
                    st.dataframe(stats_trans.style.format({col: '{:.4f}' for col in stats_trans.columns if col != 'game_id'}))
                
                # Transactional Heatmap
                if not overview_df_trans.empty:
                    heatmap_data_trans = overview_df_trans[overview_df_trans['game_id'] != 'ALL_GAMES'].set_index('game_id')[[f"m{i}_error_trans" for i in range(2, 8)]]
                    x_labels = [f"m{i}_error_trans" for i in range(2, 8)]
                    
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
                        title=f"Heatmap of M2 {model_display} Transactional Forecast Errors (%) - Months 2-7",
                        xaxis_title="Forecast Month",
                        yaxis_title="Game",
                        height=600
                    )
                    st.plotly_chart(fig_heatmap_trans, use_container_width=True)
                
                # Cumulative Errors Overview
                overview_rows_cum = []
                for game, df in all_results.items():
                    error_cols_cum = [f"m{i}_error_cum" for i in range(2, 8)]  # M2: months 2-7
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
                monthly_errors_cum = overview_df_cum[['game_id'] + [f'm{i}_error_cum' for i in range(2,8)]]
                if error_method == "Absolute Percentage Error":
                    st.subheader(f"Average Monthly Cumulative Errors (%) - M2 {model_display}")
                    st.dataframe(monthly_errors_cum.style.format({f'm{i}_error_cum': '{:.2f}%' for i in range(2,8)}))
                else:
                    st.subheader(f"Average Monthly Cumulative Errors - M2 {model_display}")
                    st.dataframe(monthly_errors_cum.style.format({f'm{i}_error_cum': '{:.4f}' for i in range(2,8)}))

                stats_cum = overview_df_cum[['game_id', 'rmspe_cum', 'avg_rmse_cum', 'mape_cum', 'min_error_pct_cum', 'max_error_pct_cum', 'mean_error_pct_cum', 'sd_error_pct_cum']]
                if error_method == "Absolute Percentage Error":
                    st.subheader(f"Cumulative Error Statistics (%) - M2 {model_display}")
                    st.dataframe(stats_cum.style.format({col: '{:.2f}%' for col in stats_cum.columns if col != 'game_id'}))
                else:
                    st.subheader(f"Cumulative Error Statistics - M2 {model_display}")
                    st.dataframe(stats_cum.style.format({col: '{:.4f}' for col in stats_cum.columns if col != 'game_id'}))
                
                # Cumulative Heatmap
                if not overview_df_cum.empty:
                    heatmap_data_cum = overview_df_cum[overview_df_cum['game_id'] != 'ALL_GAMES'].set_index('game_id')[[f"m{i}_error_cum" for i in range(2, 8)]]
                    x_labels_cum = [f"m{i}_error_cum" for i in range(2, 8)]
                    
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
                        title=f"Heatmap of M2 {model_display} Cumulative Forecast Errors (%) - Months 2-7",
                        xaxis_title="Forecast Month",
                        yaxis_title="Game",
                        height=600
                    )
                    st.plotly_chart(fig_heatmap_cum, use_container_width=True)
                
                # Collect data for overall comparison
                for game in all_results:
                    all_overview_data.append({
                        "game_id": game,
                        "model_type": f"M2 {model_display}",
                        "rmspe_trans": game_stats_all[game]['trans']['rmspe_trans'],
                        "rmspe_cum": game_stats_all[game]['cum']['rmspe_cum']
                    })
                    
            else:
                st.warning(f"No M2 {model_display} forecast data available for overview.")
        
        # Add overall comparison tables at the end - just like M1
        if all_overview_data:
            all_overview_df = pd.DataFrame(all_overview_data)
            
            # Best models based on transactional RMSPE
            best_models_trans = all_overview_df.loc[all_overview_df.groupby('game_id')['rmspe_trans'].idxmin()]
            best_models_trans = best_models_trans.sort_values('game_id')
            st.header("Best M2 Model Performance for Each Game (Based on Transactional RMSPE)")
            st.dataframe(best_models_trans[['game_id', 'model_type', 'rmspe_trans', 'rmspe_cum']].style.format({
                'rmspe_trans': "{:.2f}%",
                'rmspe_cum': "{:.2f}%"
            }))
            
            # Best models based on cumulative RMSPE
            best_models_cum = all_overview_df.loc[all_overview_df.groupby('game_id')['rmspe_cum'].idxmin()]
            best_models_cum = best_models_cum.sort_values('game_id')
            st.header("Best M2 Model Performance for Each Game (Based on Cumulative RMSPE)")
            st.dataframe(best_models_cum[['game_id', 'model_type', 'rmspe_trans', 'rmspe_cum']].style.format({
                'rmspe_trans': "{:.2f}%",
                'rmspe_cum': "{:.2f}%"
            }))
                
    else:
        # Individual Game Analysis
        game_id = view_option
        summary_df, forecast_plot_data, game_stats_trans, game_stats_cum = run_m2_cohort_forecasts(game_id, selected_model_type, error_method)
        if summary_df is not None and not summary_df.empty:
            # Show which M2 model is being used
            st.info(f"Using M2 {selected_model_display} model for game {game_id}")
            
            # Transactional Errors Table with Average Row
            if error_method == "Absolute Percentage Error":
                st.header(f"M2 {selected_model_display} Transactional Prediction Errors for Game: {game_id} (%)")
                format_dict_trans = {f"m{i}_error_trans": "{:.2f}%" for i in range(2, 8)}
                format_dict_trans["cohort_rmse_trans"] = "{:.2f}%"
            else:
                st.header(f"M2 {selected_model_display} Transactional Prediction Errors for Game: {game_id} (Absolute)")
                format_dict_trans = {f"m{i}_error_trans": "{:.4f}" for i in range(2, 8)}
                format_dict_trans["cohort_rmse_trans"] = "{:.4f}"
                
            trans_cols = ['cohort', 'spends'] + [f"m{i}_error_trans" for i in range(2, 8)] + ['cohort_rmse_trans']
            avg_row_trans = {
                'cohort': 'Average',
                'spends': np.nan,
                **{f'm{i}_error_trans': summary_df[f'm{i}_error_trans'].mean() for i in range(2, 8)},
                'cohort_rmse_trans': summary_df['cohort_rmse_trans'].mean()
            }
            summary_df_trans_display = pd.concat([summary_df, pd.DataFrame([avg_row_trans])], ignore_index=True)
            st.dataframe(summary_df_trans_display[trans_cols].style.format(format_dict_trans))
            if error_method == "Absolute Percentage Error":
                st.subheader(f"M2 {selected_model_display} Transactional Error Statistics (%)")
                stats_format = "{:.2f}%"
            else:
                st.subheader(f"M2 {selected_model_display} Transactional Error Statistics")
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
                st.header(f"M2 {selected_model_display} Cumulative Prediction Errors for Game: {game_id} (%)")
                format_dict_cum = {f"m{i}_error_cum": "{:.2f}%" for i in range(2, 8)}
                format_dict_cum["cohort_rmse_cum"] = "{:.2f}%"
            else:
                st.header(f"M2 {selected_model_display} Cumulative Prediction Errors for Game: {game_id}")
                format_dict_cum = {f"m{i}_error_cum": "{:.4f}" for i in range(2, 8)}
                format_dict_cum["cohort_rmse_cum"] = "{:.4f}"
                
            cum_cols = ['cohort', 'spends'] + [f"m{i}_error_cum" for i in range(2, 8)] + ['cohort_rmse_cum']
            avg_row_cum = {
                'cohort': 'Average',
                'spends': np.nan,
                **{f'm{i}_error_cum': summary_df[f'm{i}_error_cum'].mean() for i in range(2, 8)},
                'cohort_rmse_cum': summary_df['cohort_rmse_cum'].mean()
            }
            summary_df_cum_display = pd.concat([summary_df, pd.DataFrame([avg_row_cum])], ignore_index=True)
            st.dataframe(summary_df_cum_display[cum_cols].style.format(format_dict_cum))
            if error_method == "Absolute Percentage Error":
                st.subheader(f"M2 {selected_model_display} Cumulative Error Statistics (%)")
                stats_format_cum = "{:.2f}%"
            else:
                st.subheader(f"M2 {selected_model_display} Cumulative Error Statistics")
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
            st.subheader(f"M2 {selected_model_display} Prediction Curves per Cohort")
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
                    name=f"M2 {selected_model_display} Predicted Cumulative"
                ))
                fig_cum.update_layout(
                    title=f"M2 {selected_model_display} Cumulative Prediction - Cohort {cohort_str}",
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
                    name=f"M2 {selected_model_display} Predicted Transactional"
                ))
                fig_trans.update_layout(
                    title=f"M2 {selected_model_display} Transactional Prediction - Cohort {cohort_str}",
                    xaxis_title="Calendar Month",
                    yaxis_title="Transactional Value"
                )
                st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.write(f"No M2 {selected_model_display} prediction data available for game {game_id}.")
else:
    st.write("Click **Run M2 Cohort Forecasts** to generate M2 predictions starting from month 2.") 