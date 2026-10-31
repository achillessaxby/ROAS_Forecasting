import pandas as pd
import numpy as np
import os
import cloudpickle
from google.cloud import storage
import io
import streamlit as st

# Configuration
MAX_MONTHS = 12
BUCKET_NAME = "your-gcs-bucket"  # placeholder: real bucket name removed
DATA_PATH = "Models - Training/Latest Data/Segregated/"
HELPER_PATH = ""

# Cached GCS client
@st.cache_resource
def get_gcs_client():
    try:
        return storage.Client()
    except Exception:
        return None

storage_client = get_gcs_client()

@st.cache_data
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

# Games list and helper data
DEFAULT_GAMES = ['G001','G003','G018','G019','G024','G025','G027','G028','G029',
                     'G030', 'G034','G035','G036','G037','G038','G041','G042','G057',
                     'G053','G054','G059','G060','G062','G064','G067','G069','G070',
                     'G079','G074','G085','G087','G086','G088','G090','G092','G099',
                     'G100','G103','G107','G108','G115','GS006','GS015','GS023','GS065',
                     'GS075','G105','G136','G091', 'G102','GS125','G141','G142','GS999',
                     'G139','G145','G118','GS176','GS268','G154','G156','GS287','GS288',
                     'G160','G152','G111','GS189','GS129','GS264','GS274','GS349','GS342','GS242','GS331','GS332','GS338']


# ['G001','G003','G018','G019','G024','G025','G027','G028','G029',
#                      'G030', 'G034','G035','G036','G037','G038','G041','G042','G057',
#                      'G053','G054','G059','G060','G062','G064','G067','G069','G070',
#                      'G079','G074','G085','G087','G086','G088','G090','G092','G099',
#                      'G100','G103','G107','G108','G115','GS006','GS015','GS023','GS065',
#                      'GS075','G105','G136','G091', 'G102','GS125','G141','G142','GS999','G139','G145','G118']

# ['G001','G003','G018','G019','G024','G025','G030','G036','G037','G038','G053','G054','G057','G059','G062','G064','G067','G070','G074','G079','GS999']

game_metrics_df = None
category_averages_df = None
game_to_category = {}

@st.cache_data
def load_helper_data():
    global game_metrics_df, category_averages_df, game_to_category
    try:
        gm_blob = f"{HELPER_PATH}game_metrics_output.csv"
        ca_blob = f"{HELPER_PATH}category_averages_output.csv"
        game_metrics_df = read_csv_from_gcs(BUCKET_NAME, gm_blob)
        category_averages_df = read_csv_from_gcs(BUCKET_NAME, ca_blob)
        if game_metrics_df is None or category_averages_df is None:
            return False
        game_to_category = dict(zip(game_metrics_df['game_id'], game_metrics_df['Category']))
        return True
    except Exception:
        return False

def get_gamma(total_cac: float) -> float:
    if total_cac < 300000:
        return 1.0
    elif total_cac < 1000000:
        return 1.4
    else:
        return 1.7

@st.cache_data
def compute_alpha_prime(game_id: str) -> float:
    default_alpha = 0.05
    try:
        if game_metrics_df is None or category_averages_df is None or not game_to_category:
            if not load_helper_data():
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

@st.cache_data(show_spinner=False)
def list_available_games() -> list:
    from inference import available_game_ids
    return available_game_ids

@st.cache_data(ttl=3600)
def run_m1_validation(game_id: str, model_type: str, error_method: str, case: int = None):
    """
    Run M1 validation with support for multiple cases.
    
    Args:
        game_id: Game identifier
        model_type: Model type (only "Extrapolation" supported)
        error_method: Error calculation method
        case: Specific case to run (1-6), or None to run default (backward compatible)
    
    Returns:
        summary_df: DataFrame with validation results including case and predicted_on
        forecast_plot_data: Forecast data for plotting
        game_stats_trans: Transactional stats
        game_stats_cum: Cumulative stats
    """
    if model_type != "Extrapolation":
        return None, None, None, None
    
    blob_name = f"{DATA_PATH}{game_id}_Dataset_12_2024.csv"
    df_game = read_csv_from_gcs(BUCKET_NAME, blob_name)
    if df_game is None:
        return None, None, None, None
    
    df_game['transaction_period_date'] = pd.to_datetime(df_game['transaction_period'], errors='coerce', infer_datetime_format=True)
    df_game['cohort'] = pd.to_datetime(df_game['cohort'], format='%d/%m/%Y', errors='coerce')
    df_game = df_game.dropna(subset=['elapsed_months'])
    df_game['actual_date'] = df_game.apply(lambda row: row['cohort'] + pd.DateOffset(months=int(row['elapsed_months']) - 1), axis=1)
    all_cohorts = np.sort(df_game['cohort'].dropna().unique())
    
    if len(all_cohorts) < 1:
        return None, None, None, None
    
    def cohort_has_months(cohort_dt, min_months=1):
        cohort_df = df_game[df_game['cohort'] == cohort_dt]
        return cohort_df['elapsed_months'].nunique() >= min_months
    
    # For base cohort: Get LATEST cohorts with >=12 months (these will be "Predicted On" dates)
    valid_full_cohorts = sorted([co for co in all_cohorts if cohort_has_months(co, min_months=MAX_MONTHS)])
    
    # For predicted cohorts: Latest 12 with >=1 month (these are cohorts we're predicting FOR)
    valid_min_cohorts = sorted([co for co in all_cohorts if cohort_has_months(co, min_months=1)])[-12:]
    
    # Determine which case(s) to run
    if case is not None:
        # Single case mode (for backward compatibility and specific case queries)
        cases_to_run = [case]
    else:
        # Default: Run case 1 only (backward compatible)
        cases_to_run = [1]
    
    # Calculate the maximum possible cases for this game (up to 12)
    max_possible_cases = min(12, len(valid_full_cohorts))
    
    all_summary_rows = []
    all_forecast_plot_data = {}
    
    for case_num in cases_to_run:
        # Select base cohort based on case number
        # Case 1: Most recent scenario (like current behavior)
        # Case 2: Use older base cohort
        # etc.
        
        if valid_full_cohorts:
            # Calculate which base cohort to use for this case
            # Case 1 = oldest (6th from last), Case 6 = latest (most recent for extrapolation)
            base_cohort_idx = -(max_possible_cases - case_num + 1)
            
            if abs(base_cohort_idx) <= len(valid_full_cohorts):
                base_cohort_dt = valid_full_cohorts[base_cohort_idx]
            else:
                # Not enough cohorts for this case, skip
                continue
        else:
            # No cohort with >=12 months; use fallback logic
            months_per_cohort = {co: df_game[df_game['cohort'] == co]['elapsed_months'].nunique() for co in all_cohorts}
            if not months_per_cohort:
                continue
            max_months_available = max(months_per_cohort.values())
            candidates = [co for co, m in months_per_cohort.items() if m == max_months_available]
            if len(candidates) < case_num:
                continue
            base_cohort_dt = sorted(candidates)[-(case_num)]
        
        # Get base cohort CAC
        base_df_tmp = df_game[df_game['cohort'] == base_cohort_dt].sort_values('elapsed_months')
        if 'client_actual_spend' in base_df_tmp.columns or 'total_cac' in base_df_tmp.columns:
            if 'client_actual_spend' in base_df_tmp.columns:
                base_df_tmp['base_input'] = base_df_tmp.apply(
                    lambda r: r.get('client_actual_spend', np.nan) 
                    if pd.isnull(r.get('total_cac')) or r.get('total_cac', 0) == 0 
                    else r.get('total_cac', np.nan), axis=1)
            else:
                base_df_tmp['base_input'] = base_df_tmp.get('total_cac')
        else:
            base_df_tmp['base_input'] = 100000.0
        
        try:
            base_cohort_cac = base_df_tmp['base_input'].dropna().iloc[0]
        except IndexError:
            continue
        
        # Get base predictions from base cohort
        base_df = df_game[df_game['cohort'] == base_cohort_dt].sort_values('elapsed_months')
        base_predictions = {}
        for m in range(1, MAX_MONTHS + 1):
            m_df = base_df[base_df['elapsed_months'] == m]
            if not m_df.empty:
                val = m_df['transaction_period_ltv_cac'].iloc[0]
                base_predictions[m] = float(val) if pd.notnull(val) and float(val) > 0 else base_predictions.get(m-1, 0.0)
            else:
                base_predictions[m] = base_predictions.get(m-1, 0.0)
        
        alpha_prime = compute_alpha_prime(game_id)
        
        # For each predicted cohort that comes AFTER the base cohort
        predicted_cohorts = [co for co in valid_min_cohorts if co >= base_cohort_dt]
        
        for cohort_date in predicted_cohorts:
            cohort_str = pd.Timestamp(cohort_date).strftime('%Y-%m-%d')
            
            # Predicted On date is calculated relative to the cohort being predicted
            # Case 12 = cohort_date, Case 11 = cohort_date - 1 month, etc.
            months_back = max_possible_cases - case_num
            predicted_on_dt = cohort_date - pd.DateOffset(months=months_back)
            predicted_on_str = pd.Timestamp(predicted_on_dt).strftime('%Y-%m-%d')
            
            cohort_df = df_game[df_game['cohort'] == cohort_date].copy().sort_values(by='actual_date')
            
            if 'client_actual_spend' in cohort_df.columns or 'total_cac' in cohort_df.columns:
                if 'client_actual_spend' in cohort_df.columns:
                    cohort_df['base_input'] = cohort_df.apply(
                        lambda r: r.get('client_actual_spend', np.nan) 
                        if pd.isnull(r.get('total_cac')) or r.get('total_cac', 0) == 0 
                        else r.get('total_cac', np.nan), axis=1)
                else:
                    cohort_df['base_input'] = cohort_df.get('total_cac')
            else:
                cohort_df['base_input'] = 100000.0
            
            try:
                total_cac_val = cohort_df['base_input'].dropna().iloc[0]
            except IndexError:
                continue
            
            if total_cac_val <= 0:
                continue
            
            inc = (total_cac_val - base_cohort_cac) / base_cohort_cac if base_cohort_cac > 0 else 0.0
            gamma = get_gamma(total_cac_val)
            adjustment_factor = 1 / (1 + alpha_prime * gamma * inc)
            predicted_transactional = [max(base_predictions.get(m, 0.0), 1e-6) * adjustment_factor for m in range(1, MAX_MONTHS + 1)]
            predicted_cumulative = np.cumsum(predicted_transactional)
            
            forecast_df = pd.DataFrame({
                'forecast_month': [(cohort_date + pd.DateOffset(months=m-1)).strftime('%Y-%m') for m in range(1, MAX_MONTHS + 1)],
                'predicted_transactional': predicted_transactional,
                'predicted_cumulative': predicted_cumulative
            })
            
            actual_df = cohort_df[['elapsed_months', 'actual_date', 'transaction_period_ltv_cac']].drop_duplicates().sort_values(by='actual_date')
            actual_df['actual_cumulative'] = actual_df['transaction_period_ltv_cac'].cumsum()
            
            row = {
                "game_id": game_id, 
                "cohort": cohort_str, 
                "spends": total_cac_val,
                "case": case_num,
                "predicted_on": predicted_on_str
            }
            
            for idx, f_row in enumerate(forecast_df.itertuples(), start=1):
                actual_trans = actual_df[actual_df['elapsed_months'] == idx]['transaction_period_ltv_cac'].values
                actual_cum = actual_df[actual_df['elapsed_months'] == idx]['actual_cumulative'].values
                
                if actual_trans.size > 0:
                    diff_trans = f_row.predicted_transactional - actual_trans[0]
                    diff_cum = f_row.predicted_cumulative - actual_cum[0]
                    if error_method == "Absolute Percentage Error":
                        row[f"m{idx}_error_trans"] = abs(diff_trans / actual_trans[0]) * 100 if actual_trans[0] != 0 else 0
                        row[f"m{idx}_error_cum"] = abs(diff_cum / actual_cum[0]) * 100 if actual_cum[0] != 0 else 0
                    else:
                        row[f"m{idx}_error_trans"] = abs(diff_trans)
                        row[f"m{idx}_error_cum"] = abs(diff_cum)
                else:
                    row[f"m{idx}_error_trans"] = np.nan
                    row[f"m{idx}_error_cum"] = np.nan
            
            all_summary_rows.append(row)
            all_forecast_plot_data[f"{cohort_str}_case{case_num}"] = {'forecast': forecast_df, 'actual': actual_df}
    
    if not all_summary_rows:
        return None, None, None, None
    
    summary_df = pd.DataFrame(all_summary_rows)
    return summary_df, all_forecast_plot_data, {}, {}


@st.cache_data(ttl=3600)
def run_m1_validation_all_cases(game_id: str, model_type: str, error_method: str, max_cases: int = 12):
    """
    Run M1 validation for multiple cases (1 through max_cases).
    
    This generates data similar to M2's case-based structure, where each case
    represents using a different historical base cohort for extrapolation.
    
    Args:
        game_id: Game identifier
        model_type: Model type (only "Extrapolation" supported)
        error_method: Error calculation method
        max_cases: Maximum number of cases to generate (default 12)
    
    Returns:
        Combined DataFrame with all cases
    """
    if model_type != "Extrapolation":
        return None, None, None, None
    
    blob_name = f"{DATA_PATH}{game_id}_Dataset_12_2024.csv"
    df_game = read_csv_from_gcs(BUCKET_NAME, blob_name)
    if df_game is None:
        return None, None, None, None
    
    df_game['transaction_period_date'] = pd.to_datetime(df_game['transaction_period'], errors='coerce', infer_datetime_format=True)
    df_game['cohort'] = pd.to_datetime(df_game['cohort'], format='%d/%m/%Y', errors='coerce')
    df_game = df_game.dropna(subset=['elapsed_months'])
    df_game['actual_date'] = df_game.apply(lambda row: row['cohort'] + pd.DateOffset(months=int(row['elapsed_months']) - 1), axis=1)
    all_cohorts = np.sort(df_game['cohort'].dropna().unique())
    
    if len(all_cohorts) < 1:
        return None, None, None, None
    
    def cohort_has_months(cohort_dt, min_months=1):
        cohort_df = df_game[df_game['cohort'] == cohort_dt]
        return cohort_df['elapsed_months'].nunique() >= min_months
    
    # For base cohort: Get LATEST cohorts with >=12 months (these will be "Predicted On" dates)
    valid_full_cohorts = sorted([co for co in all_cohorts if cohort_has_months(co, min_months=MAX_MONTHS)])
    
    # For predicted cohorts: Latest 12 with >=1 month (these are cohorts we're predicting FOR)
    valid_min_cohorts = sorted([co for co in all_cohorts if cohort_has_months(co, min_months=1)])[-12:]
    
    # Determine how many cases we can actually generate (up to max_cases, default 12)
    # We can generate as many cases as we have valid full cohorts, up to max_cases
    num_possible_cases = min(max_cases, len(valid_full_cohorts))
    if num_possible_cases < 1:
        num_possible_cases = 1
    
    all_summary_rows = []
    all_forecast_plot_data = {}
    
    for case_num in range(1, num_possible_cases + 1):
        # Select base cohort based on case number
        if valid_full_cohorts:
            # Calculate which base cohort to use for this case
            # Case 1 = oldest (6th from last), Case 6 = latest (most recent for extrapolation)
            base_cohort_idx = -(num_possible_cases - case_num + 1)
            
            if abs(base_cohort_idx) <= len(valid_full_cohorts):
                base_cohort_dt = valid_full_cohorts[base_cohort_idx]
            else:
                # Not enough cohorts for this case, stop
                break
        else:
            # Fallback logic
            months_per_cohort = {co: df_game[df_game['cohort'] == co]['elapsed_months'].nunique() for co in all_cohorts}
            if not months_per_cohort:
                break
            max_months_available = max(months_per_cohort.values())
            candidates = [co for co, m in months_per_cohort.items() if m == max_months_available]
            if len(candidates) < case_num:
                break
            base_cohort_dt = sorted(candidates)[-(case_num)]
        
        # Get base cohort CAC
        base_df_tmp = df_game[df_game['cohort'] == base_cohort_dt].sort_values('elapsed_months')
        if 'client_actual_spend' in base_df_tmp.columns or 'total_cac' in base_df_tmp.columns:
            if 'client_actual_spend' in base_df_tmp.columns:
                base_df_tmp['base_input'] = base_df_tmp.apply(
                    lambda r: r.get('client_actual_spend', np.nan) 
                    if pd.isnull(r.get('total_cac')) or r.get('total_cac', 0) == 0 
                    else r.get('total_cac', np.nan), axis=1)
            else:
                base_df_tmp['base_input'] = base_df_tmp.get('total_cac')
        else:
            base_df_tmp['base_input'] = 100000.0
        
        try:
            base_cohort_cac = base_df_tmp['base_input'].dropna().iloc[0]
        except IndexError:
            continue
        
        # Get base predictions from base cohort
        base_df = df_game[df_game['cohort'] == base_cohort_dt].sort_values('elapsed_months')
        base_predictions = {}
        for m in range(1, MAX_MONTHS + 1):
            m_df = base_df[base_df['elapsed_months'] == m]
            if not m_df.empty:
                val = m_df['transaction_period_ltv_cac'].iloc[0]
                base_predictions[m] = float(val) if pd.notnull(val) and float(val) > 0 else base_predictions.get(m-1, 0.0)
            else:
                base_predictions[m] = base_predictions.get(m-1, 0.0)
        
        alpha_prime = compute_alpha_prime(game_id)
        
        # For each predicted cohort that comes AFTER the base cohort
        predicted_cohorts = [co for co in valid_min_cohorts if co >= base_cohort_dt]
        
        for cohort_date in predicted_cohorts:
            cohort_str = pd.Timestamp(cohort_date).strftime('%Y-%m-%d')
            
            # Predicted On date is calculated relative to the cohort being predicted
            # Case 12 = cohort_date, Case 11 = cohort_date - 1 month, etc.
            months_back = num_possible_cases - case_num
            predicted_on_dt = cohort_date - pd.DateOffset(months=months_back)
            predicted_on_str = pd.Timestamp(predicted_on_dt).strftime('%Y-%m-%d')
            
            cohort_df = df_game[df_game['cohort'] == cohort_date].copy().sort_values(by='actual_date')
            
            if 'client_actual_spend' in cohort_df.columns or 'total_cac' in cohort_df.columns:
                if 'client_actual_spend' in cohort_df.columns:
                    cohort_df['base_input'] = cohort_df.apply(
                        lambda r: r.get('client_actual_spend', np.nan) 
                        if pd.isnull(r.get('total_cac')) or r.get('total_cac', 0) == 0 
                        else r.get('total_cac', np.nan), axis=1)
                else:
                    cohort_df['base_input'] = cohort_df.get('total_cac')
            else:
                cohort_df['base_input'] = 100000.0
            
            try:
                total_cac_val = cohort_df['base_input'].dropna().iloc[0]
            except IndexError:
                continue
            
            if total_cac_val <= 0:
                continue
            
            inc = (total_cac_val - base_cohort_cac) / base_cohort_cac if base_cohort_cac > 0 else 0.0
            gamma = get_gamma(total_cac_val)
            adjustment_factor = 1 / (1 + alpha_prime * gamma * inc)
            predicted_transactional = [max(base_predictions.get(m, 0.0), 1e-6) * adjustment_factor for m in range(1, MAX_MONTHS + 1)]
            predicted_cumulative = np.cumsum(predicted_transactional)
            
            forecast_df = pd.DataFrame({
                'forecast_month': [(cohort_date + pd.DateOffset(months=m-1)).strftime('%Y-%m') for m in range(1, MAX_MONTHS + 1)],
                'predicted_transactional': predicted_transactional,
                'predicted_cumulative': predicted_cumulative
            })
            
            actual_df = cohort_df[['elapsed_months', 'actual_date', 'transaction_period_ltv_cac']].drop_duplicates().sort_values(by='actual_date')
            actual_df['actual_cumulative'] = actual_df['transaction_period_ltv_cac'].cumsum()
            
            row = {
                "game_id": game_id, 
                "cohort": cohort_str, 
                "spends": total_cac_val,
                "case": case_num,
                "predicted_on": predicted_on_str
            }
            
            for idx, f_row in enumerate(forecast_df.itertuples(), start=1):
                actual_trans = actual_df[actual_df['elapsed_months'] == idx]['transaction_period_ltv_cac'].values
                actual_cum = actual_df[actual_df['elapsed_months'] == idx]['actual_cumulative'].values
                
                if actual_trans.size > 0:
                    diff_trans = f_row.predicted_transactional - actual_trans[0]
                    diff_cum = f_row.predicted_cumulative - actual_cum[0]
                    if error_method == "Absolute Percentage Error":
                        row[f"m{idx}_error_trans"] = abs(diff_trans / actual_trans[0]) * 100 if actual_trans[0] != 0 else 0
                        row[f"m{idx}_error_cum"] = abs(diff_cum / actual_cum[0]) * 100 if actual_cum[0] != 0 else 0
                    else:
                        row[f"m{idx}_error_trans"] = abs(diff_trans)
                        row[f"m{idx}_error_cum"] = abs(diff_cum)
                else:
                    row[f"m{idx}_error_trans"] = np.nan
                    row[f"m{idx}_error_cum"] = np.nan
            
            all_summary_rows.append(row)
            all_forecast_plot_data[f"{cohort_str}_case{case_num}"] = {'forecast': forecast_df, 'actual': actual_df}
    
    if not all_summary_rows:
        return None, None, None, None
    
    summary_df = pd.DataFrame(all_summary_rows)
    return summary_df, all_forecast_plot_data, {}, {}
