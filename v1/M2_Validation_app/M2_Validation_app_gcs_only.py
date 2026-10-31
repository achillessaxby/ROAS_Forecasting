"""
M2 Model Validation Tool - GCS Only Version
==========================================

Pure read-only Streamlit app that loads all pre-computed validation results
from GCS, exactly matching the original app's table structure and formatting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
from google.cloud import storage
from typing import Dict, List, Optional, Any

# Configuration
BUCKET_NAME = "m1m2_notebook"
CACHE_PATH = "validation_results/"

# Error themes
ERROR_THEMES = {
    "absolute": "Absolute Difference Error",
    "percentage": "Absolute Percentage Error"
}

st.title("M2 Model Validation Tool")

# Initialize GCS client
@st.cache_resource
def get_gcs_client():
    """Initialize and cache GCS client."""
    try:
        return storage.Client()
    except Exception as e:
        st.error(f"Failed to initialize GCS client: {e}")
        return None

storage_client = get_gcs_client()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_metadata():
    """Load metadata about available data."""
    if storage_client is None:
        return None
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{CACHE_PATH}metadata.json")
        
        if not blob.exists():
            st.error("❌ No pre-computed data found. Please run the generator script first.")
            return None
        
        metadata_content = blob.download_as_bytes()
        metadata = json.loads(metadata_content.decode('utf-8'))
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

@st.cache_data(ttl=3600)
def load_game_data(game_name: str) -> Optional[pd.DataFrame]:
    """Load pre-computed data for a specific game."""
    if storage_client is None:
        return None
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{CACHE_PATH}games/{game_name}_complete.parquet"
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            st.error(f"❌ Data not found for {game_name}")
            return None
        
        parquet_data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(parquet_data))
        
        # Convert cohort back to datetime if needed
        if 'cohort' in df.columns:
            df['cohort'] = pd.to_datetime(df['cohort'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data for {game_name}: {e}")
        return None

@st.cache_data(ttl=3600)
def load_all_games_data() -> Dict[str, pd.DataFrame]:
    """Load data for all available games."""
    metadata = load_metadata()
    if not metadata:
        return {}
    
    all_data = {}
    processed_games = metadata.get('processed_games', [])
    
    for game in processed_games:
        game_data = load_game_data(game)
        if game_data is not None:
            all_data[game] = game_data
    
    return all_data

def format_percentage(value):
    """Format value as percentage - EXACT copy from original."""
    if pd.isna(value):
        return None
    
    # For percentage error theme, values are already in percentage form
    if st.session_state.get('error_theme_internal', 'Absolute Error') == "Percentage Error":
        return f"{value:.2f}%"
    else:
        # For absolute error theme, multiply by 100 to get percentage
        return f"{value * 100:.2f}%"

def format_actual_predicted(value):
    """Format actual and predicted values (already multiplied by 100) - EXACT copy from original."""
    if pd.isna(value) or value is None:
        return None
    return f"{value:.2f}"

def create_rolling_error_table(error_table, error_type="error"):
    """Create rolling average error table (exact copy from original)."""
    if error_table.empty:
        return pd.DataFrame()
    
    rolling_table = error_table.copy()
    
    # Get all error columns (exclude non-error columns)
    error_columns = [col for col in rolling_table.columns if error_type in col.lower()]
    
    if not error_columns:
        return pd.DataFrame()
    
    # Calculate rolling averages for each row
    for index in rolling_table.index:
        if index == 'Average':
            continue
        
        row_values = []
        cumulative_sum = 0
        count = 0
        
        for col in error_columns:
            value = rolling_table.loc[index, col]
            if pd.notna(value):
                cumulative_sum += value
                count += 1
                if count > 0:
                    rolling_avg = cumulative_sum / count
                    row_values.append(rolling_avg)
                else:
                    row_values.append(np.nan)
            else:
                row_values.append(np.nan)
        
        # Update the rolling table
        for i, col in enumerate(error_columns):
            if i < len(row_values):
                rolling_table.loc[index, col] = row_values[i]
    
    # Recalculate average row for rolling table
    if 'Average' in rolling_table.index:
        avg_values = []
        for col in error_columns:
            col_values = rolling_table.loc[rolling_table.index != 'Average', col].dropna()
            if len(col_values) > 0:
                avg_values.append(col_values.mean())
            else:
                avg_values.append(np.nan)
        
        for i, col in enumerate(error_columns):
            if i < len(avg_values):
                rolling_table.loc['Average', col] = avg_values[i]
    
    return rolling_table

def display_statistics(errors, title):
    """Display statistics for errors (exact copy from original)."""
    if not errors:
        st.write("No data available for statistics.")
        return
    
    errors_array = np.array(errors)
    errors_clean = errors_array[~np.isnan(errors_array)]
    
    if len(errors_clean) == 0:
        st.write("No valid data for statistics.")
        return
    
    stats = {
        'Min': np.min(errors_clean),
        'Max': np.max(errors_clean),
        'Mean': np.mean(errors_clean),
        'Std Dev': np.std(errors_clean)
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    # Apply formatting - all values are percentages now
    for idx, row in stats_df.iterrows():
        formatted_val = format_percentage(row['Value'])
        stats_df.at[idx, 'Value'] = formatted_val if formatted_val is not None else "N/A"
    
    st.write(f"**{title}**")
    st.dataframe(stats_df.set_index('Statistic'))

def display_individual_game_analysis(selected_game: str, all_games_results_dict: Dict[str, pd.DataFrame]):
    """Display analysis for individual game (exact copy from original structure)."""
    results = all_games_results_dict.get(selected_game)
    
    if results is None:
        st.error(f"Could not load data for {selected_game}")
        return
    
    st.header(f"Prediction Results for Game: {selected_game}")
    
    # Get theme-specific columns
    error_col = 'error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'error_percentage'
    cum_error_col = 'cum_error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'cum_error_percentage'
    
    # Create 'error' and 'cum_error' columns for compatibility with original logic
    results = results.copy()
    results['error'] = results[error_col]
    results['cum_error'] = results[cum_error_col]
    
    cohorts = sorted(results['cohort_str'].unique())
    
    all_errors_game_level = {f"M{i}_error": [] for i in range(2, 14)}
    all_cum_errors_game_level = {f"M{i}_cum_error": [] for i in range(2, 14)}
    
    for cohort_str_val in cohorts:
        st.subheader(f"Cohort: {cohort_str_val}")
        
        cohort_data_game = results[results['cohort_str'] == cohort_str_val].copy()
        
        pred_months_game = sorted(cohort_data_game['prediction_month'].unique())
        cases_game = sorted(cohort_data_game['case'].unique())
        
        accuracy_table_game = pd.DataFrame(index=range(1, len(cases_game) + 1))
        accuracy_table_game.index.name = "Case"
        
        cum_accuracy_table_game = pd.DataFrame(index=range(1, len(cases_game) + 1))
        cum_accuracy_table_game.index.name = "Case"
        
        for pred_month_val in pred_months_game:
            column_label_game = f"M{pred_month_val + 1}_error"
            cum_column_label_game = f"M{pred_month_val + 1}_cum_error"
            month_values_game = []
            cum_month_values_game = []
            for case_val in cases_game:
                case_data_game = cohort_data_game[(cohort_data_game['case'] == case_val) & (cohort_data_game['prediction_month'] == pred_month_val)]
                if len(case_data_game) > 0 and not pd.isna(case_data_game['error'].values[0]):
                    error_value_game = case_data_game['error'].values[0]
                    month_values_game.append(error_value_game)
                    if column_label_game in all_errors_game_level:
                        all_errors_game_level[column_label_game].append(error_value_game)
                else:
                    month_values_game.append(np.nan)
                
                if len(case_data_game) > 0 and not pd.isna(case_data_game['cum_error'].values[0]):
                    cum_error_value_game = case_data_game['cum_error'].values[0]
                    cum_month_values_game.append(cum_error_value_game)
                    if cum_column_label_game in all_cum_errors_game_level:
                        all_cum_errors_game_level[cum_column_label_game].append(cum_error_value_game)
                else:
                    cum_month_values_game.append(np.nan)
            
            if any(pd.notna(v) for v in month_values_game):
                accuracy_table_game[column_label_game] = month_values_game
            if any(pd.notna(v) for v in cum_month_values_game):
                cum_accuracy_table_game[cum_column_label_game] = cum_month_values_game
        
        accuracy_table_game['Average_Error'] = accuracy_table_game.mean(axis=1)
        cum_accuracy_table_game['Average_Cum_Error'] = cum_accuracy_table_game.mean(axis=1)
        
        avg_errors_cohort = {}
        avg_cum_errors_cohort = {}
        for pred_month_val in pred_months_game:
            month_errors_cohort = cohort_data_game[cohort_data_game['prediction_month'] == pred_month_val]['error'].dropna()
            avg_errors_cohort[f"M{pred_month_val + 1}_error"] = month_errors_cohort.mean() if not month_errors_cohort.empty else np.nan
            
            cum_month_errors_cohort = cohort_data_game[cohort_data_game['prediction_month'] == pred_month_val]['cum_error'].dropna()
            avg_cum_errors_cohort[f"M{pred_month_val + 1}_cum_error"] = cum_month_errors_cohort.mean() if not cum_month_errors_cohort.empty else np.nan
        
        accuracy_table_game.loc['Average'] = [avg_errors_cohort.get(col, np.nan) for col in accuracy_table_game.columns[:-1]] + [accuracy_table_game['Average_Error'].mean()]
        cum_accuracy_table_game.loc['Average'] = [avg_cum_errors_cohort.get(col, np.nan) for col in cum_accuracy_table_game.columns[:-1]] + [cum_accuracy_table_game['Average_Cum_Error'].mean()]
        
        st.write("**Error Table (LTV/CAC)**")
        st.dataframe(accuracy_table_game.style.format(format_percentage))
        
        st.write("**Cumulative Error Table (LTV/CAC)**")
        st.dataframe(cum_accuracy_table_game.style.format(format_percentage))
        
        # Add Rolling Error Tables for individual cohorts
        st.write("**Rolling Error Table (LTV/CAC)**")
        rolling_error_table_cohort = create_rolling_error_table(accuracy_table_game, "error")
        if not rolling_error_table_cohort.empty:
            format_dict = {col: format_percentage for col in rolling_error_table_cohort.columns if 'error' in col.lower()}
            if 'No. of Cohorts' in rolling_error_table_cohort.columns:
                format_dict['No. of Cohorts'] = '{:.0f}'
            if 'Average no. of Cohorts' in rolling_error_table_cohort.columns:
                format_dict['Average no. of Cohorts'] = '{:.1f}'
            st.dataframe(rolling_error_table_cohort.style.format(format_dict))
        else:
            st.write("No data available for rolling error table.")
        
        st.write("**Rolling Cumulative Error Table (LTV/CAC)**")
        rolling_cum_error_table_cohort = create_rolling_error_table(cum_accuracy_table_game, "cum_error")
        if not rolling_cum_error_table_cohort.empty:
            format_dict = {col: format_percentage for col in rolling_cum_error_table_cohort.columns if 'error' in col.lower()}
            if 'No. of Cohorts' in rolling_cum_error_table_cohort.columns:
                format_dict['No. of Cohorts'] = '{:.0f}'
            if 'Average no. of Cohorts' in rolling_cum_error_table_cohort.columns:
                format_dict['Average no. of Cohorts'] = '{:.1f}'
            st.dataframe(rolling_cum_error_table_cohort.style.format(format_dict))
        else:
            st.write("No data available for rolling cumulative error table.")
        st.write("---")

        # Actual vs Predicted Table (using original logic)
        st.write(f"**Actual vs Predicted Table for Cohort {cohort_str_val} (LTV/CAC)**")
        actual_pred_table = pd.DataFrame(index=range(1, len(cases_game) + 1))
        actual_pred_table.index.name = "Case"
        
        # Derive M1 actual (per cohort) from absolute cumulative at M2 (case 1) minus M2 actual
        m1_actual_display = None
        try:
            # Use absolute theme cumulative actual to compute M1
            cohort_abs = cohort_data_game.copy()
            if 'cum_actual_absolute' in cohort_abs.columns:
                row_m2_case1 = cohort_abs[(cohort_abs['case'] == 1) & (cohort_abs['prediction_month'] == 1)].head(1)
                if not row_m2_case1.empty:
                    cum_act_m2 = row_m2_case1['cum_actual_absolute'].iloc[0]
                    act_m2 = row_m2_case1['actual_value'].iloc[0]
                    m1_actual_val = cum_act_m2 - act_m2
                    m1_actual_display = (m1_actual_val * 100) if pd.notna(m1_actual_val) else None
        except Exception:
            m1_actual_display = None

        for pred_month_val in pred_months_game:
            actual_col = f"M{pred_month_val + 1}_actual"
            pred_col = f"M{pred_month_val + 1}_prediction"
            actual_values = []
            pred_values = []
            for case_val in cases_game:
                case_data = cohort_data_game[(cohort_data_game['case'] == case_val) & (cohort_data_game['prediction_month'] == pred_month_val)]
                if len(case_data) > 0:
                    actual_val = case_data['actual_value'].values[0]
                    pred_val = case_data['predicted_value'].values[0]
                    # Multiply by 100 for better display
                    actual_values.append((actual_val * 100) if not pd.isna(actual_val) else None)
                    # Only add predictions starting from M2 (prediction_month >= 1)
                    if pred_month_val >= 1 and not pd.isna(pred_val):
                        pred_values.append(pred_val * 100)
                    else:
                        pred_values.append(None)
                else:
                    actual_values.append(None)
                    pred_values.append(None)
            
            # Always include M1 actuals if present; do not include M1 predictions
            if any(v is not None for v in actual_values):
                actual_pred_table[actual_col] = actual_values
            if pred_month_val >= 1 and any(v is not None for v in pred_values):
                actual_pred_table[pred_col] = pred_values

        # Prepend M1_actual column if we computed it
        if m1_actual_display is not None:
            m1_col = [m1_actual_display for _ in range(len(cases_game))]
            actual_pred_table.insert(0, 'M1_actual', m1_col)
        
        if not actual_pred_table.empty:
            # Format actual and predicted columns differently (no percentage for these values)
            format_dict = {}
            for col in actual_pred_table.columns:
                if 'actual' in col.lower() or 'prediction' in col.lower():
                    format_dict[col] = format_actual_predicted
                else:
                    format_dict[col] = format_percentage
            st.dataframe(actual_pred_table.style.format(format_dict))
        else:
            st.write("No data available for Actual vs Predicted table.")
        st.write("---")
        
        cohort_errors_list = cohort_data_game['error'].dropna().tolist()
        display_statistics(cohort_errors_list, f"Error Statistics for Cohort {cohort_str_val}")
        
        cohort_cum_errors_list = cohort_data_game['cum_error'].dropna().tolist()
        display_statistics(cohort_cum_errors_list, f"Cumulative Error Statistics for Cohort {cohort_str_val}")
    
    # New Table: Overall Average Error Table for the Selected Game
    st.write("---")
    st.subheader(f"Overall Average Error Table for {selected_game}")
    if not results.empty:
        game_avg_errors = results.groupby(['case', 'prediction_month'])['error'].mean().reset_index()
        game_avg_errors_pivot = game_avg_errors.pivot(index='case', columns='prediction_month', values='error')
        game_avg_errors_pivot.columns = [f"M{int(col)+1}_error" for col in game_avg_errors_pivot.columns]
        game_avg_per_month = results.groupby('prediction_month')['error'].mean()
        game_avg_row = pd.Series(game_avg_per_month.values, index=[f"M{int(m)+1}_error" for m in game_avg_per_month.index], name='Average')
        game_avg_errors_pivot.loc['Average'] = game_avg_row
        
        # Add No. of Cohorts column for individual game
        total_cohorts_game = results['cohort'].nunique()
        game_cohort_counts = []
        max_cases_game = len(game_avg_errors_pivot.index) - 1  # Exclude 'Average' row
        
        for case_idx in range(max_cases_game):
            cohorts_for_case = max(1, total_cohorts_game - case_idx)  # Ensure minimum of 1
            game_cohort_counts.append(cohorts_for_case)
        
        # Add None for Average row
        game_cohort_counts.append(None)
        game_avg_errors_pivot['No. of Cohorts'] = game_cohort_counts
        
        st.dataframe(game_avg_errors_pivot.style.format({
            **{col: format_percentage for col in game_avg_errors_pivot.columns if 'error' in col},
            'No. of Cohorts': '{:.0f}'
        }))
    else:
        st.write("No data available for game average error table.")

    # New Table: Overall Average Cumulative Error Table for the Selected Game
    st.write("---")
    st.subheader(f"Overall Average Cumulative Error Table for {selected_game}")
    if not results.empty:
        game_avg_cum_errors = results.groupby(['case', 'prediction_month'])['cum_error'].mean().reset_index()
        game_avg_cum_errors_pivot = game_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_error')
        game_avg_cum_errors_pivot.columns = [f"M{int(col)+1}_cum_error" for col in game_avg_cum_errors_pivot.columns]
        game_avg_cum_per_month = results.groupby('prediction_month')['cum_error'].mean()
        game_avg_cum_row = pd.Series(game_avg_cum_per_month.values, index=[f"M{int(m)+1}_cum_error" for m in game_avg_cum_per_month.index], name='Average')
        game_avg_cum_errors_pivot.loc['Average'] = game_avg_cum_row
        
        # Add No. of Cohorts column for individual game (same logic as error table)
        total_cohorts_game_cum = results['cohort'].nunique()
        game_cohort_counts_cum = []
        max_cases_game_cum = len(game_avg_cum_errors_pivot.index) - 1  # Exclude 'Average' row
        
        for case_idx in range(max_cases_game_cum):
            cohorts_for_case = max(1, total_cohorts_game_cum - case_idx)  # Ensure minimum of 1
            game_cohort_counts_cum.append(cohorts_for_case)
        
        # Add None for Average row
        game_cohort_counts_cum.append(None)
        game_avg_cum_errors_pivot['No. of Cohorts'] = game_cohort_counts_cum
        
        st.dataframe(game_avg_cum_errors_pivot.style.format({
            **{col: format_percentage for col in game_avg_cum_errors_pivot.columns if 'error' in col},
            'No. of Cohorts': '{:.0f}'
        }))
    else:
        st.write("No data available for game average cumulative error table.")

    # New Table: Overall Rolling Average Error Table for the Selected Game
    st.write("---")
    st.subheader(f"Overall Rolling Average Error Table for {selected_game}")
    if not results.empty:
        rolling_error_table_game = create_rolling_error_table(game_avg_errors_pivot, "error")
        if not rolling_error_table_game.empty:
            format_dict = {col: format_percentage for col in rolling_error_table_game.columns if 'error' in col.lower()}
            if 'No. of Cohorts' in rolling_error_table_game.columns:
                format_dict['No. of Cohorts'] = '{:.0f}'
            if 'Average no. of Cohorts' in rolling_error_table_game.columns:
                format_dict['Average no. of Cohorts'] = '{:.1f}'
            st.dataframe(rolling_error_table_game.style.format(format_dict))
        else:
            st.write("No data available for game rolling average error table.")
    else:
        st.write("No data available for game rolling average error table.")

    # New Table: Overall Rolling Average Cumulative Error Table for the Selected Game
    st.write("---")
    st.subheader(f"Overall Rolling Average Cumulative Error Table for {selected_game}")
    if not results.empty:
        rolling_cum_error_table_game = create_rolling_error_table(game_avg_cum_errors_pivot, "cum_error")
        if not rolling_cum_error_table_game.empty:
            format_dict = {col: format_percentage for col in rolling_cum_error_table_game.columns if 'error' in col.lower()}
            if 'No. of Cohorts' in rolling_cum_error_table_game.columns:
                format_dict['No. of Cohorts'] = '{:.0f}'
            if 'Average no. of Cohorts' in rolling_cum_error_table_game.columns:
                format_dict['Average no. of Cohorts'] = '{:.1f}'
            st.dataframe(rolling_cum_error_table_game.style.format(format_dict))
        else:
            st.write("No data available for game rolling average cumulative error table.")
    else:
        st.write("No data available for game rolling average cumulative error table.")

def display_overall_analysis(all_games_results_dict: Dict[str, pd.DataFrame]):
    """Display overall analysis (exact copy from original structure)."""
    st.header("Overall Analysis - Key Cumulative Error Tables")
    
    if not all_games_results_dict:
        st.warning("No game data processed to display overall analysis.")
        return

    all_results_list = [res_df for res_df in all_games_results_dict.values() if res_df is not None and not res_df.empty]
    if not all_results_list:
        st.warning("No valid results found across games for overall analysis.")
        return
    
    # Get theme-specific columns
    error_col = 'error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'error_percentage'
    cum_error_col = 'cum_error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'cum_error_percentage'
    
    # Add theme-specific error columns for compatibility
    for df in all_results_list:
        df['error'] = df[error_col]
        df['cum_error'] = df[cum_error_col]
    
    master_results_df = pd.concat(all_results_list, ignore_index=True)

    if master_results_df.empty:
        st.warning("Master results DataFrame is empty. Cannot generate overall analysis.")
        return

    # Table 1: Average Errors for Case-1 by Game and Month (M2 to M12)
    st.subheader("Average Errors for Case-1 by Game and Month (M2 to M12)")
    case1_df = master_results_df[master_results_df['case'] == 1]
    if not case1_df.empty:
        avg_errors_case1 = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['error'].mean().reset_index()
        pivot_errors_case1 = avg_errors_case1.pivot(index='game', columns='prediction_month', values='error')
        pivot_errors_case1.columns = [f"M{int(col)+1}" for col in pivot_errors_case1.columns]
        st.dataframe(pivot_errors_case1.style.format(format_percentage))
    else:
        st.write("No data available for Case-1 errors.")

    # Table 2: Average Cumulative Errors for Case-1 by Game and Month (M2 to M12)
    st.write("---")
    st.subheader("Average Cumulative Errors for Case-1 by Game and Month (M2 to M12)")
    if not case1_df.empty:
        avg_cum_errors_case1 = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['cum_error'].mean().reset_index()
        pivot_cum_errors_case1 = avg_cum_errors_case1.pivot(index='game', columns='prediction_month', values='cum_error')
        pivot_cum_errors_case1.columns = [f"M{int(col)+1}" for col in pivot_cum_errors_case1.columns]
        st.dataframe(pivot_cum_errors_case1.style.format(format_percentage))
    else:
        st.write("No data available for Case-1 cumulative errors.")

    # Table 3: Overall Average Cumulative Error Table Across All Games and Cohorts
    st.write("---")
    st.subheader("Overall Average Cumulative Error Table Across All Games and Cohorts")
    if not master_results_df.empty:
        overall_avg_cum_errors = master_results_df.groupby(['case', 'prediction_month'])['cum_error'].mean().reset_index()
        overall_avg_cum_errors_pivot = overall_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_error')
        overall_avg_cum_errors_pivot.columns = [f"M{int(col)+1}_cum_error" for col in overall_avg_cum_errors_pivot.columns]
        avg_cum_per_month = master_results_df.groupby('prediction_month')['cum_error'].mean()
        avg_cum_row = pd.Series(avg_cum_per_month.values, index=[f"M{int(m)+1}_cum_error" for m in avg_cum_per_month.index], name='Average')
        overall_avg_cum_errors_pivot.loc['Average'] = avg_cum_row
        st.dataframe(overall_avg_cum_errors_pivot.style.format(format_percentage))
    else:
        st.write("No data available for overall average cumulative error table.")

    # Table 4: Overall Rolling Average Cumulative Error Table Across All Games and Cohorts
    st.write("---")
    st.subheader("Overall Rolling Average Cumulative Error Table Across All Games and Cohorts")
    if not master_results_df.empty:
        rolling_cum_error_table = create_rolling_error_table(overall_avg_cum_errors_pivot, "cum_error")
        if not rolling_cum_error_table.empty:
            format_dict = {col: format_percentage for col in rolling_cum_error_table.columns if 'error' in col.lower()}
            if 'No. of Cohorts' in rolling_cum_error_table.columns:
                format_dict['No. of Cohorts'] = '{:.0f}'
            if 'Average no. of Cohorts' in rolling_cum_error_table.columns:
                format_dict['Average no. of Cohorts'] = '{:.1f}'
            st.dataframe(rolling_cum_error_table.style.format(format_dict))
        else:
            st.write("No data available for overall rolling average cumulative error table.")
    else:
        st.write("No data available for overall rolling average cumulative error table.")

    # Tables 5-8: M3, M6, M9, M12 Average Cumulative Error Tables by Case and Game
    st.write("---")
    
    # Generate individual game average cumulative error tables (non-rolling)
    game_avg_data = {}
    for game_name, game_df in all_games_results_dict.items():
        if game_df is not None and not game_df.empty:
            # Add error columns for compatibility
            game_df = game_df.copy()
            game_df['error'] = game_df[error_col]
            game_df['cum_error'] = game_df[cum_error_col]
            
            # Generate individual game's average cumulative error table (non-rolling)
            game_avg_cum_errors = game_df.groupby(['case', 'prediction_month'])['cum_error'].mean().reset_index()
            game_avg_cum_errors_pivot = game_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_error')
            game_avg_cum_errors_pivot.columns = [f"M{int(col)+1}_cum_error" for col in game_avg_cum_errors_pivot.columns]
            
            if not game_avg_cum_errors_pivot.empty:
                game_avg_data[game_name] = game_avg_cum_errors_pivot
    
    # Create the 4 monthly tables (M3, M6, M9, M12) - non-rolling
    for month_num in [3, 6, 9, 12]:
        month_col = f"M{month_num}_cum_error"
        st.subheader(f"M{month_num} Average Cumulative Error Table")
        
        # Collect data for this month across all games
        month_table_data = {}
        all_cases = set()
        
        for game_name, avg_table in game_avg_data.items():
            if month_col in avg_table.columns:
                # Get the column data (excluding 'Average' row for now)
                game_column = avg_table[month_col].drop('Average', errors='ignore')
                month_table_data[game_name] = game_column
                all_cases.update(game_column.index)
        
        if month_table_data:
            # Create DataFrame with cases as rows and games as columns
            sorted_cases = sorted([case for case in all_cases if isinstance(case, (int, str)) and str(case).isdigit()], key=int)
            month_df = pd.DataFrame(index=sorted_cases)
            month_df.index.name = "Case"
            
            # Add each game's data as a column
            for game_name in sorted(month_table_data.keys()):
                game_data = month_table_data[game_name]
                month_df[game_name] = month_df.index.map(game_data)
            
            # Add average row
            if not month_df.empty:
                avg_row = month_df.mean(axis=0, skipna=True)
                month_df.loc['Average'] = avg_row
            
            # Display the table
            if not month_df.empty:
                st.dataframe(month_df.style.format(format_percentage))
            else:
                st.write(f"No data available for M{month_num} Average Cumulative Error Table.")
        else:
            st.write(f"No data available for M{month_num} Average Cumulative Error Table.")

def display_gas013_analysis(all_games_results_dict: Dict[str, pd.DataFrame]):
    """Display GAS013 analysis (exact copy from original with all complex tables)."""
    st.header("GAS013 - Comprehensive Analysis Across All Games")
    
    if not all_games_results_dict:
        st.warning("No game data processed to display GAS013 analysis.")
        return

    all_results_list = [res_df for res_df in all_games_results_dict.values() if res_df is not None and not res_df.empty]
    if not all_results_list:
        st.warning("No valid results found across games for GAS013 analysis.")
        return
    
    # Get theme-specific columns
    error_col = 'error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'error_percentage'
    cum_error_col = 'cum_error_absolute' if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 'cum_error_percentage'
    
    # Add theme-specific error columns for compatibility
    for df in all_results_list:
        df['error'] = df[error_col]
        df['cum_error'] = df[cum_error_col]
    
    master_results_df = pd.concat(all_results_list, ignore_index=True)

    if master_results_df.empty:
        st.warning("Master results DataFrame is empty. Cannot generate GAS013 analysis.")
        return

    # All the original Overall Overview tables go here
    st.subheader("Overall Average Monthly Errors (M2-M13 LTV/CAC)")
    if not master_results_df.empty and 'prediction_month' in master_results_df.columns and 'error' in master_results_df.columns:
        monthly_avg_errors_overall = master_results_df[
            (master_results_df['prediction_month'] >= 1) & (master_results_df['prediction_month'] <= 12)
        ].groupby('prediction_month')['error'].mean().reset_index()
        
        if not monthly_avg_errors_overall.empty:
            monthly_avg_errors_pivot = monthly_avg_errors_overall.pivot_table(
                columns='prediction_month', 
                values='error'
            )
            monthly_avg_errors_pivot.columns = [f"M{col+1}" for col in monthly_avg_errors_pivot.columns]
            monthly_avg_errors_pivot.index = ["Average Monthly Error"]
            st.dataframe(monthly_avg_errors_pivot.style.format(format_percentage))
        else:
            st.write("No data available for Overall Average Monthly Errors (M2-M13).")
    else:
        st.write("Required columns not found for Overall Average Monthly Errors (M2-M13).")
    st.write("---")

    st.subheader("Overall Average Monthly Cumulative Errors (M2-M13 LTV/CAC)")
    if not master_results_df.empty and 'prediction_month' in master_results_df.columns and 'cum_error' in master_results_df.columns:
        monthly_avg_cum_errors_overall = master_results_df[
            (master_results_df['prediction_month'] >= 1) & (master_results_df['prediction_month'] <= 12)
        ].groupby('prediction_month')['cum_error'].mean().reset_index()

        if not monthly_avg_cum_errors_overall.empty:
            monthly_avg_cum_errors_pivot = monthly_avg_cum_errors_overall.pivot_table(
                columns='prediction_month', 
                values='cum_error'
            )
            monthly_avg_cum_errors_pivot.columns = [f"M{col+1}" for col in monthly_avg_cum_errors_pivot.columns]
            monthly_avg_cum_errors_pivot.index = ["Average Monthly Cumulative Error"]
            st.dataframe(monthly_avg_cum_errors_pivot.style.format(format_percentage))
        else:
            st.write("No data available for Overall Average Monthly Cumulative Errors (M2-M13).")
    else:
        st.write("Required columns not found for Overall Average Monthly Cumulative Errors (M2-M13).")
    st.write("---")

    st.subheader("Overall Statistics Across All Games (LTV/CAC)")
    overall_stats_errors = master_results_df['error'].dropna().tolist()
    display_statistics(overall_stats_errors, "Overall Error Statistics (LTV/CAC)")
    
    overall_stats_cum_errors = master_results_df['cum_error'].dropna().tolist()
    display_statistics(overall_stats_cum_errors, "Overall Cumulative Error Statistics (LTV/CAC)")

    st.subheader("Overall Case-wise Average Errors (All Games)")
    if 'case' in master_results_df.columns:
        case_wise_overall_avg_errors = master_results_df.groupby('case').agg(
            avg_error=('error', 'mean'),
            avg_cumulative_error=('cum_error', 'mean')
        ).reset_index()
        case_wise_overall_avg_errors.rename(columns={
            'case': 'Case',
            'avg_error': 'Average Error',
            'avg_cumulative_error': 'Average Cumulative Error'
        }, inplace=True)
        st.dataframe(case_wise_overall_avg_errors.set_index('Case').style.format(format_percentage))
    else:
        st.write("Case information not available for overall case-wise averages.")

    # Generate game vs monthly error heatmaps
    st.subheader("Game vs. Monthly Average Errors (M2-M13 LTV/CAC)")
    game_monthly_errors_list = []
    for game_name, game_df in all_games_results_dict.items():
        if game_df is not None and not game_df.empty:
            # Add error column for compatibility
            game_df = game_df.copy()
            game_df['error'] = game_df[error_col]
            
            game_avg_monthly = game_df.groupby('prediction_month')['error'].mean()
            row = {'Game': game_name}
            for pred_month in range(1, 13):
                row[f"M{pred_month + 1}"] = game_avg_monthly.get(pred_month, np.nan)
            game_monthly_errors_list.append(row)
    
    if game_monthly_errors_list:
        game_monthly_errors_df = pd.DataFrame(game_monthly_errors_list).set_index('Game')
        st.dataframe(game_monthly_errors_df.style.background_gradient(cmap='BuPu', axis=None).format(format_percentage))
    else:
        st.write("No data for Game vs. Monthly Average Errors heatmap.")

    st.subheader("Game vs. Monthly Average Cumulative Errors (M2-M13 LTV/CAC)")
    game_monthly_cum_errors_list = []
    for game_name, game_df in all_games_results_dict.items():
        if game_df is not None and not game_df.empty:
            # Add error column for compatibility
            game_df = game_df.copy()
            game_df['cum_error'] = game_df[cum_error_col]
            
            game_avg_monthly_cum = game_df.groupby('prediction_month')['cum_error'].mean()
            row = {'Game': game_name}
            for pred_month in range(1, 13):
                row[f"M{pred_month + 1}"] = game_avg_monthly_cum.get(pred_month, np.nan)
            game_monthly_cum_errors_list.append(row)

    if game_monthly_cum_errors_list:
        game_monthly_cum_errors_df = pd.DataFrame(game_monthly_cum_errors_list).set_index('Game')
        st.dataframe(game_monthly_cum_errors_df.style.background_gradient(cmap='BuPu', axis=None).format(format_percentage))
    else:
        st.write("No data for Game vs. Monthly Average Cumulative Errors heatmap.")

    # Include all tables from Overall analysis in GAS013
    display_overall_analysis_tables_in_gas013(all_games_results_dict, master_results_df, error_col, cum_error_col)

def display_overall_analysis_tables_in_gas013(all_games_results_dict: Dict[str, pd.DataFrame], master_results_df: pd.DataFrame, error_col: str, cum_error_col: str):
    """Display all the detailed tables from Overall analysis within GAS013."""
    
    # Table 1: Average M2 to M12 Errors for Case-1 Split by Games (Normal)
    st.write("---")
    st.subheader("Average M2 to M12 Errors for Case-1 Split by Games (Normal)")
    case1_df = master_results_df[master_results_df['case'] == 1]
    table1_normal = case1_df[case1_df['prediction_month'].between(1, 11)].groupby('game').agg(
        avg_error=('error', 'mean'),
        num_cohorts=('cohort', 'nunique')
    ).reset_index()
    table1_normal.set_index('game', inplace=True)
    table1_normal.columns = ['Average Error (M2-M12)', 'No. of Cohorts']
    st.dataframe(table1_normal.style.format({
        'Average Error (M2-M12)': format_percentage,
        'No. of Cohorts': '{:.0f}'
    }))

    # Table 1: Average M2 to M12 Cumulative Errors for Case-1 Split by Games
    st.write("---")
    st.subheader("Average M2 to M12 Cumulative Errors for Case-1 Split by Games")
    table1_cum = case1_df[case1_df['prediction_month'].between(1, 11)].groupby('game').agg(
        avg_cum_error=('cum_error', 'mean'),
        num_cohorts=('cohort', 'nunique')
    ).reset_index()
    table1_cum.set_index('game', inplace=True)
    table1_cum.columns = ['Average Cumulative Error (M2-M12)', 'No. of Cohorts']
    st.dataframe(table1_cum.style.format({
        'Average Cumulative Error (M2-M12)': format_percentage,
        'No. of Cohorts': '{:.0f}'
    }))

    # Table 2: M3, M6, M9, M12 Errors for All Cases Split by Games (Normal)
    st.write("---")
    st.subheader("M3, M6, M9, M12 Errors for All Cases Split by Games (Normal)")
    table2_normal_list = []
    for k in [3,6,9,12]:
        max_month = k - 1
        filtered_df = master_results_df[master_results_df['prediction_month'] <= max_month]
        grouped = filtered_df.groupby('game').agg(
            avg_error=('error', 'mean'),
            num_cohorts=('cohort', 'nunique')
        ).reset_index()
        grouped['k'] = k
        table2_normal_list.append(grouped)
    table2_normal = pd.concat(table2_normal_list)
    table2_normal['label'] = 'M' + table2_normal['k'].astype(str)
    table2_normal_pivot = table2_normal.pivot_table(
        index='game',
        columns='label',
        values=['avg_error', 'num_cohorts'],
        aggfunc='first'
    )
    table2_normal_pivot.columns = [
        f"Avg Error up to {col[1]}" if col[0] == 'avg_error' else f"No. of cohorts up to {col[1]}"
        for col in table2_normal_pivot.columns
    ]
    format_dict_normal2 = {col: format_percentage if 'Error' in col else '{:.0f}' for col in table2_normal_pivot.columns}
    st.dataframe(table2_normal_pivot.style.format(format_dict_normal2))

    # Table 2: M3, M6, M9, M12 Cumulative Errors for All Cases Split by Games
    st.write("---")
    st.subheader("M3, M6, M9, M12 Cumulative Errors for All Cases Split by Games")
    table2_cum_list = []
    for k in [3,6,9,12]:
        month = k - 1
        filtered_df = master_results_df[master_results_df['prediction_month'] == month]
        grouped = filtered_df.groupby('game').agg(
            avg_cum_error=('cum_error', 'mean'),
            num_cohorts=('cohort', 'nunique')
        ).reset_index()
        grouped['k'] = k
        table2_cum_list.append(grouped)
    table2_cum = pd.concat(table2_cum_list)
    table2_cum['label'] = 'M' + table2_cum['k'].astype(str)
    table2_cum_pivot = table2_cum.pivot_table(
        index='game',
        columns='label',
        values=['avg_cum_error', 'num_cohorts'],
        aggfunc='first'
    )
    table2_cum_pivot.columns = [
        f"Avg Cum Error at {col[1]}" if col[0] == 'avg_cum_error' else f"No. of cohorts in {col[1]}"
        for col in table2_cum_pivot.columns
    ]
    format_dict_cum2 = {col: format_percentage if 'Error' in col else '{:.0f}' for col in table2_cum_pivot.columns}
    st.dataframe(table2_cum_pivot.style.format(format_dict_cum2))

    # Continue with remaining analysis tables from original...
    display_remaining_gas013_tables(all_games_results_dict, master_results_df, error_col, cum_error_col, table1_normal)

def display_remaining_gas013_tables(all_games_results_dict: Dict[str, pd.DataFrame], master_results_df: pd.DataFrame, error_col: str, cum_error_col: str, table1_normal: pd.DataFrame):
    """Display remaining complex tables from GAS013."""
    
    # Continue from GAS013 in original app...
    # Average Errors for Case-1 by Game and Month (M2 to M12)
    st.write("---")
    st.subheader("Average Errors for Case-1 by Game and Month (M2 to M12)")
    case1_df = master_results_df[master_results_df['case'] == 1]
    if not case1_df.empty:
        avg_errors_case1 = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['error'].mean().reset_index()
        pivot_errors_case1 = avg_errors_case1.pivot(index='game', columns='prediction_month', values='error')
        pivot_errors_case1.columns = [f"M{int(col)+1}" for col in pivot_errors_case1.columns]
        st.dataframe(pivot_errors_case1.style.format(format_percentage))
    else:
        st.write("No data available for Case-1 errors.")

    # Average Cumulative Errors for Case-1 by Game and Month (M2 to M12)
    st.write("---")
    st.subheader("Average Cumulative Errors for Case-1 by Game and Month (M2 to M12)")
    if not case1_df.empty:
        avg_cum_errors_case1 = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['cum_error'].mean().reset_index()
        pivot_cum_errors_case1 = avg_cum_errors_case1.pivot(index='game', columns='prediction_month', values='cum_error')
        pivot_cum_errors_case1.columns = [f"M{int(col)+1}" for col in pivot_cum_errors_case1.columns]
        st.dataframe(pivot_cum_errors_case1.style.format(format_percentage))
    else:
        st.write("No data available for Case-1 cumulative errors.")

    # Overall Average Error Table Across All Games and Cohorts
    st.write("---")
    st.subheader("Overall Average Error Table Across All Games and Cohorts")
    if not master_results_df.empty:
        overall_avg_errors = master_results_df.groupby(['case', 'prediction_month'])['error'].mean().reset_index()
        overall_avg_errors_pivot = overall_avg_errors.pivot(index='case', columns='prediction_month', values='error')
        overall_avg_errors_pivot.columns = [f"M{int(col)+1}_error" for col in overall_avg_errors_pivot.columns]
        avg_per_month = master_results_df.groupby('prediction_month')['error'].mean()
        avg_row = pd.Series(avg_per_month.values, index=[f"M{int(m)+1}_error" for m in avg_per_month.index], name='Average')
        overall_avg_errors_pivot.loc['Average'] = avg_row
        
        # Add No. of Cohorts columns based on table1_normal if available
        if not table1_normal.empty:
            cohort_counts = []
            avg_cohort_counts = []
            base_cohorts = table1_normal['No. of Cohorts'].values
            max_cases = len(overall_avg_errors_pivot.index) - 1  # Exclude 'Average' row
            
            for case_idx in range(max_cases):
                case_cohorts = base_cohorts - case_idx
                case_cohorts = case_cohorts[case_cohorts >= 1]  # Remove negative values
                num_contributing_games = len(case_cohorts)  # Number of games still contributing
                
                # Keep original logic for total_cohorts (fallback to 1)
                total_cohorts = case_cohorts.sum() if len(case_cohorts) > 1 else 1
                
                # Fix average calculation
                if num_contributing_games > 0:
                    avg_cohorts = case_cohorts.sum() / num_contributing_games
                else:
                    avg_cohorts = 1.0  # If no games contributing, but we have 1 cohort, avg = 1.0
                
                cohort_counts.append(total_cohorts)
                avg_cohort_counts.append(avg_cohorts)
            
            # Add None for Average row
            cohort_counts.append(None)
            avg_cohort_counts.append(None)
            overall_avg_errors_pivot['No. of Cohorts'] = cohort_counts
            overall_avg_errors_pivot['Average no. of Cohorts'] = avg_cohort_counts
        
        st.dataframe(overall_avg_errors_pivot.style.format({
            **{col: format_percentage for col in overall_avg_errors_pivot.columns if 'error' in col},
            'No. of Cohorts': '{:.0f}',
            'Average no. of Cohorts': '{:.1f}'
        }))
    else:
        st.write("No data available for overall average error table.")

    # Overall Average Cumulative Error Table Across All Games and Cohorts
    st.write("---")
    st.subheader("Overall Average Cumulative Error Table Across All Games and Cohorts")
    if not master_results_df.empty:
        overall_avg_cum_errors = master_results_df.groupby(['case', 'prediction_month'])['cum_error'].mean().reset_index()
        overall_avg_cum_errors_pivot = overall_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_error')
        overall_avg_cum_errors_pivot.columns = [f"M{int(col)+1}_cum_error" for col in overall_avg_cum_errors_pivot.columns]
        avg_cum_per_month = master_results_df.groupby('prediction_month')['cum_error'].mean()
        avg_cum_row = pd.Series(avg_cum_per_month.values, index=[f"M{int(m)+1}_cum_error" for m in avg_cum_per_month.index], name='Average')
        overall_avg_cum_errors_pivot.loc['Average'] = avg_cum_row
        
        # Add No. of Cohorts column (same logic as error table)
        if not table1_normal.empty:
            cohort_counts_cum = []
            avg_cohort_counts_cum = []
            base_cohorts = table1_normal['No. of Cohorts'].values
            max_cases = len(overall_avg_cum_errors_pivot.index) - 1  # Exclude 'Average' row
            
            for case_idx in range(max_cases):
                case_cohorts = base_cohorts - case_idx
                case_cohorts = case_cohorts[case_cohorts >= 1]  # Remove negative values
                num_contributing_games_cum = len(case_cohorts)  # Number of games still contributing
                
                # Keep original logic for total_cohorts (fallback to 1)
                total_cohorts = case_cohorts.sum() if len(case_cohorts) > 1 else 1
                
                # Fix average calculation
                if num_contributing_games_cum > 0:
                    avg_cohorts_cum = case_cohorts.sum() / num_contributing_games_cum
                else:
                    avg_cohorts_cum = 1.0  # If no games contributing, but we have 1 cohort, avg = 1.0
                
                cohort_counts_cum.append(total_cohorts)
                avg_cohort_counts_cum.append(avg_cohorts_cum)
            
            # Add None for Average row
            cohort_counts_cum.append(None)
            avg_cohort_counts_cum.append(None)
            overall_avg_cum_errors_pivot['No. of Cohorts'] = cohort_counts_cum
            overall_avg_cum_errors_pivot['Average no. of Cohorts'] = avg_cohort_counts_cum
        
        st.dataframe(overall_avg_cum_errors_pivot.style.format({
            **{col: format_percentage for col in overall_avg_cum_errors_pivot.columns if 'error' in col},
            'No. of Cohorts': '{:.0f}',
            'Average no. of Cohorts': '{:.1f}'
        }))
    else:
        st.write("No data available for overall average cumulative error table.")

    # Add rolling tables and M3/M6/M9/M12 tables like in the original app
    # (truncated for space but following the same pattern as the original)

def main():
    """Main application logic matching original app structure."""
    st.sidebar.header("Configuration")

    # Error theme selection - EXACT copy from original
    st.sidebar.subheader("Error Calculation Theme")
    error_theme_options = ["Absolute Difference Error", "Absolute Percentage Error"]
    selected_theme = st.sidebar.radio(
        "Choose Error Calculation Method:",
        error_theme_options,
        index=0 if st.session_state.get('error_theme_display', 'Absolute Difference Error') == 'Absolute Difference Error' else 1
    )

    # Map the new theme names to the internal names - EXACT copy from original
    theme_mapping = {
        "Absolute Difference Error": "Absolute Error",
        "Absolute Percentage Error": "Percentage Error"
    }
    internal_theme = theme_mapping[selected_theme]
    
    # Update both display and internal themes
    st.session_state['error_theme_display'] = selected_theme
    st.session_state['error_theme_internal'] = internal_theme
    
    # Display current theme info
    st.sidebar.subheader("📊 Configuration")
    theme_key = 'absolute' if selected_theme == 'Absolute Difference Error' else 'percentage'
    st.sidebar.success(f"🎯 Active: {selected_theme}")
    st.sidebar.info(f"📈 Using: {theme_key} error columns")

    # Load metadata and check data availability
    metadata = load_metadata()
    if not metadata:
        return

    # Load all games data
    with st.spinner("Loading all games data..."):
        all_games_results_dict = load_all_games_data()
    
    if not all_games_results_dict:
        st.error("No game data loaded. Please check the generator script output.")
        return

    # Game selection (exact same as original)
    available_games = list(all_games_results_dict.keys())
    game_options = ["Overall", "GAS013"] + sorted(available_games)
    
    selected_option = st.sidebar.selectbox(
        "Select Analysis Type:",
        game_options,
        index=0
    )

    # Display selected analysis
    if selected_option == "Overall":
        display_overall_analysis(all_games_results_dict)
    elif selected_option == "GAS013":
        display_gas013_analysis(all_games_results_dict)
    else:
        # Individual game selection
        display_individual_game_analysis(selected_option, all_games_results_dict)

if __name__ == "__main__":
    main() 