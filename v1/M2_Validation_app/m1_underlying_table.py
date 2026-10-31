"""
M1 Underlying Table Tool
====================

Shows a combined view of M1 validation errors across all games in a single table,
focusing on Case 1 errors derived from the M1 validation logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import the M1 validation function directly
from m1_validation_helper import run_m1_validation_all_cases, list_available_games

st.title("M1 Underlying Table")

@st.cache_data(ttl=3600)
def load_all_m1_underlying_data(error_theme="Percentage Error", error_type="Incremental Error"):
    """Load and combine all game data for the M1 underlying table with multiple cases."""
    available_games = list_available_games()
    all_data = []

    # Determine which error calculation method to use based on the theme
    error_method = "Absolute Percentage Error" if error_theme == "Percentage Error" else "Absolute Difference Error"
    
    # Determine which error type to extract (transactional vs. cumulative)
    error_suffix = "trans" if error_type == "Incremental Error" else "cum"

    for game in available_games:
        # Use the new all_cases function to get data for all cases (up to 12)
        summary_df, _, _, _ = run_m1_validation_all_cases(game, "Extrapolation", error_method, max_cases=12)
        
        if summary_df is not None and not summary_df.empty:
            all_data.append(summary_df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Transform to the underlying table format (matching M2 structure) ---
    
    # Rename columns for consistency
    combined_df = combined_df.rename(columns={"game_id": "Game ID", "cohort": "Cohort"})
    
    # Convert cohort to datetime
    combined_df['Cohort'] = pd.to_datetime(combined_df['Cohort'])
    
    # Convert predicted_on to datetime and rename
    combined_df['Predicted On'] = pd.to_datetime(combined_df['predicted_on'])

    # Create the final DataFrame structure - one row per (Cohort, Predicted On, Game ID) combination
    final_rows = []
    for _, row in combined_df.iterrows():
        new_row = {
            'Cohort': row['Cohort'],
            'Predicted On': row['Predicted On'],
            'Game ID': row['Game ID']
        }
        
        # Extract M1 to M12 errors
        for i in range(1, 13):
            error_col_name = f'm{i}_error_{error_suffix}'
            if error_col_name in row and pd.notna(row[error_col_name]):
                error_value = row[error_col_name]
                # Format as percentage if needed
                if error_theme == "Percentage Error":
                    new_row[f'M{i}_error'] = f"{error_value:.2f}%"
                else:
                    new_row[f'M{i}_error'] = f"{error_value:.4f}"
            else:
                new_row[f'M{i}_error'] = 'None'
        
        final_rows.append(new_row)

    if not final_rows:
        return pd.DataFrame()

    underlying_df = pd.DataFrame(final_rows)

    # Sort columns: Cohort, Predicted On, Game ID, then M1_error, M2_error, etc.
    base_columns = ['Cohort', 'Predicted On', 'Game ID']
    error_columns_sorted = sorted([col for col in underlying_df.columns if col.endswith('_error')], 
                                 key=lambda x: int(x.split('_')[0][1:]))  # Sort by month number
    
    final_column_order = base_columns + error_columns_sorted
    underlying_df = underlying_df[final_column_order]
    
    # Sort by Cohort, Predicted On, then Game ID (matching M2 structure)
    underlying_df = underlying_df.sort_values(['Cohort', 'Predicted On', 'Game ID'])
    
    return underlying_df

# --- Streamlit UI ---

st.sidebar.header("Configuration")
error_theme = st.sidebar.selectbox(
    "Error Theme:",
    ["Percentage Error", "Absolute Error"],
    index=0
)

error_type = st.sidebar.selectbox(
    "Error Type:",
    ["Incremental Error", "Cumulative Error"],
    index=0
)

st.sidebar.header("Filters")

with st.spinner("Loading M1 underlying table data..."):
    underlying_data = load_all_m1_underlying_data(error_theme, error_type)

if underlying_data.empty:
    st.warning("No data available to display.")
else:
    # Sidebar game filter
    all_games = ['All'] + sorted(underlying_data['Game ID'].unique().tolist())
    selected_game = st.sidebar.selectbox("Select Game:", all_games)
    
    # Sidebar cohort filter
    all_cohorts = ['All'] + sorted(underlying_data['Cohort'].dt.strftime('%Y-%m-%d').unique().tolist())
    selected_cohort = st.sidebar.selectbox("Select Cohort:", all_cohorts)
    
    # Sidebar predicted on date filter
    predicted_on_dates = ['All'] + sorted(underlying_data['Predicted On'].dt.strftime('%Y-%m-%d').unique().tolist())
    selected_predicted_on = st.sidebar.selectbox("Select Predicted On:", predicted_on_dates)

    # Apply filters
    filtered_data = underlying_data.copy()
    if selected_game != 'All':
        filtered_data = filtered_data[filtered_data['Game ID'] == selected_game]
    if selected_cohort != 'All':
        filtered_data = filtered_data[filtered_data['Cohort'].dt.strftime('%Y-%m-%d') == selected_cohort]
    if selected_predicted_on != 'All':
        filtered_data = filtered_data[filtered_data['Predicted On'].dt.strftime('%Y-%m-%d') == selected_predicted_on]

    # Display the table
    st.subheader(f"{error_type} Analysis")
    
    display_data = filtered_data.copy()
    display_data['Cohort'] = display_data['Cohort'].dt.strftime('%Y-%m-%d')
    display_data['Predicted On'] = display_data['Predicted On'].dt.strftime('%Y-%m-%d')

    display_data = display_data.replace('None', '')
    
    st.dataframe(
        display_data,
        use_container_width=True,
        height=600,
        hide_index=True
    )
