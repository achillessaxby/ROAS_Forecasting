"""
Underlying Table Tool
====================

Shows combined view of all cohorts across all games in a single table
with Cohort and Predicted On columns.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import storage
import io

# Configuration
BUCKET_NAME = "m1m2_notebook"
CACHE_PATH = "validation_results/"

st.title("Underlying Table")

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

@st.cache_data(ttl=3600)
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
        import json
        metadata = json.loads(metadata_content.decode('utf-8'))
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

@st.cache_data(ttl=3600)
def load_all_underlying_data(error_theme="Percentage Error", error_type="Incremental Error"):
    """Load and combine all game data into underlying table format."""
    metadata = load_metadata()
    if not metadata:
        return pd.DataFrame()
    
    processed_games = metadata.get('processed_games', [])
    all_data = []
    
    for game in processed_games:
        try:
            bucket = storage_client.bucket(BUCKET_NAME)
            blob_name = f"{CACHE_PATH}games/{game}_complete.parquet"
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                parquet_data = blob.download_as_bytes()
                df = pd.read_parquet(io.BytesIO(parquet_data))
                
                # Convert cohort to datetime if needed
                if 'cohort' in df.columns:
                    df['cohort'] = pd.to_datetime(df['cohort'])
                
                all_data.append(df)
        except Exception as e:
            st.warning(f"Could not load data for {game}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Transform to underlying table format - pivot to show M2_error, M3_error, etc.
    
    # Calculate "Predicted On" date from cohort + case months
    combined_df['predicted_on'] = combined_df.apply(
        lambda row: row['cohort'] + pd.DateOffset(months=int(row['case'])), 
        axis=1
    )
    
    # Determine which error column to use based on theme and type
    if error_type == "Cumulative Error":
        if error_theme == "Percentage Error":
            error_col_name = 'cum_error_percentage'
        else:
            error_col_name = 'cum_error_absolute'
    else:  # Incremental Error
        if error_theme == "Percentage Error":
            error_col_name = 'error_percentage'
        else:
            error_col_name = 'error_absolute'
    
    # Create month-wise error columns (M2_error, M3_error, etc.)
    # Group by cohort, game, and case to create pivot table
    pivot_data = []
    
    for (cohort, game, case), group in combined_df.groupby(['cohort', 'game', 'case']):
        row_data = {
            'Cohort': cohort,
            'Game ID': game,
            'Predicted On': cohort + pd.DateOffset(months=int(case))
        }
        
        # Add month-wise errors
        for _, row in group.iterrows():
            pred_month = int(row['prediction_month']) + 1  # Convert 0-based to 1-based (M1, M2, etc.)
            
            # Use the selected error column
            if error_col_name in row and pd.notna(row[error_col_name]):
                error_value = row[error_col_name]
                # Match M2 formatting: show percentages for both themes
                # - Percentage Error: value is already percentage
                # - Absolute Error: multiply by 100 to display as percentage
                display_val = error_value if error_theme == "Percentage Error" else (error_value * 100)
                row_data[f'M{pred_month}_error'] = f"{display_val:.2f}%"
        
        pivot_data.append(row_data)
    
    if not pivot_data:
        return pd.DataFrame()
    
    underlying_df = pd.DataFrame(pivot_data)
    
    # Fill missing error columns with "None"
    error_columns = [col for col in underlying_df.columns if col.endswith('_error')]
    for col in error_columns:
        underlying_df[col] = underlying_df[col].fillna('None')
    
    # Sort columns: Cohort, Predicted On, Game ID, then M2_error, M3_error, etc.
    base_columns = ['Cohort', 'Predicted On', 'Game ID']
    error_columns_sorted = sorted([col for col in underlying_df.columns if col.endswith('_error')], 
                                 key=lambda x: int(x.split('_')[0][1:]))  # Sort by month number
    
    final_column_order = base_columns + error_columns_sorted
    underlying_df = underlying_df[final_column_order]
    
    # Sort by Cohort, Game ID
    underlying_df = underlying_df.sort_values(['Cohort', 'Game ID'])
    
    return underlying_df

# Main interface
# st.header("Combined Cohort Data - All Games")

# Sidebar configuration
st.sidebar.header("Configuration")
error_theme = st.sidebar.selectbox(
    "Error Theme:",
    ["Absolute Error", "Percentage Error"],
    index=1  # Default to Percentage Error
)

error_type = st.sidebar.selectbox(
    "Error Type:",
    ["Incremental Error", "Cumulative Error"],
    index=0  # Default to Incremental Error
)

# Sidebar filters
st.sidebar.header("Filters")

# Load data
with st.spinner("Loading underlying table data..."):
    underlying_data = load_all_underlying_data(error_theme, error_type)

if underlying_data.empty:
    st.warning("No data available to display.")
else:
    # Sidebar game filter
    if 'Game ID' in underlying_data.columns:
        all_games = ['All'] + sorted(underlying_data['Game ID'].unique().tolist())
        selected_game = st.sidebar.selectbox("Select Game:", all_games)
    else:
        selected_game = 'All'
    
    # Sidebar cohort filter
    if 'Cohort' in underlying_data.columns:
        all_cohorts = ['All'] + sorted(underlying_data['Cohort'].dt.strftime('%Y-%m-%d').unique().tolist())
        selected_cohort = st.sidebar.selectbox("Select Cohort:", all_cohorts)
    else:
        selected_cohort = 'All'
    
    # Sidebar predicted on date filter
    if 'Predicted On' in underlying_data.columns:
        predicted_on_dates = ['All'] + sorted(underlying_data['Predicted On'].dt.strftime('%Y-%m-%d').unique().tolist())
        selected_predicted_on = st.sidebar.selectbox("Select Predicted On:", predicted_on_dates)
    else:
        selected_predicted_on = 'All'
    
    # st.header("Combined Cohort Data - All Games")
    
    # Display summary stats
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     st.metric("Total Records", len(underlying_data))
    
    # with col2:
    #     unique_games = underlying_data['Game ID'].nunique() if 'Game ID' in underlying_data.columns else 0
    #     st.metric("Unique Games", unique_games)
    
    # with col3:
    #     unique_cohorts = underlying_data['Cohort'].nunique() if 'Cohort' in underlying_data.columns else 0
    #     st.metric("Unique Cohorts", unique_cohorts)
    
    # with col4:
    #     date_range = ""
    #     if 'Cohort' in underlying_data.columns and not underlying_data['Cohort'].empty:
    #         min_date = underlying_data['Cohort'].min().strftime('%Y-%m')
    #         max_date = underlying_data['Cohort'].max().strftime('%Y-%m')
    #         date_range = f"{min_date} to {max_date}"
    #     st.metric("Date Range", date_range)
    
    # Apply filters
    filtered_data = underlying_data.copy()
    
    if selected_game != 'All' and 'Game ID' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Game ID'] == selected_game]
    
    if selected_cohort != 'All' and 'Cohort' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Cohort'].dt.strftime('%Y-%m-%d') == selected_cohort]
    
    if selected_predicted_on != 'All' and 'Predicted On' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Predicted On'].dt.strftime('%Y-%m-%d') == selected_predicted_on]
    
    # Display filtered data
    analysis_title = "Cumulative Error Analysis" if error_type == "Cumulative Error" else "Incremental Error Analysis"
    st.subheader(analysis_title)
    
    # Format display
    display_data = filtered_data.copy()
    
    # Format datetime columns
    if 'Cohort' in display_data.columns:
        display_data['Cohort'] = display_data['Cohort'].dt.strftime('%Y-%m-%d')
    
    if 'Predicted On' in display_data.columns:
        display_data['Predicted On'] = display_data['Predicted On'].dt.strftime('%Y-%m-%d')
    
    # Group by cohort to show cohort only when it changes
    display_data_grouped = display_data.copy()
    
    # Create a version where cohort is shown only when it changes
    prev_cohort = None
    for idx, row in display_data_grouped.iterrows():
        current_cohort = row['Cohort']
        if current_cohort == prev_cohort:
            display_data_grouped.at[idx, 'Cohort'] = ''  # Empty string instead of repeating
        prev_cohort = current_cohort
    
    # Replace 'None' with empty strings for cleaner appearance
    display_data_grouped = display_data_grouped.replace('None', '')
    
    # Display the table with month-wise errors (hide index)
    st.dataframe(
        display_data_grouped, 
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # # Download option
    # if not filtered_data.empty:
    #     csv = filtered_data.to_csv(index=False)
    #     st.download_button(
    #         label="Download as CSV",
    #         data=csv,
    #         file_name=f"underlying_table_monthwise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    #         mime="text/csv"
    #     )