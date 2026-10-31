import pandas as pd
import numpy as np
from datetime import datetime

def calculate_holdout_split(df):
    """Calculate training/holdout split based on elapsed_months with 6-month holdout strategy"""
    max_elapsed = df['elapsed_months'].max()
    
    # If we have more than 8 months (i.e., max_elapsed >= 8), reserve last 6 months for holdout
    # Otherwise, use all available data for training
    if max_elapsed >= 8:
        training_cutoff = max_elapsed - 6  # Reserve last 6 months
        training_data = df[df['elapsed_months'] <= training_cutoff].copy()
        holdout_data = df[df['elapsed_months'] > training_cutoff].copy()
    else:
        # Not enough data for 6-month holdout, use all data for training
        training_data = df.copy()
        holdout_data = pd.DataFrame()  # Empty holdout
    
    return training_data, holdout_data, max_elapsed

def calculate_current_month(row):
    """Calculate current month based on cohort and elapsed months"""
    cohort_start = row['cohort']
    elapsed_months = row['elapsed_months']
    current_month = (cohort_start + pd.DateOffset(months=int(elapsed_months))).month
    return current_month

def preprocess_m1(game, use_holdout=True):
    """Preprocess data for M1 model with optional holdout strategy"""
    df = pd.read_csv(f'Latest Data/Segregated/{game}_Dataset_12_2024.csv')
    df['elapsed_months'] = df['elapsed_months'] - 1
    df = df[~((df['client_actual_spend'].fillna(0) == 0) & (df['total_cac'].fillna(0) == 0))]
    
    # Handle total_cac and client_actual_spend
    df['total_cac'] = df.apply(
        lambda row: row['client_actual_spend'] 
        if (pd.isnull(row['total_cac']) or row['total_cac'] == 0) and not pd.isnull(row['client_actual_spend']) 
        else row['total_cac'], 
        axis=1
    )
    df['client_actual_spend'] = df.apply(
        lambda row: row['total_cac'] 
        if (pd.isnull(row['client_actual_spend']) or row['client_actual_spend'] == 0) and not pd.isnull(row['total_cac']) 
        else row['client_actual_spend'], 
        axis=1
    )
    
    # Get MC columns
    cl1 = df.loc[:, df.columns.str.startswith('MC_')].columns.tolist()
    
    # Filter out old cohorts
    df = df[~df['cohort'].str.endswith(('2021', '2022','2020','2019','2018'))]
    df = df[~df['cohort'].str.startswith(('2021','2022','2020','2019','2018'))]
    
    # Select relevant columns
    columns_to_keep = ['game_id', 'total_cac', 'client_actual_spend', 'cohort', 'elapsed_months', 
                       'transaction_period_ltv_cac', 'transaction_period'] + cl1
    df = df[columns_to_keep]
    df = df.reset_index(drop=True)
    df = df[df['game_id'] == game]
    
    # Convert cohort to datetime
    df['cohort'] = pd.to_datetime(df['cohort'], format='%d/%m/%Y')
    df['cohort_month'] = df['cohort'].dt.month

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("missing")
    
    # Calculate current month
    df['current_month'] = df.apply(calculate_current_month, axis=1)
    
    # Default features for M1 (do not include elapsed_months here)
    feature_columns = ['total_cac', 'cohort_month', 'current_month'] + cl1
    
    # Create lag features
    for lag in range(1, 4):
        df[f'ltv_lag_{lag}'] = df.groupby('cohort')['transaction_period_ltv_cac'].shift(lag)
    
    # Apply holdout split if requested
    if use_holdout:
        training_df, holdout_df, max_elapsed = calculate_holdout_split(df)
        return training_df, feature_columns, holdout_df, max_elapsed
    else:
        return df, feature_columns

def preprocess_m2(game, use_holdout=True):
    """Preprocess data for M2 model with optional holdout strategy"""
    df = pd.read_csv(f'Latest Data/Segregated/{game}_Dataset_12_2024.csv')
    df['elapsed_months'] = df['elapsed_months'].fillna(1)
    df = df[(df['total_cac'] != 0) & (df['total_cac'].notna())]
    df['elapsed_months'] -= 1
    df = df[df['elapsed_months'] >= 0]
    
    # Handle total_cac and client_actual_spend
    df['total_cac'] = df.apply(
        lambda row: row['client_actual_spend'] 
        if (pd.isnull(row['total_cac']) or row['total_cac'] == 0) and not pd.isnull(row['client_actual_spend'])
        else row['total_cac'],
        axis=1
    )
    df['client_actual_spend'] = df.apply(
        lambda row: row['total_cac']
        if (pd.isnull(row['client_actual_spend']) or row['client_actual_spend'] == 0) and not pd.isnull(row['total_cac'])
        else row['client_actual_spend'],
        axis=1
    )
    
    # Filter out old cohorts
    df = df[~df['cohort'].str.endswith(('2021', '2022','2020','2019','2018'))]
    df = df[~df['cohort'].str.startswith(('2021','2022','2020','2019','2018'))]
    
    # Convert cohort to datetime
    df['cohort'] = pd.to_datetime(df['cohort'], format='%d/%m/%Y')
    df['cohort_month'] = df['cohort'].dt.month
    df['current_month'] = df.apply(calculate_current_month, axis=1)

    # First day of the current month as a Timestamp (fix C1: comparing dates to the string
    # "1/10/2026" made pandas read it month-first as 10 January and drop recent data)
    current_month_start = pd.Timestamp.today().normalize().replace(day=1)
    
    # Filter data before current month
    filtered_df = df[
        pd.to_datetime(df['transaction_period'], format='%d/%m/%Y', errors='coerce') < current_month_start
    ].copy()

    # Identify exclude columns
    exclude_columns = [
        'transaction_period_ltv_cac',  # target variable
        'game_id',  # identifier
        'cohort',  # used for grouping
        'transaction_period',  # used for filtering
        'elapsed_months',  # added separately
        'client_actual_spend'  # used as total_cac
    ]
    additional_excludes = ['interest', 'adjustment', 'adjustment_interest', 'client_id', 'elapsed_months.1']
    for exc in additional_excludes:
        if exc in df.columns:
            exclude_columns.append(exc)

    # Exclude date columns
    for col in df.columns:
        if 'date' in col.lower():
            exclude_columns.append(col)

    # Base features
    mc_cols = [col for col in df.columns if col.startswith('MC_')]
    day_cols = [col for col in df.columns if col.startswith('day_')]
    base_features = mc_cols + day_cols + ['total_cac', 'm0_ltv_cac', 'cohort_month', 'current_month']

    # Remaining columns
    remaining_cols = [col for col in df.columns if col not in exclude_columns and col not in base_features]

    # Add m1_features
    m1_df = df[df['elapsed_months'] == 1][['cohort'] + remaining_cols].copy()
    if not m1_df.empty:
        m1_df = m1_df.rename(columns={col: f'm1_feature_{col}' for col in remaining_cols})
        m1_df.set_index('cohort', inplace=True)
        for new_col in m1_df.columns:
            df[new_col] = df['cohort'].map(m1_df[new_col])
        df[m1_df.columns] = df[m1_df.columns].fillna(0)

    # Fill missing numeric values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Feature columns
    feature_columns = base_features + [f'm1_feature_{col}' for col in remaining_cols if f'm1_feature_{col}' in df.columns]

    # Create lag features
    for lag in range(1, 4):
        df[f'ltv_lag_{lag}'] = df.groupby('cohort')['transaction_period_ltv_cac'].shift(lag)
    
    # Apply current month filtering
    dft = df[
        pd.to_datetime(df['transaction_period'], format='%d/%m/%Y', errors='coerce') < current_month_start
    ].copy()
    
    # Apply holdout split if requested
    if use_holdout:
        training_dft, holdout_dft, max_elapsed = calculate_holdout_split(dft)
        return training_dft, feature_columns, df, holdout_dft, max_elapsed
    else:
        return dft, feature_columns, df 