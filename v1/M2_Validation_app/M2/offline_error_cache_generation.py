"""
M2 Validation Data Generator for Offline Files - Enhanced Multi-Model Version
===========================================================================

This script wraps the existing M2 validation pipeline to pre-compute 
all validation results for all games and themes, then saves everything 
offline for fast, consistent retrieval by the Streamlit app.

Updated to work with the enhanced multi-model system where different 
model types (XGBoost, LightGBM, CatBoost, Random Forest, Ridge) are 
used and only the best performing model is saved for each game.

Run this offline to generate the complete validation dataset.
"""

import pandas as pd
import numpy as np
import os
import cloudpickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob

import warnings
warnings.filterwarnings("ignore")

# Configuration - Adapted for offline files
MODEL_PATH = "models/"
DATA_PATH = "Latest Data/Segregated/"
OUTPUT_PATH = "validation_results/"
CONFIG_PATH = "Latest Data/Segregated/config.json"

# Error themes to compute - EXACT match with original
# Original uses display names -> internal names mapping:
# "Absolute Difference Error" -> "Absolute Error"  
# "Absolute Percentage Error" -> "Percentage Error"
ERROR_THEMES = {
    "absolute": "Absolute Error",
    "percentage": "Percentage Error"  
}

print("🚀 M2 Validation Data Generator Starting...")
print(f"📁 Output Path: {OUTPUT_PATH}")
print("🔧 Enhanced Multi-Model System Support")

# Create output directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "games"), exist_ok=True)

# ========================================================================================
# ENHANCED FUNCTIONS FOR MULTI-MODEL SYSTEM
# ========================================================================================

def load_config():
    """Load the configuration file to find best models for each game"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return {}

def get_best_model_info(game_id, model_type="M2"):
    """Get the best model type and filename for a given game"""
    config = load_config()
    config_key = f"{model_type}_{game_id}"
    
    if config_key in config:
        entry = config[config_key]
        best_model_type = entry.get('best_model', 'xgboost')  # fallback to xgboost
        best_rmse = entry.get('best_rmse', float('inf'))
        
        # Construct the expected filename
        model_filename = f"{game_id}_{best_model_type}_{model_type}.pkl"
        model_path = os.path.join(MODEL_PATH, model_filename)
        
        return {
            'model_type': best_model_type,
            'filename': model_filename,
            'path': model_path,
            'rmse': best_rmse,
            'config_entry': entry
        }
    
    return None

def discover_available_games():
    """Discover games that have trained models using the new naming pattern"""
    # Get all M2 model files
    model_pattern = os.path.join(MODEL_PATH, "*_M2.pkl")
    model_files = glob.glob(model_pattern)
    
    games = set()
    for model_file in model_files:
        filename = os.path.basename(model_file)
        # Extract game ID from filename like "G001_catboost_M2.pkl"
        parts = filename.replace('.pkl', '').split('_')
        if len(parts) >= 3 and parts[-1] == 'M2':
            game_id = parts[0]
            games.add(game_id)
    
    return sorted(list(games))

# ========================================================================================
# ADAPTED FUNCTIONS FROM ORIGINAL M2_Validation_tool_b.py FOR OFFLINE FILES
# ========================================================================================

def file_exists(file_path):
    return os.path.exists(file_path)

def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {str(e)}")
        return None

def list_dir_files(directory, pattern):
    try:
        return glob.glob(os.path.join(directory, pattern))
    except Exception as e:
        print(f"❌ Error listing files in {directory}: {str(e)}")
        return []

def load_models(game):
    """Load the best model for a given game using the enhanced naming system"""
    model_info = get_best_model_info(game, "M2")
    
    if model_info and file_exists(model_info['path']):
        try:
            with open(model_info['path'], 'rb') as f:
                model = cloudpickle.load(f)
            print(f"  ✅ Loaded {model_info['model_type']} model for {game} (RMSE: {model_info['rmse']:.4f})")
            return model
        except Exception as e:
            print(f"❌ Error loading model {model_info['path']}: {str(e)}")
    else:
        print(f"❌ No model found for {game}")
    
    return None

def calculate_current_month(cohort, elapsed_months):
    return (cohort + pd.DateOffset(months=int(elapsed_months))).month

def calculate_error(actual, predicted, theme="Absolute Error"):
    """Calculate error - EXACT copy from original."""
    if pd.isna(actual) or pd.isna(predicted):
        return np.nan
    
    if theme == "Absolute Error":
        return abs(predicted - actual)
    elif theme == "Percentage Error":
        if actual == 0:
            return np.nan
        return abs(predicted - actual) * 100 / abs(actual)
    else:
        return abs(predicted - actual)

def calculate_cumulative_error(cum_actual, cum_predicted, theme="Absolute Error"):
    """Calculate cumulative error - EXACT copy from original."""
    if pd.isna(cum_actual) or pd.isna(cum_predicted):
        return np.nan
    
    if theme == "Absolute Error":
        return abs(cum_predicted - cum_actual)
    elif theme == "Percentage Error":
        if cum_actual == 0:
            return np.nan
        return abs(cum_predicted - cum_actual) * 100 / abs(cum_actual)
    else:
        return abs(cum_predicted - cum_actual)

def preprocess_data(game):
    try:
        data_file = os.path.join(DATA_PATH, f"{game}_Dataset_12_2024.csv")
        if not file_exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return None
        
        df = read_csv_file(data_file)
        if df is None:
            return None
        
        # Basic preprocessing
        df['elapsed_months'].fillna(1, inplace=True)
        df = df[(df['total_cac'] != 0) & (df['total_cac'].notna())]
        df['elapsed_months'] -= 1
        df = df[df['elapsed_months'] >= 0]
        
        # Handle CAC and spend
        df['total_cac'] = df.apply(
            lambda row: row['client_actual_spend'] 
            if (pd.isnull(row['total_cac']) or row['total_cac'] == 0) and not pd.isnull(row['client_actual_spend']) 
            else row['total_cac'], axis=1)
        df['client_actual_spend'] = df.apply(
            lambda row: row['total_cac'] 
            if (pd.isnull(row['client_actual_spend']) or row['client_actual_spend'] == 0) and not pd.isnull(row['total_cac']) 
            else row['client_actual_spend'], axis=1)
        
        # Filter years
        df = df[~df['cohort'].str.endswith(('2021', '2022', '2020', '2019', '2018'))]
        df = df[~df['cohort'].str.startswith(('2021', '2022', '2020', '2019', '2018'))]
        
        # Date handling
        if game == "G090":
            df['cohort'] = df['cohort'].str.replace('-', '/')
            df['transaction_period'] = df['transaction_period'].str.replace('-', '/')
        
        df['cohort'] = pd.to_datetime(df['cohort'], format='%d/%m/%Y')
        df['transaction_period'] = pd.to_datetime(df['transaction_period'], format='%d/%m/%Y', errors='coerce')
        
        # Filter out current month data
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        initial_rows = len(df)
        df = df[~((df['transaction_period'].dt.month == current_month) & 
                 (df['transaction_period'].dt.year == current_year))]
        filtered_rows = len(df)
        
        if initial_rows > filtered_rows:
            print(f"  🔍 Filtered out {initial_rows - filtered_rows} rows from current month ({current_month:02d}-{current_year}) for game {game}")
        
        # Add month columns
        df['cohort_month'] = df['cohort'].dt.month
        df['current_month'] = df.apply(lambda row: calculate_current_month(row['cohort'], row['elapsed_months']), axis=1)
        df['cohort_month'] = df['cohort_month'].astype('category')
        df['current_month'] = df['current_month'].astype('category')
        
        # Add lag features
        for lag in range(1, 4):
            df[f'ltv_lag_{lag}'] = df.groupby('cohort')['transaction_period_ltv_cac'].shift(lag)
        
        return df
    except Exception as e:
        print(f"❌ Error processing data for {game}: {str(e)}")
        return None

def has_enough_cohorts(df, min_cohorts=3):
    if df is None:
        return False
    return len(df['cohort'].unique()) >= min_cohorts

def run_predictions(game, df, model, theme="Absolute Error"):
    if df is None or model is None or df.empty:
        return pd.DataFrame()
    
    latest_cohorts = sorted(df['cohort'].unique())[-12:]
    results = []
    
    for cohort_dt_obj in latest_cohorts:
        cohort_data = df[df['cohort'] == cohort_dt_obj].copy()
        if cohort_data.empty:
            continue
        max_elapsed_month = int(cohort_data['elapsed_months'].max())
        
        for start_month in range(0, max_elapsed_month + 1):
            case_num = start_month + 1
            actual_data = cohort_data[cohort_data['elapsed_months'] <= start_month].copy()
            
            if actual_data.empty:
                continue
            
            lag_values = {}
            for lag in range(1, 4):
                lag_hist_month = start_month - lag
                if lag_hist_month >= 0:
                    lag_val_series = cohort_data[cohort_data['elapsed_months'] == lag_hist_month]['transaction_period_ltv_cac']
                    lag_values[f'ltv_lag_{lag}'] = lag_val_series.iloc[0] if not lag_val_series.empty else 0.0
                else:
                    lag_values[f'ltv_lag_{lag}'] = 0.0
            
            for pred_elapsed_month in range(start_month + 1, max_elapsed_month + 1):
                static_features_series = actual_data.iloc[-1].copy()
                
                # Extract feature names from model
                feature_names = None
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
                    feature_names = model.steps[-1][1].feature_names_in_
                elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                    feature_names = model.get_booster().feature_names
                
                pred_features_df = pd.DataFrame([static_features_series])
                pred_features_df['elapsed_months'] = pred_elapsed_month
                
                if 'current_month' in pred_features_df.columns:
                    pred_features_df['current_month'] = calculate_current_month(cohort_dt_obj, pred_elapsed_month)

                for lag in range(1, 4):
                    lag_col = f'ltv_lag_{lag}'
                    source_elapsed_month_for_lag = pred_elapsed_month - lag
                    
                    if source_elapsed_month_for_lag <= start_month and source_elapsed_month_for_lag >= 0:
                        actual_lag_val_series = cohort_data[cohort_data['elapsed_months'] == source_elapsed_month_for_lag]['transaction_period_ltv_cac']
                        pred_features_df[lag_col] = actual_lag_val_series.iloc[0] if not actual_lag_val_series.empty else 0.0
                    else:
                        pred_features_df[lag_col] = lag_values.get(lag_col, 0.0)
                
                try:
                    if feature_names is not None:
                        for col in feature_names:
                            if col not in pred_features_df.columns:
                                pred_features_df[col] = 0.0 
                        pred_features_for_model = pred_features_df[feature_names]
                    else:
                        pred_features_for_model = pred_features_df

                    prediction = float(model.predict(pred_features_for_model)[0])
                except Exception as e:
                    print(f"    ⚠️ Prediction error for {game}: {str(e)}")
                    # Fix: record a missing prediction (excluded from error metrics) instead of a
                    # silent 0.0 that was scored as a real forecast
                    prediction = np.nan

                actual_val_series = cohort_data[cohort_data['elapsed_months'] == pred_elapsed_month]['transaction_period_ltv_cac']
                actual_val = float(actual_val_series.iloc[0]) if not actual_val_series.empty else np.nan
                
                error = calculate_error(actual_val, prediction, theme)

                results.append({
                    'game': game,
                    'cohort': cohort_dt_obj,
                    'cohort_str': cohort_dt_obj.strftime('%Y-%m-%d'),
                    'case': case_num,
                    'prediction_month': pred_elapsed_month,
                    'actual_value': actual_val,
                    'predicted_value': prediction,
                    'error': error 
                })
                
                lag_values['ltv_lag_3'] = lag_values.get('ltv_lag_2', 0.0)
                lag_values['ltv_lag_2'] = lag_values.get('ltv_lag_1', 0.0)
                lag_values['ltv_lag_1'] = prediction

    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cohort'] = pd.to_datetime(results_df['cohort'])
    return results_df

def calculate_cumulative(results_df, theme="Absolute Error"):
    if results_df is None or results_df.empty:
        return results_df
    
    results_df = results_df.sort_values(by=['cohort_str', 'case', 'prediction_month'])
    
    def calculate_cumulative_for_group(group):
        game = group['game'].iloc[0]
        cohort_dt = group['cohort'].iloc[0]
        
        try:
            # Get M1 actual value
            df = preprocess_data(game)
            if df is not None:
                cohort_data = df[df['cohort'] == cohort_dt].copy()
                if not cohort_data.empty:
                    m1_actual_series = cohort_data[cohort_data['elapsed_months'] == 0]['transaction_period_ltv_cac']
                    m1_actual = float(m1_actual_series.iloc[0]) if not m1_actual_series.empty else 0.0
                    
                    group = group.copy()
                    group['cum_actual'] = m1_actual + group['actual_value'].cumsum()
                    group['cum_predicted'] = m1_actual + group['predicted_value'].cumsum()
                    
                    group['cum_error'] = group.apply(
                        lambda row: calculate_cumulative_error(row['cum_actual'], row['cum_predicted'], theme), 
                        axis=1
                    )
                    return group
        except Exception as e:
            pass
        
        # Fallback
        group = group.copy()
        group['cum_actual'] = group['actual_value'].cumsum()
        group['cum_predicted'] = group['predicted_value'].cumsum()
        group['cum_error'] = group.apply(
            lambda row: calculate_cumulative_error(row['cum_actual'], row['cum_predicted'], theme), 
            axis=1
        )
        return group
    
    results_df = results_df.groupby(['cohort_str', 'case']).apply(calculate_cumulative_for_group).reset_index(drop=True)
    return results_df

def get_eligible_games():
    """Get games that have both trained models and data files"""
    # First, discover games with trained models
    available_games = discover_available_games()
    print(f"🔍 Found {len(available_games)} games with trained models")
    
    eligible_games = []
    for game in available_games:
        # Check if data file exists
        data_file = os.path.join(DATA_PATH, f"{game}_Dataset_12_2024.csv")
        if file_exists(data_file):
            # Check if model can be loaded
            model_info = get_best_model_info(game, "M2")
            if model_info and file_exists(model_info['path']):
                df = preprocess_data(game)
                if has_enough_cohorts(df):
                    eligible_games.append(game)
                    print(f"  ✅ {game} - {model_info['model_type']} model (RMSE: {model_info['rmse']:.4f})")
                else:
                    print(f"  ❌ {game} - insufficient cohorts")
            else:
                print(f"  ❌ {game} - model not found")
        else:
            print(f"  ❌ {game} - data file not found")
    
    return eligible_games

# ========================================================================================
# WRAPPER FUNCTIONS TO GENERATE AND SAVE DATA
# ========================================================================================

def save_parquet_file(df, file_path):
    try:
        df.to_parquet(file_path, index=False)
        print(f"✅ Saved: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving {file_path}: {str(e)}")
        return False

def save_json_file(data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Saved: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving {file_path}: {str(e)}")
        return False

def generate_game_data_for_all_themes(game):
    """Generate data for a single game with all themes."""
    print(f"\n🎮 Processing Game: {game}")
    
    # Load model and data using original functions
    print(f"  📦 Loading model...")
    model = load_models(game)
    if model is None:
        print(f"  ❌ Failed to load model for {game}")
        return None
    
    print(f"  📊 Loading data...")
    df = preprocess_data(game)
    if not has_enough_cohorts(df):
        print(f"  ❌ Insufficient cohorts for {game}")
        return None
    
    # Generate predictions for both themes
    all_theme_data = {}
    
    for theme_key, theme_name in ERROR_THEMES.items():
        print(f"  🎯 Calculating theme: {theme_name}")
        
        # Use original pipeline exactly
        results = run_predictions(game, df, model, theme_name)
        if results is not None and not results.empty:
            results = calculate_cumulative(results, theme_name)
            all_theme_data[theme_key] = results
        else:
            print(f"    ❌ No results for {game} - {theme_name}")
            return None
    
    # Combine both themes into one DataFrame
    if len(all_theme_data) == 2:
        # Start with base data (without error columns)
        base_data = all_theme_data['absolute'][['game', 'cohort', 'cohort_str', 'case', 
                                               'prediction_month', 'actual_value', 'predicted_value']].copy()
        
        # Add theme-specific error columns with validation
        for theme_key, theme_df in all_theme_data.items():
            base_data[f'error_{theme_key}'] = theme_df['error']
            base_data[f'cum_error_{theme_key}'] = theme_df['cum_error']
            base_data[f'cum_actual_{theme_key}'] = theme_df['cum_actual']
            base_data[f'cum_predicted_{theme_key}'] = theme_df['cum_predicted']
        
        # Validate all required columns are present
        required_columns = [
            'game', 'cohort', 'cohort_str', 'case', 'prediction_month', 
            'actual_value', 'predicted_value',
            'error_absolute', 'error_percentage',
            'cum_error_absolute', 'cum_error_percentage',
            'cum_actual_absolute', 'cum_actual_percentage',
            'cum_predicted_absolute', 'cum_predicted_percentage'
        ]
        
        missing_cols = [col for col in required_columns if col not in base_data.columns]
        if missing_cols:
            print(f"  ⚠️  Missing columns for {game}: {missing_cols}")
            return None
        
        print(f"  ✅ Completed {game}: {len(base_data)} records with all themes")
        return base_data
    
    return None

def main():
    """Main function to generate all validation data."""
    print("🚀 Starting Enhanced M2 Validation Data Generation")
    print("=" * 60)
    
    # Load and display configuration summary
    config = load_config()
    m2_configs = {k: v for k, v in config.items() if k.startswith('M2_')}
    print(f"📊 Configuration loaded: {len(m2_configs)} M2 game configurations found")
    
    # Display model type distribution
    model_type_counts = {}
    for entry in m2_configs.values():
        best_model = entry.get('best_model', 'unknown')
        model_type_counts[best_model] = model_type_counts.get(best_model, 0) + 1
    
    print("🏆 Best model distribution:")
    for model_type, count in sorted(model_type_counts.items()):
        print(f"  {model_type}: {count} games")
    
    # Get eligible games using enhanced discovery
    eligible_games = get_eligible_games()
    if not eligible_games:
        print("❌ No eligible games found. Exiting.")
        return
    
    print(f"\n✅ Found {len(eligible_games)} eligible games for processing")
    
    # Generate data for all games
    all_games_data = {}
    
    for i, game in enumerate(eligible_games, 1):
        print(f"\n[{i}/{len(eligible_games)}] Processing {game}")
        
        # Show model info
        model_info = get_best_model_info(game, "M2")
        if model_info:
            print(f"  🎯 Using {model_info['model_type']} model (RMSE: {model_info['rmse']:.4f})")
        
        game_results = generate_game_data_for_all_themes(game)
        if game_results is not None:
            all_games_data[game] = game_results
            
            # Save individual game data
            game_file = os.path.join(OUTPUT_PATH, "games", f"{game}_complete.parquet")
            if save_parquet_file(game_results, game_file):
                print(f"  💾 Saved: {game_file}")
        else:
            print(f"  ❌ Failed to process {game}")
    
    if not all_games_data:
        print("❌ No games were successfully processed. Exiting.")
        return
    
    # Generate enhanced metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_games": len(all_games_data),
        "eligible_games": eligible_games,
        "processed_games": list(all_games_data.keys()),
        "themes": list(ERROR_THEMES.keys()),
        "cache_version": "v4.0_enhanced_multi_model",
        "model_distribution": model_type_counts,
        "enhanced_features": {
            "multi_model_support": True,
            "best_model_selection": True,
            "dynamic_model_loading": True,
            "configuration_driven": True
        },
        "data_structure": {
            "games": os.path.join(OUTPUT_PATH, "games/"),
            "note": "Each game file contains all themes with columns: error_absolute, error_percentage, cum_error_absolute, cum_error_percentage, cum_actual_*, cum_predicted_*",
            "model_info": "Each game uses its best performing model as determined by training comparison",
            "required_columns": [
                "game", "cohort", "cohort_str", "case", "prediction_month", 
                "actual_value", "predicted_value",
                "error_absolute", "error_percentage",
                "cum_error_absolute", "cum_error_percentage", 
                "cum_actual_absolute", "cum_actual_percentage",
                "cum_predicted_absolute", "cum_predicted_percentage"
            ]
        },
        "model_details": {
            game: {
                "model_type": get_best_model_info(game, "M2")['model_type'] if get_best_model_info(game, "M2") else "unknown",
                "model_rmse": get_best_model_info(game, "M2")['rmse'] if get_best_model_info(game, "M2") else "unknown"
            } for game in all_games_data.keys()
        }
    }
    
    metadata_file = os.path.join(OUTPUT_PATH, "metadata.json")
    save_json_file(metadata, metadata_file)
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED DATA GENERATION COMPLETE!")
    print(f"✅ Processed: {len(all_games_data)} games")
    print(f"✅ Model types used: {list(model_type_counts.keys())}")
    print(f"✅ Themes: {list(ERROR_THEMES.keys())}")
    print(f"✅ Location: {OUTPUT_PATH}")
    print("✅ Each game uses its best performing model from training")
    print("✅ Each game file contains all error calculations for both themes")
    print("🚀 Ready to deploy enhanced M2 validation!")
    print("=" * 60)

if __name__ == "__main__":
    main()