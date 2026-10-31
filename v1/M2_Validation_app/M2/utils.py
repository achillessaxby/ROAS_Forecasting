import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import json

def auto_select_features(model_type, game_id, df, feature_columns):
    """Automatically select the best features using XGBoost feature importance"""
    if model_type == "M1":
        train_features = [col for col in feature_columns if col in df.columns]
        if 'elapsed_months' in df.columns and 'elapsed_months' not in train_features:
            train_features.append('elapsed_months')
        
        X = df[train_features].drop(df.index[-1])
        y = df['transaction_period_ltv_cac'].drop(df.index[-1])
    else:  # M2
        additional = ['ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3', 'elapsed_months']
        train_features = [col for col in feature_columns + additional if col in df.columns]
        X = df[train_features]
        y = df['transaction_period_ltv_cac']
    
    # Handle numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 0:
        X[numeric_features] = X[numeric_features].fillna(0)
        vt = VarianceThreshold(threshold=0.01)
        X_numeric = pd.DataFrame(vt.fit_transform(X[numeric_features]), columns=numeric_features[vt.get_support()], index=X.index)
        X = pd.concat([X_numeric, X.drop(numeric_features, axis=1)], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train base model for feature importance
    base_model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Get importances
    importances = base_model.feature_importances_
    feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    # Select features with importance > 0
    selected = feature_imp[feature_imp['importance'] > 0]['feature'].tolist()
    
    # Always include key features
    key_features = ['elapsed_months', 'total_cac', 'ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3']
    if model_type == "M2":
        key_features += ['m0_ltv_cac', 'cohort_month', 'current_month']
    else:
        key_features += ['cohort_month', 'current_month']
    
    for kf in key_features:
        if kf in df.columns and kf not in selected:
            selected.append(kf)
    
    # Filter to only those in feature_columns (the variable ones)
    selected_features = [f for f in selected if f in feature_columns]
    
    return selected_features

def format_percentage(value):
    """Format numbers as percentages for absolute errors"""
    if pd.isna(value):
        return None
    return f"{value * 100:.2f}%"

def format_percentage_direct(value):
    """Format numbers that are already in percentage form"""
    if pd.isna(value):
        return None
    return f"{value:.2f}%"

def load_config(config_path="Latest Data/Segregated/config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(config, config_path="Latest Data/Segregated/config.json"):
    """Save configuration to JSON file"""
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def get_available_games():
    """Get list of available games"""
    return ["GS006","GS023","GS015","G001","G003","G018","G019","G024","G025","G027","G028","G029","G030","G034","G035","G036","G038",
            "G041","G057","G053", "G054", "G059","G060", "G062", "G064", "G067","G069", "G037","G074","G070","G079","G042","G086",
            "G088","G107","G085","G108","G115","GS065","GS075","GS081","G100","G136","G105","G091","G087","G090","G092","G102",
             "G103"]
       

def prepare_training_data_with_holdout(model_type, df, dft, selected_features):
    """Prepare training data based on model type and selected features - works with holdout preprocessed data"""
    if model_type == "M1":
        train_features = selected_features.copy()
        if 'elapsed_months' not in train_features:
            train_features.append('elapsed_months')
        
        X = df[train_features].drop(df.index[-1]) if len(df) > 0 else df[train_features]
        y = df['transaction_period_ltv_cac'].drop(df.index[-1]) if len(df) > 0 else df['transaction_period_ltv_cac']
    else:  # M2
        additional_features = ['ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3', 'elapsed_months']
        train_features = [col for col in selected_features + additional_features if col in dft.columns]
        X = dft[train_features]
        y = dft['transaction_period_ltv_cac']
    
    return X, y

def prepare_training_data(model_type, df, dft, selected_features):
    """Prepare training data based on model type and selected features"""
    if model_type == "M1":
        train_features = selected_features.copy()
        if 'elapsed_months' not in train_features:
            train_features.append('elapsed_months')
        
        X = df[train_features].drop(df.index[-1])
        y = df['transaction_period_ltv_cac'].drop(df.index[-1])
    else:  # M2
        additional_features = ['ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3', 'elapsed_months']
        train_features = [col for col in selected_features + additional_features if col in dft.columns]
        X = dft[train_features]
        y = dft['transaction_period_ltv_cac']
    
    return X, y

def get_model_performance_summary(config):
    """Get summary of model performance across all games"""
    model_summary = {}
    
    for config_key, config_entry in config.items():
        if config_entry.get('all_model_rmse'):
            for model_type, rmse in config_entry['all_model_rmse'].items():
                if model_type not in model_summary:
                    model_summary[model_type] = []
                model_summary[model_type].append({
                    'game_id': config_entry['game_id'],
                    'rmse': rmse,
                    'is_best': model_type == config_entry['best_model']
                })
    
    return model_summary

def get_holdout_info_summary(config):
    """Get summary of holdout information across all games"""
    holdout_summary = []
    
    for config_key, config_entry in config.items():
        if 'training_months' in config_entry:
            holdout_summary.append({
                'game_id': config_entry['game_id'],
                'model_type': config_entry.get('model_type', 'Unknown'),
                'training_months': config_entry['training_months'],
                'holdout_months': config_entry.get('holdout_months', 0),
                'max_elapsed_months': config_entry.get('max_elapsed_months', 0),
                'best_model': config_entry['best_model'],
                'best_rmse': config_entry['best_rmse']
            })
    
    return holdout_summary 