# app/inference.py
import os
import cloudpickle
import pickle
import pandas as pd
import numpy as np
import math
from typing import Dict, Tuple, List, Optional, Any, Union
from datetime import datetime
from scipy.interpolate import interp1d
from find_latest_cohort import find_latest_nonzero_cohort
try:
    from schemas import ForecastRequest, ForecastResult, MonthlyForecast
except Exception:
    # Lightweight fallbacks so importing this module does not crash
    from dataclasses import dataclass
    from typing import Dict, List, Optional

    @dataclass
    class MonthlyForecast:  # type: ignore
        elapsed_months: int
        current_month: str
        transactional_expected: float
        cumulative_expected: float
        absolute_revenue_expected: float

    @dataclass
    class ForecastResult:  # type: ignore
        game_id: str
        cohort_month: str
        forecasts: List[MonthlyForecast]

    @dataclass
    class ForecastRequest:  # type: ignore
        game_id: str
        cohort_month: str
        forecast_months: int
        total_cac: Optional[float] = None
        use_market_channels: bool = False
        channel_spends: Optional[Dict[str, float]] = None
import traceback

# Epsilon to avoid division by zero
EPSILON = 1e-6

# Available game IDs
available_game_ids = ['G001','G003','G018','G019','G024','G025','G027','G028','G029',
                     'G030', 'G034','G035','G036','G037','G038','G041','G042','G057',
                     'G053','G054','G059','G060','G062','G064','G067','G069','G070',
                     'G079','G074','G085','G087','G086','G088','G090','G092','G099',
                     'G100','G103','G107','G108','G115','GS006','GS015','GS023','GS065',
                     'GS075','G105','G136','G091', 'G102','GS125','G141','G142','GS999',
                     'G139','G145','G118','GS176','GS268','G154','G156','GS287','GS288',
                     'G160','G152','G111','GS189','GS129','GS264','GS274','GS349','GS342','GS242','GS331','GS332','GS338']

# Global cache for CSV data
csv_data_cache = None

# Games that use piecewise linear regression models
# Note: This list can be populated with game IDs that require piecewise models
piecewise_games = []

# Define all permitted marketing channels
PERMITTED_MC_CHANNELS = [
    "MC_Google", "MC_Facebook", "MC_Applovin", "MC_Moloco",
    "MC_Unity", "MC_Ironsource", "MC_Tiktok", "MC_Organic"
]

# Global caches for models, game metrics, and category averages
models_cache = {}  # (game_id, model_type) -> model
game_metrics_df = None
category_averages_df = None
game_to_category = {}
category_alpha_df = None

# Function to load CSV data
def load_csv_data():
    """Load the final_M2.csv file and cache it"""
    global csv_data_cache
    
    if csv_data_cache is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_M2.csv")
        print(f"Loading prediction data from {csv_path}...")
        csv_data_cache = pd.read_csv(csv_path)
        print(f"Loaded CSV data: {len(csv_data_cache)} rows for {csv_data_cache['game_id'].nunique()} games")
    
    return csv_data_cache

def find_latest_cohort_csv(game_id: str) -> tuple:
    """Find the latest cohort data for a game_id from CSV"""
    df = load_csv_data()
    
    # Filter for the specific game_id
    game_data = df[df['game_id'] == game_id]
    if game_data.empty:
        return None, None, 0.0
    
    # Convert cohort to datetime and find latest
    game_data['cohort_date'] = pd.to_datetime(game_data['cohort'])
    latest_cohort_date = game_data['cohort_date'].max()
    
    # Get the cohort string format
    latest_cohort = game_data[game_data['cohort_date'] == latest_cohort_date]['cohort'].iloc[0]
    
    # Get the latest cohort's total_cac if available
    if 'total_cac' in game_data.columns:
        cohort_total_cac = game_data[
            (game_data['cohort_date'] == latest_cohort_date) & 
            (game_data['elapsed_months'] == 1)
        ]['total_cac'].iloc[0] if not game_data[
            (game_data['cohort_date'] == latest_cohort_date) & 
            (game_data['elapsed_months'] == 1)
        ].empty else 0.0
    else:
        cohort_total_cac = 0.0
    
    print(f"Found latest cohort for {game_id}: {latest_cohort} with CAC=${cohort_total_cac:.2f}")
    return latest_cohort, latest_cohort_date, cohort_total_cac

def get_predictions_from_csv(game_id: str, cohort: str) -> Dict[int, float]:
    """Get predictions directly from the CSV file for a given game_id and cohort"""
    df = load_csv_data()
    
    # Filter for the specific game and cohort
    cohort_data = df[(df['game_id'] == game_id) & (df['cohort'] == cohort)]
    if cohort_data.empty:
        print(f"No data found for game {game_id}, cohort {cohort}")
        return {}
    
    # Sort by elapsed_months
    cohort_data = cohort_data.sort_values('elapsed_months')
    
    # Create dictionary of predictions for each month
    predictions = {}
    for _, row in cohort_data.iterrows():
        month = row['elapsed_months']
        
        # For month 1, use incremental_net_roas
        if month == 1:
            if 'incremental_net_roas' in row and not pd.isna(row['incremental_net_roas']):
                predictions[month] = row['incremental_net_roas']
                predictions[month] = max(0.0, predictions[month])
        # For months >= 2, use incremental_net_roas_expected_pred
        else:
            if 'incremental_net_roas_expected_pred' in row and not pd.isna(row['incremental_net_roas_expected_pred']):
                predictions[month] = row['incremental_net_roas_expected_pred']
                predictions[month] = max(0.0, predictions[month])
    
    print(f"Retrieved {len(predictions)} months of predictions for {game_id}, cohort {cohort}")
    return predictions

def load_helper_data():
    """Load game metrics, category averages, and category alpha data"""
    global game_metrics_df, category_averages_df, game_to_category, category_alpha_df
    
    try:
        game_metrics_df = pd.read_csv('game_metrics_output.csv')
        category_averages_df = pd.read_csv('category_averages_output.csv')
        game_to_category = dict(zip(game_metrics_df['game_id'], game_metrics_df['Category']))
        
        # Load category alpha values
        try:
            category_alpha_df = pd.read_csv('category_alpha.csv')
            print(f"Loaded category alpha data for {len(category_alpha_df)} categories")
        except Exception as alpha_e:
            print(f"Warning: Could not load category_alpha.csv: {str(alpha_e)}")
            category_alpha_df = None
            
    except Exception as e:
        print(f"Error loading helper data: {str(e)}")

def get_gamma(total_cac: float) -> float:
    """Calculate gamma based on total_cac value"""
    if total_cac < 300000:
        return 1.0
    elif total_cac < 1000000:
        return 1.4
    else:
        return 1.7

def compute_alpha_prime(game_id: str) -> float:
    """
    Compute the adjusted alpha prime value based on game and category metrics
    """
    global game_metrics_df, category_averages_df, game_to_category, category_alpha_df
    
    if game_metrics_df is None or category_averages_df is None or category_alpha_df is None:
        load_helper_data()
    
    # Fallback value if all else fails
    default_alpha = 0.05
    
    try:
        # Get game category
        category = game_to_category.get(game_id)
        if not category:
            print(f"No category found for {game_id}, using default alpha")
            return default_alpha
        
        # Get game and category metrics
        game_row = game_metrics_df[game_metrics_df['game_id'] == game_id]
        if game_row.empty:
            print(f"No game metrics found for {game_id}, using default alpha")
            return default_alpha
        
        category_row = category_averages_df[category_averages_df['Category'] == category]
        if category_row.empty:
            print(f"No category averages found for {category}, using default alpha")
            return default_alpha
        
        # Get category alpha from the generated category_alpha.csv
        if category_alpha_df is not None:
            alpha_row = category_alpha_df[category_alpha_df['category'] == category]
            if not alpha_row.empty:
                category_alpha = alpha_row['category_alpha'].values[0]
                print(f"Using data-driven category alpha for {category}: {category_alpha:.6f}")
            else:
                print(f"No alpha data found for category {category}, using default")
                category_alpha = default_alpha
        else:
            print(f"Category alpha data not loaded, using default for {category}")
            category_alpha = default_alpha
        
        # Extract IAA and IAP percentages
        game_IAA = game_row['IAA_pct'].values[0] / 100
        game_IAP = game_row['IAP_pct'].values[0] / 100
        
        category_IAA = category_row['IAA_pct'].values[0] / 100
        category_IAP = category_row['IAP_pct'].values[0] / 100
        
        # Calculate deltas
        delta_IAA = game_IAA - category_IAA
        delta_IAP = game_IAP - category_IAP
        
        # Adjust alpha based on deltas
        alpha_prime = category_alpha * (game_IAA * (1 + delta_IAA) + game_IAP * (1 + delta_IAP))
        
        # Ensure positive result
        result = max(0.0, alpha_prime)
        print(f"Computed alpha_prime for {game_id} ({category}): {result:.6f}")
        return result
    
    except Exception as e:
        print(f"Error computing alpha prime for {game_id}: {str(e)}")
        return default_alpha

def find_nearest_cohort(game_id: str, total_cac: float, is_extrapolation: bool = False) -> Dict[str, float]:
    """
    Find the appropriate cohort in the data and calculate MC channel ratios
    """
    try:
        # Load the game dataset
        data_path = os.path.join("Latest Data", "Segregated", f"{game_id}_Dataset_12_2024.csv")
        
        if not os.path.exists(data_path):
            # Return default equal distribution if data not found
            print(f"Data file not found for game {game_id}, using equal distribution")
            return {c: total_cac / len(PERMITTED_MC_CHANNELS) for c in PERMITTED_MC_CHANNELS}
        
        df = pd.read_csv(data_path)
        
        # Filter to month 1 only
        df = df[df["elapsed_months"] == 1]
        
        if df.empty:
            print(f"No month 1 data found for game {game_id}, using equal distribution")
            return {c: total_cac / len(PERMITTED_MC_CHANNELS) for c in PERMITTED_MC_CHANNELS}
        
        # Determine which column could represent cohort date
        date_column = None
        potential_date_columns = ['cohort_month', 'cohort', 'cohort_date']
        
        for col in potential_date_columns:
            if col in df.columns:
                date_column = col
                break
        
        # For extrapolation, use the latest cohort by date
        if is_extrapolation and date_column:
            try:
                # Try to parse as a date
                df['tmp_date'] = pd.to_datetime(df[date_column], errors='coerce')
                
                # If date parsing succeeded for at least some rows
                if not df['tmp_date'].isna().all():
                    # Get all valid dates sorted from newest to oldest
                    valid_dates = df[~df['tmp_date'].isna()].sort_values(by='tmp_date', ascending=False)
                    
                    # Look for the most recent cohort with non-zero total_cac
                    found_valid_cohort = False
                    for idx, row in valid_dates.iterrows():
                        if abs(row['total_cac']) > EPSILON:  # Check if total_cac is non-zero
                            nearest_row = row
                            cohort_date = row[date_column]
                            found_valid_cohort = True
                            print(f"Using LATEST cohort with non-zero total_cac for {game_id} with date {cohort_date} for extrapolation")
                            break
                    
                    # If all cohorts have zero total_cac (very unlikely), use the latest anyway
                    if not found_valid_cohort:
                        nearest_row = valid_dates.iloc[0]
                        cohort_date = nearest_row[date_column]
                        print(f"Warning: All recent cohorts for {game_id} have zero total_cac. Using latest anyway: {cohort_date}")
                else:
                    # Try string sorting if date parsing failed
                    sorted_strings = df[date_column].astype(str).sort_values(ascending=False)
                    
                    # Look for the most recent cohort with non-zero total_cac
                    found_valid_cohort = False
                    for idx in sorted_strings.index:
                        row = df.loc[idx]
                        if abs(row['total_cac']) > EPSILON:  # Check if total_cac is non-zero
                            nearest_row = row
                            cohort_date = row[date_column]
                            found_valid_cohort = True
                            print(f"Using LATEST cohort with non-zero total_cac for {game_id} with date {cohort_date} (sorted as string) for extrapolation")
                            break
                    
                    # If all cohorts have zero total_cac, use the latest anyway
                    if not found_valid_cohort:
                        latest_idx = sorted_strings.index[0]
                        nearest_row = df.loc[latest_idx]
                        cohort_date = nearest_row[date_column]
                        print(f"Warning: All recent cohorts for {game_id} have zero total_cac. Using latest anyway: {cohort_date}")
            except Exception as e:
                print(f"Error sorting by date for {game_id}: {str(e)}")
                # Fall back to nearest total_cac
                df['cac_diff'] = abs(df['total_cac'] - total_cac)
                nearest_row = df.loc[df['cac_diff'].idxmin()]
                print(f"Falling back to nearest total_cac for {game_id} due to date parsing error")
        else:
            # For non-extrapolation cases or when no date column is available, find nearest by total_cac
            df['cac_diff'] = abs(df['total_cac'] - total_cac)
            nearest_row = df.loc[df['cac_diff'].idxmin()]
            
            if is_extrapolation:
                print(f"No suitable date column found for {game_id}, using nearest total_cac for extrapolation")
        nearest_cac = nearest_row['total_cac']
        
        if is_extrapolation:
            print(f"Using latest cohort for {game_id} with total_cac {nearest_cac} for extrapolation (requested: {total_cac})")
        else:
            print(f"Found nearest cohort for {game_id} with total_cac {nearest_cac} (requested: {total_cac})")
        
        # Calculate MC channel values from the nearest cohort
        mc_inputs = {}
        other_channels_sum = 0.0
        total_mc = 0.0
        
        # Get all columns that might be marketing channels (case insensitive)
        marketing_cols = [col for col in nearest_row.index if isinstance(col, str) and 
                         (col.lower().startswith('mc_') or col.lower().startswith('marketing'))]
        
        # Print all marketing columns found
        print(f"Found {len(marketing_cols)} marketing columns in {game_id} data:")
        for col in marketing_cols:
            if not pd.isna(nearest_row[col]):
                print(f"  - {col}: {float(nearest_row[col]):.4f}")
            else:
                print(f"  - {col}: NaN")
        
        # First pass - collect all exact matches to PERMITTED_MC_CHANNELS
        for channel in PERMITTED_MC_CHANNELS:
            # Try to find an exact match first
            if channel in nearest_row and not pd.isna(nearest_row[channel]):
                mc_inputs[channel] = float(nearest_row[channel])
                total_mc += mc_inputs[channel]
                print(f"  Exact match: {channel} = {mc_inputs[channel]:.4f}")
        
        # Second pass - try case-insensitive matching for remaining channels
        for col in marketing_cols:
            # Skip if already processed as exact match
            if col in mc_inputs:
                continue
                
            col_lower = col.lower()
            # Try to match with permitted channels case-insensitively
            matched = False
            for channel in PERMITTED_MC_CHANNELS:
                if channel.lower() == col_lower:
                    mc_inputs[channel] = float(nearest_row[col])
                    total_mc += mc_inputs[channel]
                    print(f"  Case-insensitive match: {col} → {channel} = {mc_inputs[channel]:.4f}")
                    matched = True
                    break
            
            # If not matched to a permitted channel, add to "others"
            if not matched and not pd.isna(nearest_row[col]):
                val = float(nearest_row[col])
                other_channels_sum += val
                print(f"  Adding to others: {col} = {val:.4f}")
        
        # Add MC_others with the sum of unmatched channels
        if other_channels_sum > 0:
            mc_inputs["MC_others"] = other_channels_sum
            total_mc += other_channels_sum
            print(f"  MC_others = {other_channels_sum:.4f}")
        
        # If total is zero, use equal distribution
        if total_mc <= EPSILON:
            print(f"Warning: Zero marketing spend in nearest cohort for {game_id}, using equal distribution")
            return {c: total_cac / len(PERMITTED_MC_CHANNELS) for c in PERMITTED_MC_CHANNELS}
        
        # EXTRAPOLATION CASE: Use exact marketing channel values without scaling
        if is_extrapolation:
            print(f"Using EXACT marketing channel distribution for {game_id} (extrapolation case)")
            print("Marketing channels (extrapolation):")
            for ch, val in sorted(mc_inputs.items()):
                print(f"  - {ch}: {val:.4f}")
            print(f"  Total: {sum(mc_inputs.values()):.4f}")
            return mc_inputs
        
        # INTERPOLATION CASE: Scale marketing channels based on ratio
        print(f"Scaling marketing channel distribution by ratio for {game_id} (interpolation case)")
        scaled_mc_inputs = {}
        for channel, value in mc_inputs.items():
            # Preserve the ratio but scale to match the requested total_cac
            ratio = value / total_mc
            scaled_mc_inputs[channel] = ratio * total_cac
        
        # Print marketing channels after scaling
        print("Marketing channels (after scaling):")
        for ch, val in sorted(scaled_mc_inputs.items()):
            print(f"  - {ch}: {val:.4f}")
        print(f"  Total: {sum(scaled_mc_inputs.values()):.4f}")
        
        return scaled_mc_inputs
    
    except Exception as e:
        print(f"Error finding nearest cohort for {game_id}: {str(e)}")
        traceback.print_exc()
        # Return default equal distribution on error
        return {c: total_cac / len(PERMITTED_MC_CHANNELS) for c in PERMITTED_MC_CHANNELS}

def normal_prediction(mc_inputs: Dict[str, float], game_id: str, cohort_date: datetime) -> Dict[int, float]:
    """
    Make normal predictions from CSV data without extrapolation
    """
    print(f"\n===== MAKING NORMAL PREDICTION FOR {game_id} =====")
    
    # Find the latest cohort in the CSV data
    latest_cohort, _, _ = find_latest_cohort_csv(game_id)
    if not latest_cohort:
        print(f"ERROR: No cohort data found for {game_id} in CSV")
        return {}

    # Get predictions for the latest cohort
    predictions = get_predictions_from_csv(game_id, latest_cohort)
    
    if not predictions:
        print(f"ERROR: No predictions found for {game_id}, cohort {latest_cohort}")
        return {}
    
    print(f"Retrieved {len(predictions)} months of normal predictions for {game_id}")
    return predictions

def extrapolate(new_cac: float, mc_inputs: Dict[str, float], game_id: str, 
              cohort_date: datetime) -> Dict[int, float]:
    """
    Extrapolate predictions using alpha, beta, gamma parameters
    Always using the Normal method
    """
    # Special case handling for GS015 (fallback logic)
    if game_id == 'GS015':
        print(f"\n===== SPECIAL HANDLING FOR {game_id} =====")
        # Using fixed base CAC for GS015 special case
        base_cohort_cac = 99000.0  # Fixed value for GS015
        base_cohort_date = cohort_date  # Use requested cohort date
        
        # Standard marketing channel distribution for GS015
        base_mc_inputs = {
            "MC_Google": base_cohort_cac * 0.25,       # 25%
            "MC_Facebook": base_cohort_cac * 0.25,     # 25%
            "MC_Applovin": base_cohort_cac * 0.1,      # 10%
            "MC_Moloco": base_cohort_cac * 0.1,        # 10%
            "MC_Unity": base_cohort_cac * 0.1,         # 10%
            "MC_Ironsource": base_cohort_cac * 0.1,    # 10%
            "MC_Tiktok": base_cohort_cac * 0.05,       # 5%
            "MC_Organic": base_cohort_cac * 0.05       # 5%
        }
        
        # Get base predictions
        try:
            # Find the latest cohort in the CSV data
            latest_cohort, _, _ = find_latest_cohort_csv(game_id)
            if not latest_cohort:
                print(f"ERROR: No cohort data found for {game_id}")
                return {}
    
            # Get predictions for the latest cohort
            base_predictions = get_predictions_from_csv(game_id, latest_cohort)
            
            if not base_predictions:
                print(f"WARNING: Failed to get base predictions for {game_id}. Using normal prediction.")
                return normal_prediction(mc_inputs, game_id, cohort_date)
                
            # Filter out unreasonably high predictions (they should be 0)
            # A reasonable upper limit for LTV would be around 10
            filtered_predictions = {}
            for month, value in base_predictions.items():
                if value > 100.0 or value<0:  # Unreasonably high prediction
                    print(f"WARNING: Month {month} had unusually high prediction (${value:.4f}), setting to 0")
                    filtered_predictions[month] = 0.0
                else:
                    filtered_predictions[month] = value
                    filtered_predictions[month] = max(0.0, filtered_predictions[month])
                    
            base_predictions = filtered_predictions  # Use filtered predictions
            
            # Calculate increase relative to base cohort
            inc = (new_cac - base_cohort_cac) / base_cohort_cac
            
            # Calculate key parameters for extrapolation (no beta for GS015)
            alpha_prime = compute_alpha_prime(game_id)  # Game-specific adjustment
            gamma = get_gamma(new_cac)                  # CAC-level adjustment
            beta_decay = 1.0                            # No beta for GS015
            
            # Ensure alpha is positive
            effective_alpha = abs(alpha_prime * gamma)
            
            # Calculate extrapolation factor
            combined_factor = effective_alpha * beta_decay * inc
            adjustment_factor = 1 / (1 + combined_factor)
            
            # Print debug information
            print("\n========== GS015 EXTRAPOLATION PARAMETERS ==========")
            print(f"Base CAC: ${base_cohort_cac:.2f}")
            print(f"Requested CAC: ${new_cac:.2f}, Increase: {inc:.4f} ({inc*100:.1f}%)")
            print(f"Alpha: {alpha_prime:.4f}, Gamma: {gamma:.4f}, Beta: FIXED at 1.0")
            print(f"Effective Alpha: {effective_alpha:.4f}, Combined Factor: {combined_factor:.4f}")
            print(f"Adjustment Factor: {adjustment_factor:.4f}")
            
            # Show sample predictions for debugging (all months)
            print("\n========== FILTERED PREDICTIONS ==========")
            print("Month: Value (after filtering unreasonable values)")
            for month in sorted(base_predictions.keys()):
                print(f"Month {month}: ${base_predictions[month]:.4f}")
                
            # Show sample extrapolated predictions
            print("\n========== SAMPLE PREDICTIONS ==========")
            if 1 in base_predictions and 12 in base_predictions:
                base_m1 = base_predictions[1]
                base_m12 = base_predictions[12]
                adj_m1 = base_m1 * adjustment_factor
                adj_m12 = base_m12 * adjustment_factor
                
                print(f"Month 1:  Base=${base_m1:.4f}, Adjusted=${adj_m1:.4f}, Factor={adjustment_factor:.4f}")
                print(f"Month 12: Base=${base_m12:.4f}, Adjusted=${adj_m12:.4f}, Factor={adjustment_factor:.4f}")
                print("=========================================\n")
            
            # Apply adjustment to base predictions
            return {m: base_predictions[m] * adjustment_factor for m in base_predictions}
            
        except Exception as e:
            print(f"Error in GS015 special handling: {str(e)}")
            return normal_prediction(mc_inputs, game_id, cohort_date)
    
    try:
        # Find the latest cohort with non-zero total_cac using the shared function
        base_cohort_date, base_cohort_cac = find_latest_nonzero_cohort(game_id)
        
        # If no valid latest cohort found from the data files, try from CSV
        if base_cohort_cac <= EPSILON:
            # Find the latest cohort from CSV
            _, csv_cohort_date, csv_cohort_cac = find_latest_cohort_csv(game_id)
            if csv_cohort_date and csv_cohort_cac > EPSILON:
                base_cohort_date = csv_cohort_date
                base_cohort_cac = csv_cohort_cac
                print(f"Using cohort data from CSV: date={base_cohort_date}, CAC=${base_cohort_cac:.2f}")
            else:
                print(f"WARNING: No valid base cohort found for {game_id}. Using normal prediction.")
                return normal_prediction(mc_inputs, game_id, cohort_date)
        
        # Print the extrapolation direction
        extrapolation_direction = "upward" if new_cac > base_cohort_cac else "downward"
        print(f"Performing {extrapolation_direction} extrapolation from base CAC=${base_cohort_cac:.2f} to requested CAC=${new_cac:.2f}")
        
        # Get latest cohort from CSV
        latest_cohort, _, _ = find_latest_cohort_csv(game_id)
        if not latest_cohort:
            print(f"ERROR: No cohort data found for {game_id}")
            return normal_prediction(mc_inputs, game_id, cohort_date)
        
        # Get base predictions from CSV
        base_predictions = get_predictions_from_csv(game_id, latest_cohort)
        
        if not base_predictions:
            print(f"ERROR: Failed to get base predictions for {game_id} with base CAC={base_cohort_cac}.")
            return normal_prediction(mc_inputs, game_id, cohort_date)
        
        # Calculate increase relative to base cohort
        inc = (new_cac - base_cohort_cac) / base_cohort_cac
        
        # Calculate key parameters for extrapolation
        alpha_prime = compute_alpha_prime(game_id)  # Game-specific adjustment
        gamma = get_gamma(new_cac)                  # CAC-level adjustment
        
        # Calculate beta decay factor (original logic restored)
        beta_decay = 1.0  # Starting value
        try:
            for channel, value in mc_inputs.items():
                # Apply higher weight to organic channels (they scale better)
                if 'organic' in channel.lower():
                    channel_weight = 0.6
                else:
                    channel_weight = 1.0
                    
                # Calculate the channel's contribution to beta
                channel_ratio = value / new_cac if new_cac > 0 else 0
                beta_decay += channel_weight * channel_ratio
                
            # Check if beta_decay is NaN and use fallback value if it is
            if np.isnan(beta_decay):
                print(f"WARNING: Beta calculation resulted in NaN for {game_id}. Using default beta=1.0")
                beta_decay = 1.0
        except Exception as e:
            print(f"Error calculating beta for {game_id}: {str(e)}. Using default beta=1.0")
            beta_decay = 1.0
        
        # Ensure alpha is positive
        effective_alpha = abs(alpha_prime * gamma)
        
        # Calculate extrapolation factor with NaN protection
        combined_factor = effective_alpha * beta_decay * inc
        
        # Handle NaN in combined_factor
        if np.isnan(combined_factor):
            print(f"WARNING: Combined factor calculation resulted in NaN for {game_id}. Using default value.")
            # For NaN combined_factor, use a safe default based on inc direction
            combined_factor = effective_alpha * inc  # Use without beta
            
        # Calculate adjustment factor with safeguards
        try:
            adjustment_factor = 1 / (1 + combined_factor)
            if adjustment_factor < 0:
                adjustment_factor = 0.0
            # Handle potential NaN in adjustment_factor
            if np.isnan(adjustment_factor):
                print(f"WARNING: Adjustment factor is NaN for {game_id}. Using fallback value.")
                # Safe default that preserves direction
                adjustment_factor = 1.0 if inc >= 0 else 1.0 / (1 + abs(inc))
        except Exception as e:
            print(f"Error calculating adjustment factor: {str(e)}. Using safe default.")
            # Safe default that preserves direction
            adjustment_factor = 1.0 if inc >= 0 else 1.0 / (1 + abs(inc))
        
        # Print debug information
        print("\n========== EXTRAPOLATION PARAMETERS ==========")
        print(f"Base Cohort Date: {base_cohort_date}, CAC: ${base_cohort_cac:.2f}")
        change_type = "Increase" if inc > 0 else "Decrease"
        print(f"Requested CAC: ${new_cac:.2f}, {change_type}: {abs(inc):.4f} ({abs(inc*100):.1f}%)")
        print(f"Alpha: {alpha_prime:.4f}, Gamma: {gamma:.4f}, Beta: {beta_decay:.4f}")
        print(f"Effective Alpha: {effective_alpha:.4f}, Combined Factor: {combined_factor:.4f}")
        print(f"Adjustment Factor: {adjustment_factor:.4f}")
        
        # Show sample predictions for debugging (month 1 and 12)
        if 1 in base_predictions and 12 in base_predictions:
            print("\n========== SAMPLE PREDICTIONS ==========")
            base_m1 = base_predictions[1]
            base_m12 = base_predictions[12]
            adj_m1 = base_m1 * adjustment_factor
            adj_m12 = base_m12 * adjustment_factor
            
            print(f"Month 1:  Base=${base_m1:.4f}, Adjusted=${adj_m1:.4f}, Factor={adjustment_factor:.4f}")
            print(f"Month 12: Base=${base_m12:.4f}, Adjusted=${adj_m12:.4f}, Factor={adjustment_factor:.4f}")
            print("=========================================\n")
        
        # Apply adjustment to base predictions
        return {m: base_predictions[m] * adjustment_factor for m in base_predictions}
    
    except Exception as e:
        print(f"Error in extrapolation for {game_id}: {str(e)}")
        traceback.print_exc()
        return normal_prediction(mc_inputs, game_id, cohort_date)

def run_forecast(req: ForecastRequest) -> ForecastResult:
    """
    Run forecast for a single request using CSV-based predictions
    """
    # Check if game_id is supported
    if req.game_id not in available_game_ids:
        raise ValueError(f"Game ID {req.game_id} not supported")
    
    # Parse cohort date
    cohort_date = pd.to_datetime(req.cohort_month)
    
    # Find latest cohort with non-zero total_cac using the shared function
    latest_cohort_date, latest_cohort_cac = find_latest_nonzero_cohort(req.game_id)
    
    # If not found in data file, try CSV
    if latest_cohort_cac <= EPSILON:
        _, csv_cohort_date, csv_cohort_cac = find_latest_cohort_csv(req.game_id)
        if csv_cohort_date and csv_cohort_cac > EPSILON:
            latest_cohort_date = csv_cohort_date
            latest_cohort_cac = csv_cohort_cac
    
    # NEW APPROACH: Always use extrapolation, positive or negative
    is_extrapolation = True
    if req.total_cac and latest_cohort_cac > 0:
        direction = "higher" if req.total_cac > latest_cohort_cac else "lower"
        change_pct = abs((req.total_cac - latest_cohort_cac) / latest_cohort_cac * 100)
        print(f"Always using extrapolation: requested CAC (${req.total_cac:.2f}) is {direction} than latest cohort CAC (${latest_cohort_cac:.2f}) from {latest_cohort_date} by {change_pct:.1f}%")
    
    # Determine total_cac
    if req.use_market_channels:
        # Market segmentation approach
        if req.channel_spends:
            # User provided channel spends
            # Fix C5: map known aliases and reject unknown channels instead of silently dropping their spend
            aliases = {"MC_TicToc": "MC_Tiktok", "MC_TikTok": "MC_Tiktok"}
            spends = {}
            for k, v in req.channel_spends.items():
                spends[aliases.get(k, k)] = spends.get(aliases.get(k, k), 0.0) + v
            unknown = sorted(set(spends) - set(PERMITTED_MC_CHANNELS))
            if unknown:
                raise ValueError(f"Unknown channel(s) {unknown}; allowed: {PERMITTED_MC_CHANNELS}")
            mc_inputs = {c: spends.get(c, 0.0) for c in PERMITTED_MC_CHANNELS}
            total_cac = sum(mc_inputs.values())
        else:
            # No channel spends provided, use total_cac to find nearest cohort
            if req.total_cac is None:
                raise ValueError("Either channel_spends or total_cac must be provided")
            
            total_cac = req.total_cac
            mc_inputs = find_nearest_cohort(req.game_id, total_cac, is_extrapolation)
    else:
        # Non-segmented approach
        if req.total_cac is None:
            raise ValueError("total_cac is required when not using market segmentation")
        
        total_cac = req.total_cac
        mc_inputs = find_nearest_cohort(req.game_id, total_cac, is_extrapolation)
    
    # ALWAYS use extrapolation per new approach
    # We're setting is_extrapolation=True earlier, but enforcing it here for safety
    predictions = extrapolate(total_cac, mc_inputs, req.game_id, cohort_date)
    
    # Limit to requested forecast months
    predictions = {m: v for m, v in predictions.items() if m <= req.forecast_months}
    
    # Generate result
    forecasted_results = []
    cumulative_value = 0.0
    
    for month, prediction in sorted(predictions.items()):
        current_date = cohort_date + pd.DateOffset(months=month-1)
        
        # Calculate values
        transactional_value = prediction
        cumulative_value += transactional_value
        absolute_revenue = total_cac * transactional_value if total_cac > 0 else 0.0
        
        forecasted_results.append({
            "elapsed_months": month,
            "current_month": current_date.strftime('%Y-%m'),
            "transactional_expected": transactional_value,
            "cumulative_expected": cumulative_value, 
            "absolute_revenue_expected": absolute_revenue
        })
    
    # Construct result
    monthly_forecasts = [MonthlyForecast(**res) for res in forecasted_results]
    
    return ForecastResult(
        game_id=req.game_id,
        cohort_month=req.cohort_month,
        forecasts=monthly_forecasts
    )
