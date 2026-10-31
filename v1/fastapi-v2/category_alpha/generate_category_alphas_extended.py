import pandas as pd
import os
import numpy as np
import glob

# Configuration
DATA_FOLDER = "Latest Data/Segregated"
CATEGORY_FILE = "new.xlsx"  # 1st column = game_id, 3rd column = category
OUTPUT_FILE = "category_alpha_all_variables.csv"
GAME_OUTPUT_FILE = "game_alphas_all_variables.csv"
TARGET_CATEGORIES = ['Casual', 'Core', 'Non-gaming', 'Real-money']

# Define base variables we want to calculate alpha for
BASE_TARGET_VARIABLES = [
    # LTV/CAC ratios (primary metrics)
    'transaction_period_ltv_cac',
    # Note: m0_ltv_cac and cumulative_ltv_cac are equal to transaction_period_ltv_cac for elapsed_months=1
    # So we'll use the same alpha value for them in projection
    'day_1_cumulative_ltv_cac',
    'day_7_cumulative_ltv_cac', 
    'day_14_cumulative_ltv_cac',
    'day_30_cumulative_ltv_cac',
    
    # Retention metrics
    'day_1_payer_retention',
    'day_7_payer_retention',
    'day_14_payer_retention', 
    'day_30_payer_retention',
    'day_1_net_revenue_retention',
    'day_7_net_revenue_retention',
    'day_14_net_revenue_retention',
    'day_30_net_revenue_retention',
    'user_retention',
    'revenue_retention',
    
    # Revenue metrics  
    'gross_revenue',
    'net_revenue',
    'avg_gross_revenue',
    'avg_net_revenue',
    
    # User/transaction metrics
    'users',
    'transactions'
]

def get_dynamic_target_variables():
    """
    Dynamically identify all MC_ variables from the data files and combine with base variables
    """
    try:
        # Get a sample file to identify MC_ columns
        sample_files = glob.glob(os.path.join(DATA_FOLDER, "G*.csv"))
        if sample_files:
            sample_df = pd.read_csv(sample_files[0])
            mc_columns = [col for col in sample_df.columns if col.startswith('MC_')]
            print(f"Found {len(mc_columns)} MC_ columns: {mc_columns}")
            
            # Combine base variables with MC_ columns
            all_variables = BASE_TARGET_VARIABLES + mc_columns
            print(f"Total variables for alpha calculation: {len(all_variables)}")
            return all_variables
        else:
            print("No sample files found, using base variables only")
            return BASE_TARGET_VARIABLES
    except Exception as e:
        print(f"Error identifying MC_ columns: {e}")
        return BASE_TARGET_VARIABLES

# Will be set dynamically in main()
TARGET_VARIABLES = []

def get_all_game_files():
    """Get all CSV files starting with 'G' in the data folder"""
    pattern1 = os.path.join(DATA_FOLDER, "G*.csv")
    pattern2 = os.path.join(DATA_FOLDER, "G*.CSV")
    
    files = glob.glob(pattern1) + glob.glob(pattern2)
    game_ids = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract game_id from filename (assuming format like G001_Dataset_12_2024.csv)
        game_id = filename.split('_')[0]
        if game_id.startswith('G'):
            game_ids.append(game_id)
    
    return sorted(list(set(game_ids)))

def load_category_mapping():
    """Load category mapping from new.xlsx"""
    try:
        df = pd.read_excel(CATEGORY_FILE)
        # Get 1st and 3rd columns (0-indexed: 0 and 2)
        category_mapping = {}
        
        for _, row in df.iterrows():
            game_id = str(row.iloc[0]).strip()  # 1st column
            category = str(row.iloc[2]).strip()  # 3rd column (category)
            
            # Only include target categories
            if category in TARGET_CATEGORIES:
                category_mapping[game_id] = category
        
        print(f"Loaded category mapping for {len(category_mapping)} games")
        print(f"Categories found: {set(category_mapping.values())}")
        return category_mapping
        
    except Exception as e:
        print(f"Error loading category mapping: {str(e)}")
        return {}

def compute_variable_alpha(variable_values, cac_values, variable_name):
    """Compute alpha for a specific variable using pairs of cohorts"""
    alphas = []
    
    for i in range(len(variable_values) - 1):
        base_value = variable_values[i]
        new_value = variable_values[i + 1]
        cac_base = cac_values[i]
        cac_new = cac_values[i + 1]
        
        # Skip if any value is invalid
        if pd.isna(base_value) or pd.isna(new_value) or pd.isna(cac_base) or pd.isna(cac_new):
            continue
            
        if base_value <= 0 or new_value <= 0 or cac_base <= 0 or cac_new <= 0:
            continue
            
        spend_increase = (cac_new - cac_base) / cac_base
        
        if spend_increase == 0:
            continue
        
        # Alpha formula: (base_value / new_value - 1) / spend_increase
        alpha = (base_value / new_value - 1) / spend_increase
        
        # Only keep positive alphas for most variables
        # (Some variables might naturally have negative alpha, but we'll filter later)
        if not np.isnan(alpha) and np.isfinite(alpha):
            alphas.append(alpha)
    
    return alphas

def compute_game_alphas_all_variables(game_id, data_folder, suffix="_Dataset_12_2024.csv"):
    """Compute alpha for all variables for a single game"""
    file_path = os.path.join(data_folder, f"{game_id}{suffix}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter to elapsed_months == 1
        df1 = df[df["elapsed_months"] == 1].copy()
        
        if len(df1) < 2:
            print(f"Insufficient data for {game_id} (only {len(df1)} rows for elapsed_months=1)")
            return {}
        
        # Ensure we have total_cac column
        if 'total_cac' not in df1.columns:
            print(f"Missing total_cac column for {game_id}")
            return {}
            
        # Filter to positive total_cac values
        df1 = df1[df1["total_cac"] > 0]
        
        if len(df1) < 2:
            print(f"Insufficient positive CAC data for {game_id}")
            return {}
        
        # Select top 5 highest cohorts by total_cac descending
        top5 = df1.sort_values(by="total_cac", ascending=False).head(5)
        
        if len(top5) < 2:
            print(f"Insufficient top cohorts for {game_id} (only {len(top5)})")
            return {}
        
        # Re-sort the top 5 by total_cac ascending for pairing
        top5_sorted = top5.sort_values('total_cac')
        
        # Extract CAC values for spend increase calculation
        cac_values = top5_sorted['total_cac'].values
        
        # Initialize results dictionary
        game_alphas = {'game_id': game_id}
        
        # Compute alpha for each target variable
        for variable in TARGET_VARIABLES:
            if variable in top5_sorted.columns:
                variable_values = top5_sorted[variable].values
                
                # Compute alphas for this variable
                alphas = compute_variable_alpha(variable_values, cac_values, variable)
                
                if alphas:
                    # Special handling for transaction_period_ltv_cac (use original method)
                    if variable == 'transaction_period_ltv_cac':
                        # Original method: only positive alphas, median * 3 threshold
                        positive_alphas = [alpha for alpha in alphas if alpha > 0]
                        
                        if positive_alphas:
                            alphas_series = pd.Series(positive_alphas)
                            median_alpha = alphas_series.median()
                            outlier_threshold = median_alpha * 3  # Original method
                            filtered_alphas = alphas_series[alphas_series <= outlier_threshold]
                            
                            if len(filtered_alphas) > 0:
                                final_alpha = filtered_alphas.mean()
                                game_alphas[f'{variable}_alpha'] = final_alpha
                                game_alphas[f'{variable}_pairs_count'] = len(filtered_alphas)
                            else:
                                game_alphas[f'{variable}_alpha'] = np.nan
                                game_alphas[f'{variable}_pairs_count'] = 0
                        else:
                            game_alphas[f'{variable}_alpha'] = np.nan
                            game_alphas[f'{variable}_pairs_count'] = 0
                    else:
                        # Enhanced method for all other variables
                        alphas_series = pd.Series(alphas)
                        median_alpha = alphas_series.median()
                        outlier_threshold = abs(median_alpha) * 3  # Use absolute value
                        
                        # Filter outliers based on absolute value
                        filtered_alphas = alphas_series[abs(alphas_series) <= outlier_threshold]
                        
                        if len(filtered_alphas) > 0:
                            final_alpha = filtered_alphas.mean()
                            game_alphas[f'{variable}_alpha'] = final_alpha
                            game_alphas[f'{variable}_pairs_count'] = len(filtered_alphas)
                        else:
                            game_alphas[f'{variable}_alpha'] = np.nan
                            game_alphas[f'{variable}_pairs_count'] = 0
                else:
                    game_alphas[f'{variable}_alpha'] = np.nan
                    game_alphas[f'{variable}_pairs_count'] = 0
            else:
                game_alphas[f'{variable}_alpha'] = np.nan
                game_alphas[f'{variable}_pairs_count'] = 0
        
        # Count total valid alphas computed
        valid_alphas = sum(1 for key in game_alphas.keys() 
                          if key.endswith('_alpha') and not pd.isna(game_alphas[key]))
        
        print(f"Computed alpha for {game_id}: {valid_alphas}/{len(TARGET_VARIABLES)} variables")
        return game_alphas
            
    except Exception as e:
        print(f"Error processing {game_id}: {str(e)}")
        return {}

def main():
    global TARGET_VARIABLES
    
    print("=== Generating Alpha Values for All Variables (Enhanced) ===")
    
    # Step 0: Dynamically identify all target variables including MC_ columns
    print("\n0. Identifying target variables...")
    TARGET_VARIABLES = get_dynamic_target_variables()
    
    # Step 1: Get all game IDs
    print("\n1. Finding all game files...")
    all_games = get_all_game_files()
    print(f"Found {len(all_games)} games: {all_games[:10]}...")  # Show first 10
    
    # Step 2: Load category mapping
    print("\n2. Loading category mapping...")
    category_mapping = load_category_mapping()
    
    if not category_mapping:
        print("ERROR: No category mapping loaded. Exiting.")
        return
    
    # Step 3: Compute alpha for all variables for each game
    print("\n3. Computing alpha values for all variables...")
    all_game_data = []
    
    for game_id in all_games:
        if game_id in category_mapping:
            category = category_mapping[game_id]
            game_alphas = compute_game_alphas_all_variables(game_id, DATA_FOLDER)
            
            if game_alphas:  # If we got valid results
                game_alphas['category'] = category
                all_game_data.append(game_alphas)
        else:
            print(f"No category found for {game_id}, skipping...")
    
    if not all_game_data:
        print("ERROR: No valid alpha data computed. Exiting.")
        return
    
    # Step 4: Create DataFrame
    print("\n4. Creating comprehensive alpha dataset...")
    alpha_df = pd.DataFrame(all_game_data)
    
    print(f"Computed alpha for {len(alpha_df)} games")
    print(f"Variables processed: {len(TARGET_VARIABLES)}")
    
    # Step 5: Compute category averages for each variable
    print("\n5. Computing category averages for each variable...")
    category_summary = []
    
    for category in TARGET_CATEGORIES:
        category_games = alpha_df[alpha_df['category'] == category]
        
        if not category_games.empty:
            category_row = {'category': category, 'games_count': len(category_games)}
            
            # Compute average alpha for each variable
            for variable in TARGET_VARIABLES:
                alpha_col = f'{variable}_alpha'
                
                if alpha_col in category_games.columns:
                    valid_alphas = category_games[alpha_col].dropna()
                    
                    if len(valid_alphas) > 0:
                        # Outlier removal at category level
                        median_alpha = valid_alphas.median()
                        outlier_threshold = abs(median_alpha) * 3
                        filtered_alphas = valid_alphas[abs(valid_alphas) <= outlier_threshold]
                        
                        if len(filtered_alphas) > 0:
                            category_alpha = filtered_alphas.mean()
                            category_row[alpha_col] = category_alpha
                            category_row[f'{variable}_games_count'] = len(filtered_alphas)
                        else:
                            category_row[alpha_col] = np.nan
                            category_row[f'{variable}_games_count'] = 0
                    else:
                        category_row[alpha_col] = np.nan
                        category_row[f'{variable}_games_count'] = 0
                else:
                    category_row[alpha_col] = np.nan
                    category_row[f'{variable}_games_count'] = 0
            
            category_summary.append(category_row)
            
            # Print summary for this category
            valid_variables = sum(1 for var in TARGET_VARIABLES 
                                if not pd.isna(category_row.get(f'{var}_alpha', np.nan)))
            print(f"{category}: {valid_variables}/{len(TARGET_VARIABLES)} variables have valid alpha")
        else:
            print(f"WARNING: No games found for {category}")
    
    # Step 6: Save results
    print("\n6. Saving results...")
    
    # Save detailed game-wise data
    alpha_df.to_csv(GAME_OUTPUT_FILE, index=False)
    print(f"Saved detailed game alphas to {GAME_OUTPUT_FILE}")
    
    # Save category averages
    if category_summary:
        category_df = pd.DataFrame(category_summary)
        category_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved category alphas to {OUTPUT_FILE}")
        
        # Print summary of key variables
        print("\n=== Key Variable Alpha Summary by Category ===")
        key_vars = ['transaction_period_ltv_cac', 'day_30_cumulative_ltv_cac', 
                   'day_30_payer_retention', 'gross_revenue']
        
        for _, row in category_df.iterrows():
            print(f"\n{row['category']} (from {row['games_count']} games):")
            for var in key_vars:
                alpha_val = row.get(f'{var}_alpha', np.nan)
                games_count = row.get(f'{var}_games_count', 0)
                if not pd.isna(alpha_val):
                    print(f"  {var}: α = {alpha_val:.6f} (from {games_count} games)")
                else:
                    print(f"  {var}: No valid data")
    
    print(f"\nGeneration complete! Check {OUTPUT_FILE} and {GAME_OUTPUT_FILE}")
    print("\nYou can now use these alpha values to project complete new rows based on spend changes!")

if __name__ == "__main__":
    main() 