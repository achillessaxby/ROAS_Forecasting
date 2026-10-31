import pandas as pd
import os
import numpy as np
import glob

# Configuration
DATA_FOLDER = "Latest Data/Segregated"
CATEGORY_FILE = "new.xlsx"  # 1st column = game_id, 3rd column = category
OUTPUT_FILE = "category_alpha.csv"
TARGET_CATEGORIES = ['Casual', 'Core', 'Non-gaming', 'Real-money']

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

def compute_game_alpha(game_id, data_folder, suffix="_Dataset_12_2024.csv"):
    """Compute alpha for a single game using the new formula"""
    file_path = os.path.join(data_folder, f"{game_id}{suffix}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.nan
    
    try:
        df = pd.read_csv(file_path)
        # Dropna on both relevant columns
        df = df.dropna(subset=["transaction_period_ltv_cac", "total_cac"])
        # Filter to positive values
        df = df[(df["transaction_period_ltv_cac"] > 0) & (df["total_cac"] > 0)]
        
        # Filter to elapsed_months == 1 (no adjustment)
        df1 = df[df["elapsed_months"] == 1]
        
        if len(df1) < 2:
            print(f"Insufficient data for {game_id} (only {len(df1)} rows for elapsed_months=1)")
            return np.nan
        
        # Select top 5 highest cohorts by total_cac descending
        top5 = df1.sort_values(by="total_cac", ascending=False).head(5)
        
        if len(top5) < 2:
            print(f"Insufficient top cohorts for {game_id} (only {len(top5)})")
            return np.nan
        
        # Re-sort the top 5 by total_cac ascending for pairing (to ensure positive spend_increase potential)
        top5_sorted = top5.sort_values('total_cac')
        
        alphas = []
        for i in range(len(top5_sorted) - 1):
            base_row = top5_sorted.iloc[i]
            new_row = top5_sorted.iloc[i + 1]
            
            roas_base = base_row['transaction_period_ltv_cac']
            roas_new = new_row['transaction_period_ltv_cac']
            cac_base = base_row['total_cac']
            cac_new = new_row['total_cac']
            
            spend_increase = (cac_new - cac_base) / cac_base
            
            if spend_increase == 0 or roas_new == 0:
                continue
            
            alpha = (roas_base / roas_new - 1) / spend_increase
            
            if alpha > 0:
                alphas.append(alpha)
        
        if not alphas:
            print(f"No valid positive alphas for {game_id}")
            return np.nan
        
        # Outlier removal for game alpha
        alphas_series = pd.Series(alphas)
        median_alpha = alphas_series.median()
        outlier_threshold = median_alpha * 3
        filtered_alphas = alphas_series[alphas_series <= outlier_threshold]
        
        if len(filtered_alphas) == 0:
            print(f"No valid alphas after outlier removal for {game_id}")
            return np.nan
        
        alpha_normal = filtered_alphas.mean()
        print(f"Computed alpha for {game_id}: {alpha_normal:.6f} (from {len(filtered_alphas)} valid pairs after outlier removal)")
        return alpha_normal
            
    except Exception as e:
        print(f"Error processing {game_id}: {str(e)}")
        return np.nan

def main():
    print("=== Generating Category Alpha Values ===")
    
    # Step 1: Get all game IDs
    print("\n1. Finding all game files...")
    all_games = get_all_game_files()
    print(f"Found {len(all_games)} games: {all_games}")
    
    # Step 2: Load category mapping
    print("\n2. Loading category mapping...")
    category_mapping = load_category_mapping()
    
    if not category_mapping:
        print("ERROR: No category mapping loaded. Exiting.")
        return
    
    # Step 3: Compute alpha for each game
    print("\n3. Computing alpha values for each game...")
    alpha_data = []
    
    for game_id in all_games:
        if game_id in category_mapping:
            category = category_mapping[game_id]
            alpha = compute_game_alpha(game_id, DATA_FOLDER)
            
            alpha_data.append({
                'game_id': game_id,
                'category': category,
                'alpha': alpha
            })
        else:
            print(f"No category found for {game_id}, skipping...")
    
    # Step 4: Create DataFrame and compute category averages
    print("\n4. Computing category averages...")
    alpha_df = pd.DataFrame(alpha_data)
    
    if alpha_df.empty:
        print("ERROR: No valid alpha data computed. Exiting.")
        return
    
    print(f"Computed alpha for {len(alpha_df)} games")
    print("\nGame-wise alpha summary:")
    print(alpha_df.groupby('category')['alpha'].describe())
    
    # Compute category averages
    category_averages = []
    
    for category in TARGET_CATEGORIES:
        category_games = alpha_df[alpha_df['category'] == category]
        
        if not category_games.empty:
            # Use normal average, excluding NaN values
            valid_alphas = category_games['alpha'].dropna()
            
            if len(valid_alphas) > 0:
                # Compute median
                median_alpha = valid_alphas.median()
                # Outlier threshold: median * 3
                outlier_threshold = median_alpha * 3
                # Filter out outliers (> threshold)
                filtered_alphas = valid_alphas[valid_alphas <= outlier_threshold]
                
                if len(filtered_alphas) > 0:
                    category_alpha = filtered_alphas.mean()
                else:
                    category_alpha = 0.05  # Placeholder if all are outliers
                
                # Ensure positive value, use 0.05 placeholder if negative or zero
                if category_alpha <= 0:
                    print(f"WARNING: {category} alpha is non-positive ({category_alpha:.6f}), using placeholder 0.05")
                    category_alpha = 0.05
                
                category_averages.append({
                    'category': category,
                    'category_alpha': category_alpha,
                    'games_count': len(filtered_alphas),
                    'games_list': ', '.join(category_games['game_id'].tolist())
                })
                
                print(f"{category}: alpha={category_alpha:.6f} (from {len(filtered_alphas)} games after outlier removal)")
            else:
                print(f"WARNING: No valid alpha values for {category}, using placeholder 0.05")
                category_averages.append({
                    'category': category,
                    'category_alpha': 0.05,
                    'games_count': 0,
                    'games_list': 'No valid data'
                })
        else:
            print(f"WARNING: No games found for {category}, using placeholder 0.05")
            category_averages.append({
                'category': category,
                'category_alpha': 0.05,
                'games_count': 0,
                'games_list': 'No games in category'
            })
    
    # Step 5: Save results
    print("\n5. Saving results...")
    
    # Save detailed game-wise data
    alpha_df.to_csv('game_alphas.csv', index=False)
    print(f"Saved detailed game alphas to game_alphas.csv")
    
    # Save category averages
    category_df = pd.DataFrame(category_averages)
    category_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved category alphas to {OUTPUT_FILE}")
    
    print("\n=== Final Category Alpha Values ===")
    for _, row in category_df.iterrows():
        print(f"{row['category']}: {row['category_alpha']:.6f} (from {row['games_count']} games)")
    
    print(f"\nGeneration complete! Use {OUTPUT_FILE} in your inference code.")

if __name__ == "__main__":
    main()