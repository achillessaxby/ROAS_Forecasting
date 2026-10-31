import os
import pandas as pd

# Epsilon to avoid floating-point comparison issues
EPSILON = 1e-6

def find_latest_nonzero_cohort(game_id):
    """
    Find the latest cohort with a non-zero total_cac value for a game.
    Returns a tuple of (latest_cohort_date, latest_cohort_cac)
    """
    try:
        # Load the game dataset
        data_path = os.path.join("Latest Data", "Segregated", f"{game_id}_Dataset_12_2024.csv")
        
        if not os.path.exists(data_path):
            print(f"Data file not found for game {game_id}")
            return None, 0
        
        df = pd.read_csv(data_path)
        
        # Filter to month 1 only
        if 'elapsed_months' in df.columns:
            df = df[df["elapsed_months"] == 1]
        
        if df.empty:
            print(f"No month 1 data found for game {game_id}")
            return None, 0
        
        # Find date column
        date_column = None
        for col in ['cohort_month', 'cohort', 'cohort_date']:
            if col in df.columns:
                date_column = col
                break
        
        if not date_column:
            print(f"No date column found for game {game_id}")
            return None, 0
        
        # Get latest cohort with non-zero total_cac
        try:
            # Try to parse dates
            df['tmp_date'] = pd.to_datetime(df[date_column], errors='coerce')
            
            if not df['tmp_date'].isna().all():
                # Sort by date (newest first)
                sorted_df = df[~df['tmp_date'].isna()].sort_values(by='tmp_date', ascending=False)
                
                # Find first non-zero cac
                for idx, row in sorted_df.iterrows():
                    if abs(row['total_cac']) > EPSILON:
                        cohort_date = row[date_column]
                        cohort_cac = row['total_cac']
                        print(f"Found latest cohort with non-zero total_cac for {game_id}: {cohort_date} with CAC = {cohort_cac}")
                        return cohort_date, cohort_cac
                
                print(f"No non-zero total_cac values found for any cohort of {game_id}")
                return None, 0
        except Exception as e:
            print(f"Error finding latest cohort for {game_id}: {str(e)}")
            return None, 0
    
    except Exception as e:
        print(f"Error processing game {game_id}: {str(e)}")
        return None, 0

# if __name__ == "__main__":
#     game_id = "G001"
#     cohort_date, cohort_cac = find_latest_nonzero_cohort(game_id)
#     print(f"Latest cohort date: {cohort_date}, CAC: {cohort_cac}")
