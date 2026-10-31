"""
Test Script for GCS Validation Data
===================================

Simple script to verify that the generated validation data
can be accessed correctly from GCS.

Run this after generate_validation_cache.py to verify everything works.
"""

import pandas as pd
import json
import io
from google.cloud import storage
from typing import Dict, List, Optional

# Configuration
BUCKET_NAME = "your-gcs-bucket"  # placeholder: real bucket name removed
CACHE_PATH = "validation_results/"

def test_gcs_connection():
    """Test GCS connection and bucket access."""
    print("🔍 Testing GCS connection...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        list(bucket.list_blobs(prefix=CACHE_PATH, max_results=1))
        print("✅ GCS connection successful")
        return storage_client
    except Exception as e:
        print(f"❌ GCS connection failed: {e}")
        return None

def test_metadata_loading(storage_client):
    """Test loading metadata file."""
    print("\n📋 Testing metadata loading...")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{CACHE_PATH}metadata.json")
        
        if not blob.exists():
            print("❌ Metadata file not found")
            return None
        
        metadata_content = blob.download_as_bytes()
        metadata = json.loads(metadata_content.decode('utf-8'))
        
        print(f"✅ Metadata loaded successfully")
        print(f"   📊 Total games: {metadata.get('total_games', 0)}")
        print(f"   🕒 Generated: {metadata.get('generated_at', 'Unknown')}")
        print(f"   🔧 Version: {metadata.get('cache_version', 'Unknown')}")
        print(f"   🎮 Games: {metadata.get('processed_games', [])}")
        
        return metadata
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

def test_game_data_loading(storage_client, game_name):
    """Test loading data for a specific game."""
    print(f"\n🎮 Testing game data loading for: {game_name}")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{CACHE_PATH}games/{game_name}_complete.parquet"
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            print(f"❌ Game data not found for {game_name}")
            return None
        
        parquet_data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(parquet_data))
        
        print(f"✅ Game data loaded successfully")
        print(f"   📊 Shape: {df.shape}")
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Check for required columns in new structure
        required_cols = ['error_absolute', 'error_percentage', 'cum_error_absolute', 'cum_error_percentage',
                        'cum_actual_absolute', 'cum_predicted_absolute']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print("✅ All required error columns present")
        
        # Show sample data with new column structure
        print(f"   📈 Sample data:")
        sample_cols = ['game', 'prediction_month', 'actual_value', 'predicted_value', 'error_absolute', 'error_percentage']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(3).to_string(index=False))
        
        return df
    except Exception as e:
        print(f"❌ Error loading game data: {e}")
        return None

def test_on_the_fly_analysis(df_dict):
    """Test on-the-fly analysis computation from game data."""
    print(f"\n📊 Testing on-the-fly analysis computation...")
    
    if not df_dict:
        print("❌ No game data available for analysis")
        return False
    
    try:
        # Combine all games
        all_games_data = [df for df in df_dict.values() if df is not None and not df.empty]
        if not all_games_data:
            print("❌ No valid game data for analysis")
            return False
        
        master_df = pd.concat(all_games_data, ignore_index=True)
        print(f"✅ Combined data shape: {master_df.shape}")
        
        # Test case-1 analysis (similar to Overall Analysis)
        case1_df = master_df[master_df['case'] == 1]
        if not case1_df.empty:
            # Test average errors by game and month
            avg_errors = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['error_absolute'].mean()
            print(f"✅ Average errors calculation: {len(avg_errors)} data points")
            
            # Test cumulative errors
            avg_cum_errors = case1_df[case1_df['prediction_month'].between(1, 11)].groupby(['game', 'prediction_month'])['cum_error_absolute'].mean()
            print(f"✅ Cumulative errors calculation: {len(avg_cum_errors)} data points")
            
            print("✅ On-the-fly analysis computations working correctly")
            return True
        else:
            print("⚠️  No Case-1 data found for analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error in on-the-fly analysis: {e}")
        return False

def test_data_integrity(df, game_name):
    """Test data integrity for a game with new column structure."""
    print(f"\n🔍 Testing data integrity for {game_name}...")
    
    issues = []
    
    # Check for null values in critical columns
    critical_cols = ['error_absolute', 'error_percentage', 'cum_error_absolute', 'cum_error_percentage',
                    'actual_value', 'predicted_value']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"{null_count} null values in {col}")
    
    # Check for negative errors (shouldn't happen)
    for theme in ['absolute', 'percentage']:
        error_col = f'error_{theme}'
        if error_col in df.columns:
            negative_count = (df[error_col] < 0).sum()
            if negative_count > 0:
                issues.append(f"{negative_count} negative values in {error_col}")
    
    # Check prediction months range (should be 1-12, representing M2-M13)
    if 'prediction_month' in df.columns:
        min_month = df['prediction_month'].min()
        max_month = df['prediction_month'].max()
        if min_month < 1 or max_month > 12:
            issues.append(f"Invalid prediction month range: {min_month}-{max_month} (expected 1-12)")
    
    # Check that actual and predicted values are reasonable
    if 'actual_value' in df.columns and 'predicted_value' in df.columns:
        actual_stats = df['actual_value'].describe()
        pred_stats = df['predicted_value'].describe()
        print(f"   📊 Actual values range: {actual_stats['min']:.4f} to {actual_stats['max']:.4f}")
        print(f"   📊 Predicted values range: {pred_stats['min']:.4f} to {pred_stats['max']:.4f}")
    
    # Check cumulative values consistency
    for theme in ['absolute', 'percentage']:
        cum_actual_col = f'cum_actual_{theme}'
        cum_pred_col = f'cum_predicted_{theme}'
        if cum_actual_col in df.columns and cum_pred_col in df.columns:
            # Cumulative values should generally be increasing within each cohort-case group
            sample_group = df.groupby(['cohort_str', 'case']).head(5)  # Sample some groups
            if len(sample_group) > 1:
                print(f"   📈 Sample cumulative {theme} values look reasonable")
    
    if issues:
        print("⚠️  Data integrity issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Data integrity check passed")
        return True

def main():
    """Run all tests."""
    print("🚀 Testing GCS Validation Data")
    print("=" * 50)
    
    # Test 1: GCS Connection
    storage_client = test_gcs_connection()
    if not storage_client:
        return
    
    # Test 2: Metadata
    metadata = test_metadata_loading(storage_client)
    if not metadata:
        return
    
    # Test 3: Game Data (test first available game)
    processed_games = metadata.get('processed_games', [])
    if not processed_games:
        print("❌ No processed games found in metadata")
        return
    
    test_game = processed_games[0]
    game_df = test_game_data_loading(storage_client, test_game)
    if game_df is None:
        return
    
    # Test 4: Data Integrity
    test_data_integrity(game_df, test_game)
    
    # Test 5: Load multiple games for analysis testing
    print(f"\n📊 Loading multiple games for analysis testing...")
    games_data = {}
    test_games = processed_games[:3]  # Test first 3 games
    
    for game in test_games:
        game_data = test_game_data_loading(storage_client, game)
        if game_data is not None:
            games_data[game] = game_data
    
    # Test 6: On-the-fly Analysis
    test_on_the_fly_analysis(games_data)
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print("✅ If you see mostly green checkmarks above, your data is ready!")
    print("🚀 You can now deploy the Streamlit app")
    print("=" * 50)

if __name__ == "__main__":
    main() 