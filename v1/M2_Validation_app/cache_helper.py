"""
GCS-based cache helper for M2 Validation Tool

This module provides caching functionality using Google Cloud Storage
to avoid unnecessary recalculations and improve performance.
"""

import pandas as pd
import numpy as np
import json
import io
import tempfile
import os
from datetime import datetime
from google.cloud import storage
from typing import Dict, Any, Optional, Tuple, List
import streamlit as st

# GCS Configuration
BUCKET_NAME = "m1m2_notebook"
CACHE_FOLDER = "cache/predictions/"
CACHE_VERSION = "v2.0"

# Initialize GCS client
try:
    storage_client = storage.Client()
except Exception as e:
    st.error(f"Failed to initialize GCS client: {e}")
    storage_client = None

def test_gcs_connection() -> bool:
    """Test if GCS connection is working."""
    try:
        if storage_client is None:
            return False
        
        bucket = storage_client.bucket(BUCKET_NAME)
        # Test by trying to list blobs
        list(bucket.list_blobs(prefix=CACHE_FOLDER, max_results=1))
        return True
    except Exception as e:
        return False

def get_cached_games() -> List[str]:
    """Get list of games that are already cached."""
    try:
        if storage_client is None:
            return []
        
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # List all prediction files
        blobs = bucket.list_blobs(prefix=CACHE_FOLDER)
        cached_games = []
        
        for blob in blobs:
            if blob.name.endswith('_predictions.parquet'):
                # Extract game name from filename
                game_name = blob.name.split('/')[-1].replace('_predictions.parquet', '')
                cached_games.append(game_name)
        
        return cached_games
    except Exception as e:
        st.error(f"Error getting cached games: {e}")
        return []

def get_missing_games(eligible_games: List[str]) -> List[str]:
    """Get list of games that need to be calculated."""
    cached_games = get_cached_games()
    missing_games = [game for game in eligible_games if game not in cached_games]
    return missing_games

def save_single_game_cache(game_name: str, predictions_df: pd.DataFrame, theme: str = "Absolute Error") -> bool:
    """Save predictions for a single game to GCS."""
    try:
        if storage_client is None:
            return False
        
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Save predictions as parquet
        blob_name = f"{CACHE_FOLDER}{game_name}_predictions.parquet"
        blob = bucket.blob(blob_name)
        
        # Convert DataFrame to parquet bytes
        parquet_buffer = io.BytesIO()
        predictions_df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        blob.upload_from_file(parquet_buffer, content_type='application/octet-stream')
        
        # Update metadata
        update_metadata_for_game(game_name, theme)
        
        return True
    except Exception as e:
        st.error(f"Error saving cache for {game_name}: {e}")
        return False

def load_single_game_cache(game_name: str) -> Optional[pd.DataFrame]:
    """Load predictions for a single game from GCS."""
    try:
        if storage_client is None:
            return None
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{CACHE_FOLDER}{game_name}_predictions.parquet"
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            return None
        
        # Download and load parquet
        parquet_data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(parquet_data))
        
        # Convert cohort back to datetime if needed
        if 'cohort' in df.columns:
            df['cohort'] = pd.to_datetime(df['cohort'])
        
        return df
    except Exception as e:
        st.error(f"Error loading cache for {game_name}: {e}")
        return None

def update_metadata_for_game(game_name: str, theme: str) -> bool:
    """Update metadata when a game is cached."""
    try:
        if storage_client is None:
            return False
        
        bucket = storage_client.bucket(BUCKET_NAME)
        metadata_blob = bucket.blob(f"{CACHE_FOLDER}metadata.json")
        
        # Load existing metadata or create new
        if metadata_blob.exists():
            metadata_content = metadata_blob.download_as_bytes()
            metadata = json.loads(metadata_content.decode('utf-8'))
        else:
            metadata = {
                "games": [],
                "theme": theme,
                "cache_version": CACHE_VERSION,
                "last_updated": datetime.now().isoformat()
            }
        
        # Add game to metadata if not already there
        if game_name not in metadata["games"]:
            metadata["games"].append(game_name)
        
        # Update metadata
        metadata["theme"] = theme
        metadata["last_updated"] = datetime.now().isoformat()
        metadata[f"{game_name}_timestamp"] = datetime.now().isoformat()
        
        # Save updated metadata
        metadata_json = json.dumps(metadata, indent=2)
        metadata_blob.upload_from_string(metadata_json, content_type='application/json')
        
        return True
    except Exception as e:
        st.error(f"Error updating metadata for {game_name}: {e}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """Get comprehensive cache information."""
    try:
        if storage_client is None:
            return {"predictions_cached": False, "predictions_games_count": 0}
        
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Check metadata
        metadata_blob = bucket.blob(f"{CACHE_FOLDER}metadata.json")
        if metadata_blob.exists():
            metadata_content = metadata_blob.download_as_bytes()
            metadata = json.loads(metadata_content.decode('utf-8'))
            
            return {
                "predictions_cached": True,
                "predictions_games_count": len(metadata.get("games", [])),
                "predictions_timestamp": metadata.get("last_updated"),
                "cache_version": metadata.get("cache_version", "v1.0"),
                "theme": metadata.get("theme", "Absolute Error"),
                "games": metadata.get("games", [])
            }
        else:
            return {"predictions_cached": False, "predictions_games_count": 0}
    except Exception as e:
        return {"predictions_cached": False, "predictions_games_count": 0}

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        if storage_client is None:
            return {}
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=CACHE_FOLDER))
        
        total_size = sum(blob.size for blob in blobs if blob.size)
        size_mb = round(total_size / (1024 * 1024), 2) if total_size > 0 else 0
        
        return {
            "total_files": len(blobs),
            "size_mb": size_mb
        }
    except Exception as e:
        return {}

def load_all_cached_games() -> Dict[str, pd.DataFrame]:
    """Load all cached games."""
    cached_games = get_cached_games()
    predictions_dict = {}
    
    for game_name in cached_games:
        df = load_single_game_cache(game_name)
        if df is not None:
            predictions_dict[game_name] = df
    
    return predictions_dict

def clear_cache() -> bool:
    """Clear all cache files from GCS."""
    try:
        if storage_client is None:
            return False
        
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # List all blobs in cache folder
        blobs = list(bucket.list_blobs(prefix=CACHE_FOLDER))
        
        # Delete all cache files
        for blob in blobs:
            blob.delete()
        
        return True
    except Exception as e:
        st.error(f"Error clearing cache: {e}")
        return False

def clear_single_game_cache(game_name: str) -> bool:
    """Clear cache for a single game."""
    try:
        if storage_client is None:
            return False
        
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Delete game prediction file
        blob_name = f"{CACHE_FOLDER}{game_name}_predictions.parquet"
        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.delete()
        
        # Update metadata to remove game
        metadata_blob = bucket.blob(f"{CACHE_FOLDER}metadata.json")
        if metadata_blob.exists():
            metadata_content = metadata_blob.download_as_bytes()
            metadata = json.loads(metadata_content.decode('utf-8'))
            
            if game_name in metadata.get("games", []):
                metadata["games"].remove(game_name)
            
            # Remove game timestamp
            if f"{game_name}_timestamp" in metadata:
                del metadata[f"{game_name}_timestamp"]
            
            # Save updated metadata
            metadata_json = json.dumps(metadata, indent=2)
            metadata_blob.upload_from_string(metadata_json, content_type='application/json')
        
        return True
    except Exception as e:
        st.error(f"Error clearing cache for {game_name}: {e}")
        return False

def calculate_error(actual, predicted, theme="Absolute Error"):
    """Calculate error based on the selected theme."""
    if pd.isna(actual) or pd.isna(predicted):
        return np.nan
    
    if theme == "Absolute Error":
        return abs(actual - predicted)
    elif theme == "Percentage Error":
        if actual == 0:
            return np.nan
        return abs(actual - predicted) * 100 / abs(actual)
    else:
        return abs(actual - predicted)

def calculate_cumulative_error(cum_actual, cum_predicted, theme="Absolute Error"):
    """Calculate cumulative error based on the selected theme."""
    if pd.isna(cum_actual) or pd.isna(cum_predicted):
        return np.nan
    
    if theme == "Absolute Error":
        return abs(cum_actual - cum_predicted)
    elif theme == "Percentage Error":
        if cum_actual == 0:
            return np.nan
        return abs(cum_actual - cum_predicted) * 100 / abs(cum_actual)
    else:
        return abs(cum_actual - cum_predicted)

def calculate_cumulative_error_with_max(cum_actual, cum_predicted, theme="Absolute Error"):
    """Calculate cumulative error using max(actual, predicted) for percentage calculations."""
    if pd.isna(cum_actual) or pd.isna(cum_predicted):
        return np.nan
    
    if theme == "Absolute Error":
        return abs(cum_actual - cum_predicted)
    elif theme == "Percentage Error":
        denominator = max(abs(cum_actual), abs(cum_predicted))
        if denominator == 0:
            return np.nan
        return abs(cum_actual - cum_predicted) * 100 / denominator
    else:
        return abs(cum_actual - cum_predicted)

def calculate_cumulative(results_df, theme="Absolute Error"):
    """Calculate cumulative values and errors."""
    if results_df is None or results_df.empty:
        return results_df
    
    results_df = results_df.sort_values(by=['cohort_str', 'case', 'prediction_month'])
    
    # Group by cohort and case to calculate cumulative values
    def calculate_cumulative_for_group(group):
        # Get the game, cohort info to fetch M1 data
        game = group['game'].iloc[0]
        cohort_str = group['cohort_str'].iloc[0]
        cohort_dt = group['cohort'].iloc[0]
        case = group['case'].iloc[0]
        
        # We need to get M1 actual value from the original data
        # M1 = elapsed_months=0, M2 = elapsed_months=1, etc.
        try:
            # Import here to avoid circular imports
            from M2_Validation_tool import preprocess_data
            
            # Get the original data to find M1 actual value
            df = preprocess_data(game)
            if df is not None:
                cohort_data = df[df['cohort'] == cohort_dt].copy()
                if not cohort_data.empty:
                    # Get M1 actual value (elapsed_months = 0)
                    m1_actual_series = cohort_data[cohort_data['elapsed_months'] == 0]['transaction_period_ltv_cac']
                    m1_actual = float(m1_actual_series.iloc[0]) if not m1_actual_series.empty else 0.0
                    
                    # Calculate cumulative values - add M1 actual to both actual and predicted
                    group = group.copy()
                    group['cum_actual'] = m1_actual + group['actual_value'].cumsum()
                    group['cum_predicted'] = m1_actual + group['predicted_value'].cumsum()
                    
                    # Calculate cumulative error using max(actual, predicted) for percentage
                    if theme == "Percentage Error":
                        group['cum_error'] = group.apply(
                            lambda row: calculate_cumulative_error_with_max(row['cum_actual'], row['cum_predicted'], theme), 
                            axis=1
                        )
                    else:
                        group['cum_error'] = group.apply(
                            lambda row: calculate_cumulative_error(row['cum_actual'], row['cum_predicted'], theme), 
                            axis=1
                        )
                    return group
        except Exception as e:
            # Fallback to original calculation if there's an error
            pass
        
        # Fallback to original calculation (without M1)
        group = group.copy()
        group['cum_actual'] = group['actual_value'].cumsum()
        group['cum_predicted'] = group['predicted_value'].cumsum()
        
        # Calculate cumulative error
        if theme == "Percentage Error":
            group['cum_error'] = group.apply(
                lambda row: calculate_cumulative_error_with_max(row['cum_actual'], row['cum_predicted'], theme), 
                axis=1
            )
        else:
            group['cum_error'] = group.apply(
                lambda row: calculate_cumulative_error(row['cum_actual'], row['cum_predicted'], theme), 
                axis=1
            )
        return group
    
    # Apply the cumulative calculation to each group
    results_df = results_df.groupby(['cohort_str', 'case']).apply(calculate_cumulative_for_group).reset_index(drop=True)
    
    return results_df

def recalculate_errors_only(predictions_dict: Dict[str, pd.DataFrame], new_theme: str) -> Dict[str, pd.DataFrame]:
    """Recalculate only error columns without redoing predictions."""
    updated_dict = {}
    for game_name, df in predictions_dict.items():
        if df is not None and not df.empty:
            # Recalculate error columns
            df_copy = df.copy()
            
            # Recalculate individual errors
            df_copy['error'] = df_copy.apply(
                lambda row: calculate_error(row['actual_value'], row['predicted_value'], new_theme),
                axis=1
            )
            
            # Recalculate cumulative values and errors with the new theme
            df_copy = calculate_cumulative(df_copy, new_theme)
            
            updated_dict[game_name] = df_copy
        else:
            updated_dict[game_name] = df
    
    return updated_dict

def process_missing_games_incrementally(eligible_games: List[str], current_theme: str, 
                                       get_prediction_func, progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Process missing games incrementally, saving each one as it's calculated."""
    
    # Get missing games
    missing_games = get_missing_games(eligible_games)
    
    # Load already cached games
    all_predictions = load_all_cached_games()
    
    if not missing_games:
        # All games are cached, just return them
        if progress_callback:
            progress_callback(f"✅ All {len(eligible_games)} games already cached!")
        return all_predictions
    
    # Process missing games one by one
    total_missing = len(missing_games)
    
    for i, game_name in enumerate(missing_games):
        if progress_callback:
            progress_callback(f"⚙️ Processing {game_name} ({i+1}/{total_missing})")
        
        try:
            # Get predictions for this game
            predictions = get_prediction_func(game_name)
            
            if predictions is not None and not predictions.empty:
                # Save immediately to cache
                if save_single_game_cache(game_name, predictions, current_theme):
                    all_predictions[game_name] = predictions
                    if progress_callback:
                        progress_callback(f"✅ Saved {game_name} to cache")
                else:
                    if progress_callback:
                        progress_callback(f"❌ Failed to save {game_name}")
            else:
                if progress_callback:
                    progress_callback(f"❌ No data for {game_name}")
                    
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {game_name}: {str(e)}")
            continue
    
    return all_predictions 