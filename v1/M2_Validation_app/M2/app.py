import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
import os
import json

# Import custom modules
from models import save_all_models, ModelFactory, load_specific_model, get_available_models_for_game
from preprocessing import preprocess_m1, preprocess_m2
from utils import (
    auto_select_features, get_available_games, prepare_training_data_with_holdout,
    load_config, save_config
)

def main():
    st.set_page_config(page_title="Auto Feature Selection with Holdout Validation", layout="wide")
    
    st.title("🤖 Auto Feature Selection with 6-Month Holdout Validation")
    st.markdown("Automatically select best features, train multiple models, and save ALL models with holdout validation strategy!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model type selection
        model_type = st.radio("Select Model Type", options=["M1", "M2"])
        
        # Model selection for training
        available_models = list(ModelFactory.get_model_configs().keys())
        selected_models = st.multiselect(
            "Select Models to Train for All Games", 
            available_models,
            default=["xgboost", "lightgbm", "catboost", "random_forest", "ridge"]
        )
        
        if not selected_models:
            st.warning("Please select at least one model to train.")
            return
        
        # Model selection for analysis/prediction (if models exist)
        st.subheader("🔍 Model Selection for Analysis")
        sample_game = get_available_games()[0] if get_available_games() else None
        if sample_game:
            available_model_types = get_available_models_for_game(sample_game, f"_{model_type}")
            if available_model_types:
                selected_analysis_model = st.selectbox(
                    "Select Model for Analysis/Prediction",
                    available_model_types,
                    help="Choose which trained model to use for analysis and predictions"
                )
                st.session_state['selected_analysis_model'] = selected_analysis_model
            else:
                st.info("No trained models found. Train models first.")
        else:
            st.info("No games available.")
    
    # Main content area
    st.markdown("""
    ### What this app does:
    - 🔍 **Automatically select best features** for each game using XGBoost feature importance
    - 🏋️ **Train multiple models** for each game (XGBoost, LightGBM, CatBoost, Random Forest, Ridge)
    - 💾 **Save ALL trained models** (not just the best one) with descriptive naming
    - 🎯 **6-Month Holdout Strategy**: Reserve last 6 months for validation (minimum 6 months training)
    - ⚙️ **Update configuration automatically** with training results and holdout information
    - 📊 **Generate performance summary** with visualizations and holdout metrics
    - 🔄 **Model Selection**: Choose any trained model for analysis via sidebar dropdown
    
    ### 🎯 Holdout Validation Strategy:
    - **Games with ≥8 months**: Train on first N-6 months, holdout last 6 months
    - **Games with <8 months**: Use all available data for training (no holdout)
    - **Minimum training**: At least 6 months required for reliable predictions
    """)
    
    games = get_available_games()
    st.info(f"📊 Found **{len(games)} games** available for processing")
    st.info(f"🎯 Will train **{len(selected_models)} models** per game and save ALL models")
    st.info(f"🎯 6-month holdout strategy: Reserve last 6 months for validation when possible")
    st.info(f"⚡ Total training tasks: **{len(games) * len(selected_models)}** (All models will be saved)")
    
    # Show available games in an expandable section
    with st.expander("📋 View Available Games"):
        cols = st.columns(5)
        for i, game in enumerate(games):
            cols[i % 5].write(f"• {game}")
    
    if st.button("🚀 Start Auto Feature Selection for All Games", type="primary", use_container_width=True):
        run_auto_feature_selection_all(model_type, games, selected_models)

def run_auto_feature_selection_all(model_type, games, selected_models):
    """Run automated feature selection and training for all games"""
    st.header("🔄 Processing All Games...")
    
    # Progress tracking
    total_tasks = len(games) * len(selected_models)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results storage
    all_results = {}
    failed_games = []
    config = load_config()
    
    task_count = 0
    
    for game_idx, game_id in enumerate(games):
        status_text.text(f"🎮 Processing game {game_id} ({game_idx + 1}/{len(games)})")
        
        try:
            # Load and preprocess data with holdout logic
            if model_type == "M1":
                training_df, default_features, holdout_df, max_elapsed = preprocess_m1(game_id, use_holdout=True)
                training_dft = None
                holdout_dft = None
            else:
                training_dft, default_features, df, holdout_dft, max_elapsed = preprocess_m2(game_id, use_holdout=True)
                training_df = df  # Full dataset for potential use
            
            # Display holdout information
            holdout_months = 6 if max_elapsed >= 8 else 0
            training_months = max_elapsed - holdout_months + 1 if holdout_months > 0 else max_elapsed + 1
            st.info(f"🎯 Game {game_id}: {training_months} months training, {holdout_months} months holdout (Total: {max_elapsed + 1} months)")
            
            # Automated feature selection on training data
            data = training_df if model_type == "M1" else training_dft
            selected_features = auto_select_features(model_type, game_id, data, default_features)
            
            # Prepare training data
            X, y = prepare_training_data_with_holdout(model_type, training_df, training_dft, selected_features)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models for this game
            game_results = {}
            for model_name in selected_models:
                status_text.text(f"🏋️ Training {model_name} for {game_id}...")
                
                try:
                    from models import train_and_evaluate_model
                    result = train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, n_iter=50)
                    
                    if result:
                        game_results[model_name] = result
                
                except Exception as e:
                    st.warning(f"⚠️ Failed to train {model_name} for {game_id}: {str(e)}")
                
                task_count += 1
                progress_bar.progress(task_count / total_tasks)
            
            if game_results:
                all_results[game_id] = game_results
                
                # Save all models for this game
                save_result = save_all_models(game_id, game_results, f"_{model_type}")
                
                # Update configuration
                config_key = f"{model_type}_{game_id}"
                best_model_name = min(game_results.keys(), key=lambda x: game_results[x]['rmse_test'])
                
                config_entry = {
                    "model_type": model_type,
                    "game_id": game_id,
                    "selected_features": selected_features,
                    "trained_models": list(game_results.keys()),
                    "best_model": best_model_name,
                    "best_rmse": game_results[best_model_name]['rmse_test'],
                    "all_model_rmse": {model: results['rmse_test'] for model, results in game_results.items()},
                    "total_models_saved": save_result['total_models_saved'] if save_result else 0,
                    "training_months": training_months,
                    "holdout_months": holdout_months,
                    "max_elapsed_months": max_elapsed,
                    "flag": 1,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                config[config_key] = config_entry
                
            else:
                failed_games.append(game_id)
                
        except Exception as e:
            st.error(f"❌ Failed to process game {game_id}: {str(e)}")
            failed_games.append(game_id)
            task_count += len(selected_models)
            progress_bar.progress(task_count / total_tasks)
    
    # Save configuration
    if save_config(config):
        st.success("✅ Configuration saved successfully!")
    
    # Display results summary
    status_text.text("📊 Generating summary results...")
    display_auto_selection_results(all_results, failed_games, model_type, selected_models)
    
    status_text.text("✅ Auto feature selection for all games completed!")

def display_auto_selection_results(all_results, failed_games, model_type, selected_models):
    """Display results from auto feature selection for all games"""
    st.header("🎉 Auto Feature Selection Results")
    
    if all_results:
        # Success summary
        st.success(f"✅ Successfully processed **{len(all_results)}** games")
        
        # Load config to get holdout information
        config = load_config()
        
        # Create summary table with holdout information
        summary_data = []
        for game_id, model_results in all_results.items():
            best_model = min(model_results.keys(), key=lambda x: model_results[x]['rmse_test'])
            best_rmse = model_results[best_model]['rmse_test']
            
            # Get holdout info from config
            config_key = f"{model_type}_{game_id}"
            config_entry = config.get(config_key, {})
            training_months = config_entry.get('training_months', 'N/A')
            holdout_months = config_entry.get('holdout_months', 'N/A')
            total_models_saved = config_entry.get('total_models_saved', len(model_results))
            
            summary_data.append({
                'Game': game_id,
                'Best_Model': best_model,
                'Best_RMSE': best_rmse,
                'Models_Trained': len(model_results),
                'Models_Saved': total_models_saved,
                'Training_Months': training_months,
                'Holdout_Months': holdout_months
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Best_RMSE')
        
        # Display summary table
        st.subheader("📊 Performance Summary with Holdout Strategy")
        st.dataframe(
            summary_df.style.format({
                'Best_RMSE': '{:.4f}',
                'Training_Months': '{}',
                'Holdout_Months': '{}'
            }).highlight_min(subset=['Best_RMSE'], color='lightgreen'),
            use_container_width=True
        )
        
        # Holdout strategy summary
        st.subheader("🎯 Holdout Strategy Summary")
        if not summary_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                games_with_holdout = len(summary_df[summary_df['Holdout_Months'] > 0])
                st.metric("Games with 6-month Holdout", games_with_holdout)
                
            with col2:
                games_without_holdout = len(summary_df[summary_df['Holdout_Months'] == 0])
                st.metric("Games using All Data", games_without_holdout)
                
            with col3:
                avg_training_months = summary_df[summary_df['Training_Months'] != 'N/A']['Training_Months'].mean()
                st.metric("Avg Training Months", f"{avg_training_months:.1f}" if not pd.isna(avg_training_months) else "N/A")
        
        # Model distribution and statistics
        model_counts = summary_df['Best_Model'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Model Distribution")
            fig = px.pie(
                values=model_counts.values,
                names=model_counts.index,
                title="Which Models Won Most Games",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Performance Statistics")
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Average RMSE", f"{summary_df['Best_RMSE'].mean():.4f}")
                st.metric("Best RMSE", f"{summary_df['Best_RMSE'].min():.4f}")
                st.metric("Total Models Saved", f"{summary_df['Models_Saved'].sum()}")
            with col2_2:
                st.metric("Worst RMSE", f"{summary_df['Best_RMSE'].max():.4f}")
                st.metric("Total Games", f"{len(summary_df)}")
                st.metric("Avg Models per Game", f"{summary_df['Models_Saved'].mean():.1f}")
    
    # Failed games
    if failed_games:
        st.subheader("⚠️ Failed Games")
        st.error(f"Failed to process **{len(failed_games)}** games:")
        
        failed_cols = st.columns(5)
        for i, game in enumerate(failed_games):
            failed_cols[i % 5].write(f"❌ {game}")

if __name__ == "__main__":
    main() 