import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from utils import format_percentage, format_percentage_direct

def calculate_case1_errors_m2(game_id, best_model_expected, df, forecast_months=12):
    """
    Calculate prediction errors for Case 1 (starting predictions from month 1)
    for M2 models using the last 12 cohorts similar to M2_validation_app.py
    """
    # Get the last 12 cohorts similar to M2_validation_app
    latest_cohorts = sorted(df['cohort'].unique())[-12:]
    results = []
    epsilon = 0.001
    
    for cohort_dt_obj in latest_cohorts:
        cohort_data = df[df['cohort'] == cohort_dt_obj].copy()
        if cohort_data.empty:
            continue
            
        max_elapsed_month = int(cohort_data['elapsed_months'].max())
        
        # Case 1: Start predicting from month 1 - FIXED to match M2_validation_app.py
        # In M2_validation_app: start_month = 0, case_num = start_month + 1 = 1
        start_month = 0
        case_num = start_month + 1  # This gives case_num = 1
        
        # For M2, we need data from elapsed_months = 0 to start (i.e., no historical data beyond month 0)
        actual_data = cohort_data[cohort_data['elapsed_months'] <= start_month].copy()
        if actual_data.empty:
            continue
        
        # Get static features from the cohort data
        feature_names = best_model_expected.__dict__["feature_names_in_"]
        
        try:
            static_features = cohort_data.loc[
                cohort_data['elapsed_months'] == max_elapsed_month, 
                feature_names
            ].iloc[0].copy()
        except Exception as e:
            continue
        
        # Remove dynamic features that change with elapsed_months
        for col in ['elapsed_months', 'current_month', 'ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3']:
            if col in static_features:
                static_features = static_features.drop(col)
        
        # Initialize lag values for Case 1 (starting from month 0)
        lag_1 = np.nan
        lag_2 = np.nan  
        lag_3 = np.nan
        
        # Initialize lag values based on available historical data (for start_month = 0, no historical data)
        for lag in range(1, 4):
            lag_hist_month = start_month - lag
            if lag_hist_month >= 0:
                lag_val_series = cohort_data[cohort_data['elapsed_months'] == lag_hist_month]['transaction_period_ltv_cac']
                if lag == 1:
                    lag_1 = lag_val_series.iloc[0] if not lag_val_series.empty else 0.0
                elif lag == 2:
                    lag_2 = lag_val_series.iloc[0] if not lag_val_series.empty else 0.0
                elif lag == 3:
                    lag_3 = lag_val_series.iloc[0] if not lag_val_series.empty else 0.0
            else:
                # No historical data available for this lag
                if lag == 1:
                    lag_1 = 0.0
                elif lag == 2:
                    lag_2 = 0.0
                elif lag == 3:
                    lag_3 = 0.0
        
        # Generate predictions for all months starting from month 1 (elapsed_months = 1)
        # This matches M2_validation_app.py: range(start_month + 1, max_elapsed_month + 1)
        for pred_elapsed_month in range(start_month + 1, min(max_elapsed_month + 1, forecast_months + 1)):
            features = static_features.copy()
            features['elapsed_months'] = pred_elapsed_month
            
            # Calculate current month
            if 'current_month' in feature_names:
                cohort_start_date = pd.to_datetime(cohort_dt_obj)
                current_date = cohort_start_date + pd.DateOffset(months=pred_elapsed_month)
                features['current_month'] = current_date.month
            
            # Set lag features - match M2_validation_app logic
            for lag in range(1, 4):
                lag_col = f'ltv_lag_{lag}'
                source_elapsed_month_for_lag = pred_elapsed_month - lag
                
                if source_elapsed_month_for_lag <= start_month and source_elapsed_month_for_lag >= 0:
                    # Use actual historical data for lag features
                    actual_lag_val_series = cohort_data[cohort_data['elapsed_months'] == source_elapsed_month_for_lag]['transaction_period_ltv_cac']
                    features[lag_col] = actual_lag_val_series.iloc[0] if not actual_lag_val_series.empty else 0.0
                else:
                    # Use predicted values or defaults for lag features
                    if lag == 1:
                        features[lag_col] = lag_1 if pd.notna(lag_1) else 0.0
                    elif lag == 2:
                        features[lag_col] = lag_2 if pd.notna(lag_2) else 0.0
                    elif lag == 3:
                        features[lag_col] = lag_3 if pd.notna(lag_3) else 0.0
            
            # Ensure all required features are present
            for col in feature_names:
                if col not in features:
                    features[col] = 0.0
            
            try:
                features_df = pd.DataFrame([features[feature_names]])
                prediction = float(best_model_expected.predict(features_df)[0])
                prediction = max(0, prediction)  # Ensure non-negative
            except Exception as e:
                prediction = 0.0
            
            # Get actual value
            actual_val_series = cohort_data[cohort_data['elapsed_months'] == pred_elapsed_month]['transaction_period_ltv_cac']
            actual_val = float(actual_val_series.iloc[0]) if not actual_val_series.empty else np.nan
            
            # Calculate absolute error
            if pd.notna(actual_val):
                error = abs(prediction - actual_val)
            else:
                error = np.nan
            
            # Calculate percentage error: |actual - predicted| * 100 / |actual|
            if pd.notna(actual_val) and actual_val != 0:
                perc_error = abs(prediction - actual_val) * 100 / abs(actual_val)
            else:
                perc_error = np.nan
            
            results.append({
                'cohort': cohort_dt_obj,
                'cohort_str': cohort_dt_obj.strftime('%Y-%m-%d'),
                'case': case_num,
                'prediction_month': pred_elapsed_month,  # Use actual elapsed_months we're predicting
                'actual_value': actual_val,
                'predicted_value': prediction,
                'error': error,
                'perc_error': perc_error
            })
            
            # Update lag values for next prediction - match M2_validation_app logic
            lag_3 = lag_2
            lag_2 = lag_1
            lag_1 = prediction
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cohort'] = pd.to_datetime(results_df['cohort'])
        
        # Calculate cumulative values
        results_df = results_df.sort_values(by=['cohort_str', 'case', 'prediction_month'])
        results_df['cum_actual'] = results_df.groupby(['cohort_str', 'case'])['actual_value'].cumsum()
        results_df['cum_predicted'] = results_df.groupby(['cohort_str', 'case'])['predicted_value'].cumsum()
        
        # Calculate cumulative error as absolute difference
        results_df['cum_error'] = results_df.apply(
            lambda row: abs(row['cum_predicted'] - row['cum_actual']) 
            if pd.notna(row['cum_actual']) and pd.notna(row['cum_predicted'])
            else np.nan, 
            axis=1
        )
        
        # Calculate cumulative percentage error: |cum_actual - cum_predicted| * 100 / |cum_actual|
        results_df['cum_perc_error'] = results_df.apply(
            lambda row: abs(row['cum_predicted'] - row['cum_actual']) * 100 / abs(row['cum_actual'])
            if pd.notna(row['cum_actual']) and pd.notna(row['cum_predicted']) and row['cum_actual'] != 0
            else np.nan, 
            axis=1
        )
    
    return results_df

def display_case1_error_table(results_df, game_id):
    """Display the Case 1 error table exactly like M2_validation_app"""
    if results_df.empty:
        st.write("No data available for Case 1 error table.")
        return
    
    st.write("---")
    st.subheader(f"Overall Average Error Table for {game_id} (Case 1)")
    
    # Filter for Case 1 only and create the exact same logic as M2_validation_app
    case1_data = results_df[results_df['case'] == 1].copy()
    
    if case1_data.empty:
        st.write("No Case 1 data available.")
        return
    
    # Create tables exactly like M2_validation_app.py does for individual games
    game_avg_errors = case1_data.groupby(['case', 'prediction_month'])['error'].mean().reset_index()
    game_avg_errors_pivot = game_avg_errors.pivot(index='case', columns='prediction_month', values='error')
    # Fixed column naming to match M2_validation_app.py: M{pred_elapsed_month + 1}_error
    game_avg_errors_pivot.columns = [f"M{int(col)+1}_error" for col in game_avg_errors_pivot.columns]
    
    # Add No. of Cohorts column for Case 1
    total_cohorts_game = case1_data['cohort'].nunique()
    game_avg_errors_pivot['No. of Cohorts'] = [total_cohorts_game]
    
    # Display Error Table
    st.write("**Error Table (LTV/CAC)**")
    st.dataframe(game_avg_errors_pivot.style.format({
        **{col: format_percentage for col in game_avg_errors_pivot.columns if 'error' in col},
        'No. of Cohorts': '{:.0f}'
    }))
    
    # Create cumulative error table
    game_avg_cum_errors = case1_data.groupby(['case', 'prediction_month'])['cum_error'].mean().reset_index()
    game_avg_cum_errors_pivot = game_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_error')
    # Fixed column naming to match M2_validation_app.py: M{pred_elapsed_month + 1}_cum_error
    game_avg_cum_errors_pivot.columns = [f"M{int(col)+1}_cum_error" for col in game_avg_cum_errors_pivot.columns]
    
    # Add No. of Cohorts column for Case 1
    total_cohorts_game_cum = case1_data['cohort'].nunique()
    game_avg_cum_errors_pivot['No. of Cohorts'] = [total_cohorts_game_cum]
    
    # Display Cumulative Error Table
    st.write("**Cumulative Error Table (LTV/CAC)**")
    st.dataframe(game_avg_cum_errors_pivot.style.format({
        **{col: format_percentage for col in game_avg_cum_errors_pivot.columns if 'error' in col},
        'No. of Cohorts': '{:.0f}'
    }))
    
    # Create percentage error table
    game_avg_perc_errors = case1_data.groupby(['case', 'prediction_month'])['perc_error'].mean().reset_index()
    game_avg_perc_errors_pivot = game_avg_perc_errors.pivot(index='case', columns='prediction_month', values='perc_error')
    game_avg_perc_errors_pivot.columns = [f"M{int(col)+1}_perc_error" for col in game_avg_perc_errors_pivot.columns]
    
    # Add No. of Cohorts column for percentage error table
    game_avg_perc_errors_pivot['No. of Cohorts'] = [total_cohorts_game]
    
    # Display Percentage Error Table
    st.write("**Percentage Error Table (LTV/CAC)**")
    st.dataframe(game_avg_perc_errors_pivot.style.format({
        **{col: format_percentage_direct for col in game_avg_perc_errors_pivot.columns if 'perc_error' in col},
        'No. of Cohorts': '{:.0f}'
    }))
    
    # Create cumulative percentage error table
    game_avg_cum_perc_errors = case1_data.groupby(['case', 'prediction_month'])['cum_perc_error'].mean().reset_index()
    game_avg_cum_perc_errors_pivot = game_avg_cum_errors.pivot(index='case', columns='prediction_month', values='cum_perc_error')
    game_avg_cum_perc_errors_pivot.columns = [f"M{int(col)+1}_cum_perc_error" for col in game_avg_cum_perc_errors_pivot.columns]
    
    # Add No. of Cohorts column for cumulative percentage error table
    game_avg_cum_perc_errors_pivot['No. of Cohorts'] = [total_cohorts_game]
    
    # Display Cumulative Percentage Error Table
    st.write("**Cumulative Percentage Error Table (LTV/CAC)**")
    st.dataframe(game_avg_cum_perc_errors_pivot.style.format({
        **{col: format_percentage_direct for col in game_avg_cum_perc_errors_pivot.columns if 'perc_error' in col},
        'No. of Cohorts': '{:.0f}'
    }))

def create_rmse_heatmap(results_dict):
    """Create heatmap showing RMSE for all models across all games"""
    # Prepare data for heatmap
    heatmap_data = []
    for game_id, model_results in results_dict.items():
        for model_type, result in model_results.items():
            heatmap_data.append({
                'Game': game_id,
                'Model': model_type,
                'RMSE_Test': result['rmse_test'],
                'RMSE_Train': result['rmse_train']
            })
    
    if not heatmap_data:
        st.write("No data available for heatmap.")
        return None
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Create pivot table for test RMSE
    pivot_test = df_heatmap.pivot(index='Game', columns='Model', values='RMSE_Test')
    
    # Create the heatmap using plotly
    fig = px.imshow(
        pivot_test,
        labels=dict(x="Model Type", y="Game ID", color="RMSE Test"),
        x=pivot_test.columns,
        y=pivot_test.index,
        color_continuous_scale='RdYlBu_r',
        title="Model Performance Heatmap (Test RMSE)"
    )
    
    # Add text annotations
    for i, game in enumerate(pivot_test.index):
        for j, model in enumerate(pivot_test.columns):
            value = pivot_test.loc[game, model]
            if pd.notna(value):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{value:.3f}",
                    showarrow=False,
                    font=dict(color="white" if value > pivot_test.values.max()/2 else "black")
                )
    
    fig.update_layout(height=800, width=1000)
    st.plotly_chart(fig, use_container_width=True)
    
    return df_heatmap

def create_best_model_table(results_dict):
    """Create table showing which model is best for each game"""
    best_models = []
    
    for game_id, model_results in results_dict.items():
        if model_results:
            # Find best model based on test RMSE
            best_model = min(model_results.keys(), key=lambda x: model_results[x]['rmse_test'])
            best_rmse = model_results[best_model]['rmse_test']
            
            best_models.append({
                'Game_ID': game_id,
                'Best_Model': best_model,
                'Best_RMSE_Test': best_rmse,
                'Best_RMSE_Train': model_results[best_model]['rmse_train']
            })
    
    if not best_models:
        st.write("No data available for best model table.")
        return None
    
    df_best = pd.DataFrame(best_models)
    df_best = df_best.sort_values('Best_RMSE_Test')
    
    st.subheader("Best Model for Each Game")
    st.dataframe(
        df_best.style.format({
            'Best_RMSE_Test': '{:.4f}',
            'Best_RMSE_Train': '{:.4f}'
        }).highlight_min(subset=['Best_RMSE_Test'], color='lightgreen')
    )
    
    # Create summary statistics
    model_counts = df_best['Best_Model'].value_counts()
    st.subheader("Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Usage Count:**")
        st.dataframe(model_counts.reset_index().rename(columns={'index': 'Model', 'Best_Model': 'Games_Won'}))
    
    with col2:
        # Create pie chart
        fig = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="Distribution of Best Models"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    return df_best

def compare_model_performance(results_dict):
    """Compare performance across models and games"""
    if not results_dict:
        st.write("No results to compare.")
        return
    
    # Aggregate results across all games
    all_results = []
    for game_id, model_results in results_dict.items():
        for model_type, result in model_results.items():
            all_results.append({
                'Game': game_id,
                'Model': model_type,
                'RMSE_Test': result['rmse_test'],
                'RMSE_Train': result['rmse_train']
            })
    
    df_all = pd.DataFrame(all_results)
    
    # Box plot comparing model performance
    fig = px.box(
        df_all, 
        x='Model', 
        y='RMSE_Test',
        title="Model Performance Distribution (Test RMSE)",
        points="all"
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    summary_stats = df_all.groupby('Model')['RMSE_Test'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    summary_stats.columns = ['Mean_RMSE', 'Std_RMSE', 'Min_RMSE', 'Max_RMSE', 'Games_Tested']
    st.dataframe(summary_stats)
    
    return df_all 