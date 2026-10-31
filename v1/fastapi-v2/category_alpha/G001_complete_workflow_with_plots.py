#!/usr/bin/env python3
"""
G001 Complete Workflow: Alpha Projections + Forecasting + Comprehensive Plotting

This script provides the complete end-to-end workflow for:
1. Alpha-based cohort projections
2. XGBoost forecasting 
3. Comprehensive interactive plotting and analysis

All functionality from the Jupyter notebook in a standalone, reliable script.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Plotting libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# For model loading
import cloudpickle
import os

print("🎮 G001 Complete Workflow: Alpha + Forecast + Plots")
print("=" * 70)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_g001_data():
    """Load and prepare G001 dataset"""
    print("\n📊 Step 1: Loading G001 Data...")
    
    try:
        g001_data = pd.read_csv("Latest Data/Segregated/G001_Dataset_12_2024.csv")
        print(f"✅ G001 data loaded successfully!")
        print(f"Dataset shape: {g001_data.shape}")
        
        # Filter to elapsed_months = 1
        g001_month1 = g001_data[g001_data['elapsed_months'] == 1].copy()
        print(f"📈 Elapsed months = 1 data: {g001_month1.shape[0]} rows")
        
        if len(g001_month1) > 0:
            print(f"Date range: {g001_month1['cohort'].min()} to {g001_month1['cohort'].max()}")
            print(f"CAC range: ${g001_month1['total_cac'].min():,.0f} to ${g001_month1['total_cac'].max():,.0f}")
            return g001_data, g001_month1
        else:
            print("❌ No data found for elapsed_months = 1")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading G001 data: {e}")
        return None, None

def load_alpha_data():
    """Load alpha values and find G001 category"""
    print("\n🧮 Step 2: Loading Alpha Values...")
    
    try:
        alpha_data = pd.read_csv("category_alpha_all_variables.csv")
        print(f"✅ Alpha data loaded successfully!")
        print(f"Categories available: {list(alpha_data['category'])}")
        
        # Find G001 category
        category_mapping = pd.read_excel("new.xlsx")
        g001_category = None
        
        for _, row in category_mapping.iterrows():
            if str(row.iloc[0]).strip() == 'G001':
                g001_category = str(row.iloc[2]).strip()
                break
        
        if g001_category:
            g001_alpha_row = alpha_data[alpha_data['category'] == g001_category]
            if not g001_alpha_row.empty:
                g001_alpha_row = g001_alpha_row.iloc[0]
                print(f"✅ G001 belongs to category: {g001_category}")
                alpha_count = sum(1 for col in g001_alpha_row.index if '_alpha' in col and not pd.isna(g001_alpha_row[col]))
                print(f"Alpha values available: {alpha_count}")
                return g001_alpha_row, g001_category
            else:
                print(f"❌ No alpha data found for category: {g001_category}")
                return None, None
        else:
            print("❌ G001 category not found in mapping")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading alpha data: {e}")
        return None, None

# ============================================================================
# STEP 2: PROJECTION FUNCTIONS
# ============================================================================

def find_latest_cohort(g001_month1):
    """Find the latest cohort for G001"""
    print("\n🔍 Step 3: Finding Latest G001 Cohort...")
    
    if g001_month1 is not None and len(g001_month1) > 0:
        # Convert cohort to datetime for proper sorting
        g001_month1['cohort_date'] = pd.to_datetime(g001_month1['cohort'], dayfirst=True)
        
        # Get the latest cohort
        latest_cohort_row = g001_month1.loc[g001_month1['cohort_date'].idxmax()]
        
        print(f"✅ Latest cohort found!")
        print(f"Cohort date: {latest_cohort_row['cohort']}")
        print(f"Total CAC: ${latest_cohort_row['total_cac']:,.0f}")
        print(f"Transaction period LTV/CAC: {latest_cohort_row['transaction_period_ltv_cac']:.4f}")
        print(f"Users: {latest_cohort_row['users']:,.0f}")
        
        return latest_cohort_row, latest_cohort_row['total_cac']
    else:
        print("❌ Cannot find latest cohort - no data available")
        return None, None

def project_new_row_enhanced(base_row, alpha_row, new_spend, category, new_cohort_date, month_increment):
    """Enhanced projection function that projects all available variables using their respective alpha values"""
    projected_row = base_row.copy()
    spend_increase = (new_spend - base_row['total_cac']) / base_row['total_cac']
    changes_applied = 0
    
    # Update basic spend and date info
    projected_row['total_cac'] = new_spend
    projected_row['client_actual_spend'] = new_spend
    projected_row['cohort'] = new_cohort_date.strftime('%d/%m/%Y')
    projected_row['transaction_period'] = new_cohort_date.strftime('%d/%m/%Y')
    
    # Get all alpha columns from the alpha_row
    alpha_columns = [col for col in alpha_row.index if '_alpha' in col]
    
    # Process each variable that has an alpha
    for alpha_col in alpha_columns:
        variable_name = alpha_col.replace('_alpha', '')
        
        # Skip LTV-equivalent variables (handled separately)
        if variable_name in ['m0_ltv_cac', 'cumulative_ltv_cac']:
            continue
            
        if not pd.isna(alpha_row[alpha_col]) and variable_name in base_row:
            alpha = alpha_row[alpha_col]
            base_value = base_row[variable_name]
            
            if base_value > 0:
                # Apply alpha projection formula
                projected_value = base_value / (1 + alpha * spend_increase)
                projected_row[variable_name] = projected_value
                changes_applied += 1
            elif base_value == 0:
                projected_row[variable_name] = 0
    
    # Process LTV-equivalent variables (use transaction_period_ltv_cac alpha)
    ltv_equivalent_variables = ['m0_ltv_cac', 'cumulative_ltv_cac']
    for variable in ltv_equivalent_variables:
        ltv_alpha_col = 'transaction_period_ltv_cac_alpha'
        if ltv_alpha_col in alpha_row and not pd.isna(alpha_row[ltv_alpha_col]):
            alpha = alpha_row[ltv_alpha_col]
            base_value = base_row.get(variable, 0)
            if base_value > 0:
                projected_value = base_value / (1 + alpha * spend_increase)
                projected_row[variable] = projected_value
                changes_applied += 1
            elif base_value == 0:
                projected_row[variable] = 0
        else:
            projected_row[variable] = base_row.get(variable, 0)
    
    return projected_row, changes_applied

def run_projection_experiment(base_row, g001_alpha_row, base_spend, g001_category):
    """Run the complete projection experiment"""
    print("\n🚀 Step 4: Running Enhanced Projection Experiment...")
    
    if base_row is not None and g001_alpha_row is not None:
        # Define spend increase percentages
        increase_percentages = [5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 500]
        
        # Set base cohort to June 2025 as requested by user
        base_cohort_date = datetime(2025, 6, 1)  # June 1, 2025
        
        results = []
        
        # Add the base cohort first
        print("📊 Adding base cohort...")
        base_row_copy = base_row.copy()
        base_row_copy['cohort'] = base_cohort_date.strftime('%d/%m/%Y')
        base_row_copy['transaction_period'] = base_cohort_date.strftime('%d/%m/%Y')
        base_row_copy['client_actual_spend'] = base_spend
        base_row_copy['experiment_id'] = 'G001_BASE'
        base_row_copy['base_spend'] = base_spend
        base_row_copy['new_spend'] = base_spend
        base_row_copy['increase_percentage'] = 0
        base_row_copy['spend_increase_pct'] = 0.0
        base_row_copy['variables_projected'] = 0
        base_row_copy['month_increment'] = 0
        results.append(base_row_copy)
        
        # Generate projections for each increase percentage
        for i, pct in enumerate(increase_percentages, 1):
            new_spend = base_spend * (1 + pct/100)
            
            # Calculate new cohort date (+i months from base)
            new_year = base_cohort_date.year
            new_month = base_cohort_date.month + i
            while new_month > 12:
                new_month -= 12
                new_year += 1
            new_cohort_date = base_cohort_date.replace(year=new_year, month=new_month, day=1)
            
            print(f"🔮 Projecting for +{pct}% spend increase: ${new_spend:,.0f} (Cohort: {new_cohort_date.strftime('%d/%m/%Y')})")
            
            projected_row, variables_changed = project_new_row_enhanced(
                base_row, g001_alpha_row, new_spend, g001_category, new_cohort_date, i
            )
            
            # Add experiment metadata
            projected_row['experiment_id'] = f'G001_+{pct}pct'
            projected_row['base_spend'] = base_spend
            projected_row['new_spend'] = new_spend
            projected_row['increase_percentage'] = pct
            projected_row['spend_increase_pct'] = pct
            projected_row['variables_projected'] = variables_changed
            projected_row['month_increment'] = i
            
            results.append(projected_row)
            print(f"   ✅ Projected {variables_changed} variables")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        print(f"\n✅ Projection experiment completed!")
        print(f"📊 Generated {len(results_df)} rows (1 base + {len(increase_percentages)} projections)")
        print(f"🎯 Spend range: ${results_df['total_cac'].min():,.0f} - ${results_df['total_cac'].max():,.0f}")
        
        return results_df
    else:
        print("❌ Cannot run experiment - missing base data or alpha values")
        return None

# ============================================================================
# STEP 3: FORECASTING FUNCTIONS
# ============================================================================

def load_forecasting_model():
    """Load the saved XGBoost forecasting model"""
    print("\n🤖 Step 5: Loading Forecasting Model...")
    
    model_path = "models/G001_xgb_0.5.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            forecasting_model = cloudpickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"Model type: {type(forecasting_model)}")
        print(f"Features expected: {len(forecasting_model.feature_names_in_)}")
        return forecasting_model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

def forecast_alpha_cohorts(game_id, model, projected_df, forecast_months=12):
    """Simplified forecasting function for alpha-generated cohorts"""
    if model is None:
        print("❌ No model available for forecasting")
        return pd.DataFrame()
    
    print(f"\n🚀 Step 6: Running Forecasting on {len(projected_df)} Alpha-Generated Cohorts...")
    
    forecasted_results = []
    
    # Process each cohort from our projected data
    for _, cohort_row in projected_df.iterrows():
        cohort = cohort_row['cohort']
        m0_ltv_cac = cohort_row['m0_ltv_cac']
        total_cac = cohort_row['client_actual_spend'] 
        experiment_id = cohort_row['experiment_id']
        
        print(f"🔮 Forecasting for {experiment_id}: {cohort} (CAC: ${total_cac:,.0f})")
        
        # Initialize cumulative tracking
        cumulative_expected = m0_ltv_cac
        
        # Initialize lag features
        lag_1 = np.nan
        lag_2 = np.nan  
        lag_3 = np.nan
        
        # For month 1, we use the projected transaction_period_ltv_cac as actual
        month_1_actual = cohort_row['transaction_period_ltv_cac']
        
        # Add month 1 as "actual" (from alpha projection)
        forecasted_results.append({
            'game_id': game_id,
            'cohort': cohort,
            'experiment_id': experiment_id,
            'elapsed_months': 1,
            'current_month': pd.to_datetime(cohort, dayfirst=True).strftime('%Y-%m'),
            'transactional_actual': month_1_actual,
            'cumulative_actual': cumulative_expected + month_1_actual,
            'absolute_revenue_actual': total_cac * month_1_actual,
            'transactional_expected': month_1_actual,
            'cumulative_expected': cumulative_expected + month_1_actual,
            'absolute_revenue_expected': total_cac * month_1_actual,
            'total_cac': total_cac,
            'is_projected': True
        })
        
        # Update cumulative and lag for month 1
        cumulative_expected += month_1_actual
        lag_1 = month_1_actual
        
        # Forecast for months 2 to forecast_months
        for month in range(2, forecast_months + 1):
            try:
                # Prepare features for prediction
                features = cohort_row.copy()
                
                # Update dynamic features
                features['elapsed_months'] = month
                
                # Calculate current month based on cohort start
                cohort_start_date = pd.to_datetime(cohort, dayfirst=True)
                current_date = cohort_start_date + pd.DateOffset(months=month-1)
                if 'current_month' in model.feature_names_in_:
                    features['current_month'] = current_date.month
                
                # Update lag features
                features['ltv_lag_1'] = lag_1
                features['ltv_lag_2'] = lag_2  
                features['ltv_lag_3'] = lag_3
                
                # Select only the features the model expects
                feature_data = {}
                for feature_name in model.feature_names_in_:
                    if feature_name in features:
                        feature_data[feature_name] = features[feature_name]
                    else:
                        # Handle missing features with defaults
                        feature_data[feature_name] = 0
                
                # Create DataFrame for prediction
                features_df = pd.DataFrame([feature_data])
                
                # Make prediction
                predicted_val = max(0, model.predict(features_df)[0])
                
                # Update cumulative
                cumulative_expected += predicted_val
                
                # Store result
                forecasted_results.append({
                    'game_id': game_id,
                    'cohort': cohort,
                    'experiment_id': experiment_id,
                    'elapsed_months': month,
                    'current_month': current_date.strftime('%Y-%m'),
                    'transactional_actual': None,
                    'cumulative_actual': None,
                    'absolute_revenue_actual': None,
                    'transactional_expected': predicted_val,
                    'cumulative_expected': cumulative_expected,
                    'absolute_revenue_expected': total_cac * predicted_val,
                    'total_cac': total_cac,
                    'is_projected': True
                })
                
                # Update lag features for next iteration
                lag_3 = lag_2
                lag_2 = lag_1
                lag_1 = predicted_val
                
            except Exception as e:
                print(f"⚠️ Error forecasting month {month} for cohort {cohort}: {e}")
                break
    
    forecast_df = pd.DataFrame(forecasted_results)
    
    if not forecast_df.empty:
        print(f"\n✅ Forecasting completed successfully!")
        print(f"📊 Forecast dataset shape: {forecast_df.shape}")
    
    return forecast_df

# ============================================================================
# STEP 4: PLOTTING FUNCTIONS
# ============================================================================

def plot_spend_scenario_comparison(forecast_df, base_scenario='G001_BASE', comparison_scenario='G001_+5pct', end_month=12):
    """Plot comparison between base scenario and a specific spend increase scenario"""
    # Filter data for both scenarios
    base_data = forecast_df[forecast_df['experiment_id'] == base_scenario].copy()
    comp_data = forecast_df[forecast_df['experiment_id'] == comparison_scenario].copy()
    
    if base_data.empty or comp_data.empty:
        print(f"❌ Missing data for {base_scenario} or {comparison_scenario}")
        return
    
    # Get spend increase percentage for title
    comp_pct = comparison_scenario.replace('G001_+', '').replace('pct', '%')
    base_cac = base_data['total_cac'].iloc[0]
    comp_cac = comp_data['total_cac'].iloc[0]
    
    months = list(range(1, end_month + 1))
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Transactional LTV/CAC: {comp_pct} Increase vs Base',
            f'Cumulative LTV/CAC: {comp_pct} Increase vs Base'
        ),
        vertical_spacing=0.12
    )
    
    # Transactional LTV/CAC Plot
    fig.add_trace(
        go.Scatter(
            x=months,
            y=base_data['transactional_expected'][:end_month],
            mode='lines+markers',
            name=f'Base (${base_cac:,.0f})',
            line=dict(color='blue', width=3),
            marker=dict(size=8, symbol='circle')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=months,
            y=comp_data['transactional_expected'][:end_month],
            mode='lines+markers',
            name=f'{comp_pct} Increase (${comp_cac:,.0f})',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # Cumulative LTV/CAC Plot
    fig.add_trace(
        go.Scatter(
            x=months,
            y=base_data['cumulative_expected'][:end_month],
            mode='lines+markers',
            name=f'Base Cumulative (${base_cac:,.0f})',
            line=dict(color='blue', width=3),
            marker=dict(size=8, symbol='circle'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=months,
            y=comp_data['cumulative_expected'][:end_month],
            mode='lines+markers',
            name=f'{comp_pct} Cumulative (${comp_cac:,.0f})',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'G001 Forecast Comparison: {comp_pct} Spend Increase vs Base',
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Monthly LTV/CAC", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative LTV/CAC", row=2, col=1)
    
    fig.show()
    
    # Print key metrics
    base_12m = base_data[base_data['elapsed_months'] == 12]['cumulative_expected'].iloc[0]
    comp_12m = comp_data[comp_data['elapsed_months'] == 12]['cumulative_expected'].iloc[0]
    ltv_improvement = ((comp_12m / base_12m - 1) * 100)
    
    print(f"\n📊 {comp_pct} vs Base Comparison:")
    print(f"• Base 12M LTV/CAC: {base_12m:.4f}")
    print(f"• {comp_pct} 12M LTV/CAC: {comp_12m:.4f}")
    print(f"• LTV/CAC Change: {ltv_improvement:+.2f}%")
    print(f"• Spend Increase: {((comp_cac/base_cac-1)*100):+.1f}%")

def plot_all_scenarios_comparison(forecast_df, end_month=12):
    """Plot all spend scenarios together for comprehensive comparison"""
    # Get unique experiment IDs and sort them
    scenarios = sorted(forecast_df['experiment_id'].unique())
    
    # Define colors for different scenarios
    colors = px.colors.qualitative.Set3
    if len(scenarios) > len(colors):
        colors = colors * (len(scenarios) // len(colors) + 1)
    
    months = list(range(1, end_month + 1))
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Transactional LTV/CAC: All Spend Scenarios',
            'Cumulative LTV/CAC: All Spend Scenarios'
        ),
        vertical_spacing=0.12
    )
    
    # Plot each scenario
    for i, scenario in enumerate(scenarios):
        scenario_data = forecast_df[forecast_df['experiment_id'] == scenario].copy()
        total_cac = scenario_data['total_cac'].iloc[0]
        
        # Determine line style
        line_style = dict(width=3)
        if scenario == 'G001_BASE':
            line_style.update(color='black', width=4)
        else:
            line_style.update(color=colors[i])
        
        # Transactional plot
        fig.add_trace(
            go.Scatter(
                x=months,
                y=scenario_data['transactional_expected'][:end_month],
                mode='lines+markers',
                name=f'{scenario.replace("G001_", "")} (${total_cac:,.0f})',
                line=line_style,
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Cumulative plot
        fig.add_trace(
            go.Scatter(
                x=months,
                y=scenario_data['cumulative_expected'][:end_month],
                mode='lines+markers',
                name=f'{scenario.replace("G001_", "")} Cumulative',
                line=line_style,
                marker=dict(size=6),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='G001 Complete Spend Scenario Analysis: All Forecasts',
        height=900,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Monthly LTV/CAC", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative LTV/CAC", row=2, col=1)
    
    fig.show()
    
    # Print summary statistics
    month_12_data = forecast_df[forecast_df['elapsed_months'] == 12].copy()
    print(f"\n📊 12-Month Performance Summary (All Scenarios):")
    print(f"{'Scenario':<15} {'CAC':<10} {'12M LTV/CAC':<12} {'Revenue':<12} {'vs Base':<8}")
    print("-" * 65)
    
    base_ltv = month_12_data[month_12_data['experiment_id'] == 'G001_BASE']['cumulative_expected'].iloc[0]
    
    for _, row in month_12_data.sort_values('total_cac').iterrows():
        scenario = row['experiment_id'].replace('G001_', '')
        cac = row['total_cac']
        ltv = row['cumulative_expected']
        revenue = ltv * cac
        vs_base = ((ltv / base_ltv - 1) * 100) if scenario != 'BASE' else 0
        
        print(f"{scenario:<15} ${cac:<9,.0f} {ltv:<12.4f} ${revenue:<11,.0f} {vs_base:+5.1f}%")

def generate_all_plots(forecast_df):
    """Generate all individual comparison plots and comprehensive overview"""
    print("\n🎨 Step 7: Generating Comprehensive Plotting Suite...")
    
    if forecast_df.empty:
        print("❌ No forecast results available for plotting")
        return
    
    # Get all non-base scenarios
    all_scenarios = forecast_df['experiment_id'].unique()
    comparison_scenarios = [s for s in all_scenarios if s != 'G001_BASE']
    
    print(f"Generating {len(comparison_scenarios)} individual comparison plots...")
    
    # Generate individual plots for each spend scenario vs base
    for i, scenario in enumerate(comparison_scenarios, 1):
        increase_pct = scenario.replace('G001_+', '').replace('pct', '%')
        print(f"\n📊 Plot {i}/{len(comparison_scenarios)}: {increase_pct} vs Base")
        
        plot_spend_scenario_comparison(
            forecast_df=forecast_df,
            base_scenario='G001_BASE',
            comparison_scenario=scenario,
            end_month=12
        )
    
    print(f"\n✅ Generated {len(comparison_scenarios)} individual comparison plots!")
    
    # Generate comprehensive comparison plot
    print(f"\n🎨 Generating Comprehensive All-Scenarios Comparison...")
    plot_all_scenarios_comparison(forecast_df=forecast_df, end_month=12)
    
    print(f"\n✅ All plotting completed!")

# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def main():
    """Execute the complete workflow"""
    
    # Step 1: Load data
    g001_data, g001_month1 = load_g001_data()
    g001_alpha_row, g001_category = load_alpha_data()
    
    if g001_data is None or g001_alpha_row is None:
        print("❌ Cannot proceed - missing essential data")
        return
    
    # Step 2: Find latest cohort and run projections
    base_row, base_spend = find_latest_cohort(g001_month1)
    
    if base_row is None:
        print("❌ Cannot proceed - no base cohort found")
        return
    
    # Step 3: Run projection experiment
    results_df = run_projection_experiment(base_row, g001_alpha_row, base_spend, g001_category)
    
    if results_df is None:
        print("❌ Cannot proceed - projection experiment failed")
        return
    
    # Save projection results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"G001_projection_experiment_enhanced_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\n💾 Alpha projections saved to: {output_filename}")
    
    # Step 4: Load model and run forecasting
    forecasting_model = load_forecasting_model()
    
    if forecasting_model is None:
        print("⚠️ No forecasting model available - skipping forecasting")
        return
    
    # Run forecasting
    forecast_results = forecast_alpha_cohorts(
        game_id='G001',
        model=forecasting_model,
        projected_df=results_df,
        forecast_months=12
    )
    
    if forecast_results.empty:
        print("❌ Forecasting failed")
        return
    
    # Save forecast results
    forecast_filename = f"G001_forecast_results_{timestamp}.csv"
    forecast_results.to_csv(forecast_filename, index=False)
    print(f"💾 Forecast results saved to: {forecast_filename}")
    
    # Step 5: Generate all plots
    generate_all_plots(forecast_results)
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 COMPLETE WORKFLOW SUMMARY")
    print("="*80)
    
    print(f"\n🎯 WORKFLOW ACCOMPLISHED:")
    print(f"✅ 1. Alpha Generation: Enhanced alpha values for all variables")
    print(f"✅ 2. Cohort Projection: 12 spend scenarios with progressive dates")  
    print(f"✅ 3. Model Integration: XGBoost forecasting model loaded")
    print(f"✅ 4. Future Forecasting: 12-month predictions for each cohort")
    print(f"✅ 5. Performance Analysis: Comprehensive ROI calculations")
    print(f"✅ 6. Interactive Visualizations: Individual + comprehensive plots")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"• Alpha Projections: {output_filename}")
    print(f"• Forecast Results: {forecast_filename}")
    print(f"• Interactive Plots: Generated in browser")
    
    print(f"\n📊 VISUALIZATION SUITE:")
    total_scenarios = len(forecast_results['experiment_id'].unique())
    print(f"• Individual Comparisons: {total_scenarios - 1} plots (each scenario vs base)")
    print(f"• Comprehensive Overview: 1 plot (all scenarios together)")
    print(f"• Total Interactive Charts: {total_scenarios} Plotly visualizations")
    
    print(f"\n✨ STATUS: COMPLETE PIPELINE WITH FULL VISUALIZATION! 📊🚀")
    print("="*80)

if __name__ == "__main__":
    main() 