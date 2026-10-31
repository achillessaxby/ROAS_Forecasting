#!/usr/bin/env python3
"""
G001 Projection Experiment Script - Updated Version

This script performs projection experiments on G001 game data using alpha values
to project complete new rows based on spend increases.

Updates:
- Includes m0_ltv_cac and all MC_* variables in projections
- Sets client_actual_spend = total_cac
- Creates new cohorts for each scenario (+1 month for 5%, +2 for 10%, etc.)
- Includes base cohort in final results
- Removes plotting sections

Experiment Setup:
1. Find the latest cohort of G001 with elapsed_months = 1
2. Get its total_cac (base spend)
3. Apply increments: [5%, 10%, 15%, 20%, 25%, 50%, 75%, 100%, 200%, 300%, 500%]
4. Project complete elapsed_months=1 rows for each increment with new cohorts
5. Include base cohort in results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🎮 G001 Projection Experiment - Updated Version")
print("="*60)

# =============================================================================
# 1. LOAD DATA AND SETUP
# =============================================================================

print("\n📚 Step 1: Loading Data...")

# Load G001 data
print("Loading G001 data...")
g001_df = pd.read_csv("Latest Data/Segregated/G001_Dataset_12_2024.csv")

print(f"G001 data shape: {g001_df.shape}")
print(f"Date range: {g001_df['cohort'].min()} to {g001_df['cohort'].max()}")
print(f"Elapsed months range: {g001_df['elapsed_months'].min()} to {g001_df['elapsed_months'].max()}")

# Load category alpha values
print("Loading category alpha values...")
category_alphas = pd.read_csv("category_alpha_all_variables.csv")

print(f"Alpha values loaded for {len(category_alphas)} categories")
print(category_alphas[['category', 'games_count']])

# Find G001's category
game_alphas = pd.read_csv("game_alphas_all_variables.csv")
g001_category = game_alphas[game_alphas['game_id'] == 'G001']['category'].iloc[0]
print(f"\nG001 belongs to category: {g001_category}")

# Get alpha values for G001's category
g001_alpha_row = category_alphas[category_alphas['category'] == g001_category].iloc[0]
print(f"Using alpha values from {g001_category} category")

# =============================================================================
# 2. FIND LATEST COHORT
# =============================================================================

print("\n🔍 Step 2: Finding Latest Cohort...")

# Filter to elapsed_months = 1
g001_month1 = g001_df[g001_df['elapsed_months'] == 1].copy()
print(f"Found {len(g001_month1)} rows with elapsed_months = 1")

# Convert cohort to datetime for proper sorting
g001_month1['cohort_date'] = pd.to_datetime(g001_month1['cohort'])

# Find the latest cohort
latest_cohort = g001_month1.loc[g001_month1['cohort_date'].idxmax()]

print(f"\nLatest cohort details:")
print(f"Cohort: {latest_cohort['cohort']}")
print(f"Total CAC: ${latest_cohort['total_cac']:,.2f}")
print(f"Transaction Period LTV/CAC: {latest_cohort['transaction_period_ltv_cac']:.4f}")
print(f"Users: {latest_cohort['users']:,.0f}")
print(f"Gross Revenue: ${latest_cohort['gross_revenue']:,.2f}")

# Store base values
base_spend = latest_cohort['total_cac']
base_row = latest_cohort.to_dict()

# Set base cohort to June 2025 as requested by user  
# Format: day=1, month=6 (June), year=2025
base_cohort_date = datetime(2025, 6, 1)  # June 1, 2025

print(f"\nBase spend for experiment: ${base_spend:,.2f}")
print(f"Base cohort date: {base_cohort_date.strftime('%B %Y')} ({base_cohort_date.strftime('%m/%d/%Y')})")

# =============================================================================
# 3. IDENTIFY ALL PROJECTION VARIABLES
# =============================================================================

print("\n🔧 Step 3: Identifying Variables for Projection...")

# Get all column names from the dataset
all_columns = list(g001_df.columns)

# Find MC_ columns
mc_columns = [col for col in all_columns if col.startswith('MC_')]
print(f"Found {len(mc_columns)} MC_ columns: {mc_columns}")

# Base variables from original analysis
base_variables = [
    'transaction_period_ltv_cac', 'cumulative_ltv_cac', 'day_1_cumulative_ltv_cac',
    'day_7_cumulative_ltv_cac', 'day_14_cumulative_ltv_cac', 'day_30_cumulative_ltv_cac',
    'day_1_payer_retention', 'day_7_payer_retention', 'day_14_payer_retention', 
    'day_30_payer_retention', 'day_1_net_revenue_retention', 'day_7_net_revenue_retention',
    'day_14_net_revenue_retention', 'day_30_net_revenue_retention', 'user_retention',
    'revenue_retention', 'gross_revenue', 'net_revenue', 'avg_gross_revenue',
    'avg_net_revenue', 'users', 'transactions'
]

# Add MC_ columns (m0_ltv_cac and cumulative_ltv_cac will use transaction_period_ltv_cac alpha)
additional_variables = mc_columns

# Combine all variables for projection  
all_projection_variables = base_variables + additional_variables

# Special variables that use transaction_period_ltv_cac alpha (since they're equal for elapsed_months=1)
ltv_equivalent_variables = ['m0_ltv_cac', 'cumulative_ltv_cac']

print(f"Total variables for projection: {len(all_projection_variables)} + {len(ltv_equivalent_variables)} LTV-equivalent")
print(f"Additional variables: {len(additional_variables)} MC_ columns")
print(f"LTV-equivalent variables: {ltv_equivalent_variables} (use transaction_period_ltv_cac alpha)")

# =============================================================================
# 4. DEFINE ENHANCED PROJECTION FUNCTION
# =============================================================================

def project_new_row_enhanced(base_row, alpha_row, new_spend, category, new_cohort_date, month_increment):
    """
    Enhanced projection function that includes MC_ variables and creates new cohorts
    """
    # Calculate spend increase
    base_spend = base_row['total_cac']
    spend_increase = (new_spend - base_spend) / base_spend
    
    projected_row = base_row.copy()
    projected_row['total_cac'] = new_spend
    projected_row['client_actual_spend'] = new_spend  # Set client_actual_spend = total_cac
    projected_row['spend_increase_pct'] = spend_increase * 100
    
    # Update cohort and transaction_period to new dates
    # Format: D/M/YYYY (day/month/year) where day is always 1
    projected_row['cohort'] = new_cohort_date.strftime('%d/%m/%Y')
    projected_row['transaction_period'] = new_cohort_date.strftime('%d/%m/%Y')
    projected_row['cohort_date'] = new_cohort_date
    
    changes_applied = 0
    
    # Process regular variables
    for variable in all_projection_variables:
        alpha_col = f'{variable}_alpha'
        
        if alpha_col in alpha_row and not pd.isna(alpha_row[alpha_col]):
            alpha = alpha_row[alpha_col]
            base_value = base_row.get(variable, 0)
            
            if base_value > 0:
                # Formula: new_value = base_value / (1 + alpha * spend_increase)
                projected_value = base_value / (1 + alpha * spend_increase)
                projected_row[variable] = projected_value
                changes_applied += 1
            elif base_value == 0:
                # Keep zero values as zero
                projected_row[variable] = 0
        else:
            # Keep original value if no alpha available
            projected_row[variable] = base_row.get(variable, 0)
    
    # Process LTV-equivalent variables (use transaction_period_ltv_cac alpha)
    for variable in ltv_equivalent_variables:
        ltv_alpha_col = 'transaction_period_ltv_cac_alpha'
        
        if ltv_alpha_col in alpha_row and not pd.isna(alpha_row[ltv_alpha_col]):
            alpha = alpha_row[ltv_alpha_col]
            base_value = base_row.get(variable, 0)
            
            if base_value > 0:
                # Formula: new_value = base_value / (1 + alpha * spend_increase)
                projected_value = base_value / (1 + alpha * spend_increase)
                projected_row[variable] = projected_value
                changes_applied += 1
            elif base_value == 0:
                # Keep zero values as zero
                projected_row[variable] = 0
        else:
            # Keep original value if no alpha available
            projected_row[variable] = base_row.get(variable, 0)
    
    projected_row['variables_projected'] = changes_applied
    projected_row['month_increment'] = month_increment
    return projected_row

print("\n⚙️ Step 4: Enhanced projection function defined successfully!")

# =============================================================================
# 5. RUN ENHANCED EXPERIMENT
# =============================================================================

print("\n🚀 Step 5: Running Enhanced Projection Experiment...")

# Define spend increase percentages
increase_percentages = [5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 500]

# Calculate new spend levels and create experiment results
experiment_results = []

print(f"Base spend: ${base_spend:,.2f}")
print(f"Base category: {g001_category}")
print(f"Base cohort: {base_cohort_date.strftime('%Y-%m-%d')}")

# Add base cohort to results first
base_row_copy = base_row.copy()
base_row_copy['experiment_id'] = "G001_BASE"
base_row_copy['base_spend'] = base_spend
base_row_copy['new_spend'] = base_spend
base_row_copy['increase_percentage'] = 0
base_row_copy['spend_increase_pct'] = 0.0
base_row_copy['variables_projected'] = 0
base_row_copy['month_increment'] = 0
base_row_copy['client_actual_spend'] = base_spend  # Ensure base also has this
# Update base cohort to June 2025
# Format: D/M/YYYY (day/month/year) where day is always 1
base_row_copy['cohort'] = base_cohort_date.strftime('%d/%m/%Y')
base_row_copy['transaction_period'] = base_cohort_date.strftime('%d/%m/%Y')
experiment_results.append(base_row_copy)

print(f"\n✓ BASE: ${base_spend:,.0f} (cohort: {base_cohort_date.strftime('%B %Y')})")

# Project for different spend levels
print(f"\nProjecting for {len(increase_percentages)} different spend levels...")

for i, pct in enumerate(increase_percentages, 1):
    new_spend = base_spend * (1 + pct/100)
    
    # Create new cohort date (base + i months using proper month arithmetic)
    # Add i months to base cohort date
    new_year = base_cohort_date.year
    new_month = base_cohort_date.month + i
    
    # Handle year rollover if month > 12
    while new_month > 12:
        new_month -= 12
        new_year += 1
    
    new_cohort_date = base_cohort_date.replace(year=new_year, month=new_month, day=1)
    
    # Project the new row
    projected_row = project_new_row_enhanced(
        base_row, g001_alpha_row, new_spend, g001_category, new_cohort_date, i
    )
    
    # Add metadata
    projected_row['experiment_id'] = f"G001_+{pct}pct"
    projected_row['base_spend'] = base_spend
    projected_row['new_spend'] = new_spend
    projected_row['increase_percentage'] = pct
    
    experiment_results.append(projected_row)
    
    print(f"✓ {pct}% (+{i}mo): ${base_spend:,.0f} → ${new_spend:,.0f} | Cohort: {new_cohort_date.strftime('%B %Y')}")

print(f"\n✅ Experiment completed! Generated {len(experiment_results)} scenarios (including base).")

# =============================================================================
# 6. SAVE AND ANALYZE RESULTS
# =============================================================================

print("\n💾 Step 6: Saving and Analyzing Results...")

# Create DataFrame and save results
results_df = pd.DataFrame(experiment_results)

# Remove cohort_date column (it was just for processing)
if 'cohort_date' in results_df.columns:
    results_df = results_df.drop('cohort_date', axis=1)

# Save to CSV
output_filename = f"G001_projection_experiment_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_filename, index=False)

print(f"💾 Results saved to: {output_filename}")
print(f"📊 Dataset shape: {results_df.shape}")

# Show key columns for analysis
key_columns = [
    'experiment_id', 'increase_percentage', 'cohort', 'total_cac', 'client_actual_spend',
    'transaction_period_ltv_cac', 'm0_ltv_cac', 'gross_revenue', 
    'users', 'variables_projected', 'month_increment'
]

print(f"\n📋 Key metrics summary:")
summary_df = results_df[key_columns].copy()
summary_df = summary_df.round(4)
print(summary_df.to_string(index=False))

# =============================================================================
# 7. ANALYZE MC COLUMN PROJECTIONS
# =============================================================================

print("\n📈 Step 7: Analyzing MC_ Column Projections...")

if mc_columns:
    print(f"\nMC_ columns projection summary:")
    mc_summary = results_df[['experiment_id'] + mc_columns[:5]].head(6)  # Show first 5 MC columns, first 6 rows
    print(mc_summary.round(4).to_string(index=False))
    
    # Count how many MC columns were projected
    mc_projected_count = 0
    for mc_col in mc_columns:
        alpha_col = f'{mc_col}_alpha'
        if alpha_col in g001_alpha_row and not pd.isna(g001_alpha_row[alpha_col]):
            mc_projected_count += 1
    
    print(f"\nMC_ columns with alpha values: {mc_projected_count}/{len(mc_columns)}")
else:
    print("No MC_ columns found in dataset.")

# =============================================================================
# 8. GENERATE INSIGHTS
# =============================================================================

print("\n🎯 Step 8: Generating Insights...")
print("ENHANCED EXPERIMENT SUMMARY & INSIGHTS")
print("="*60)

print(f"🎮 Game: G001 ({g001_category} category)")
print(f"📅 Base Cohort: {base_row['cohort']}")
print(f"💰 Base Spend: ${base_spend:,.2f}")

print(f"\n📊 PROJECTION RESULTS:")
print(f"✅ Successfully projected {len(experiment_results)} scenarios (including base)")
print(f"🔧 Variables projected per scenario: {results_df['variables_projected'].iloc[1]}")  # Skip base (index 0)
print(f"📊 Total variables considered: {len(all_projection_variables)}")
print(f"🏷️ MC_ variables included: {len(mc_columns)}")

# Show cohort progression
print(f"\n📅 COHORT PROGRESSION:")
cohort_progression = results_df[['experiment_id', 'cohort', 'month_increment']].head(6)
print(cohort_progression.to_string(index=False))

# Key insights from first few scenarios
print(f"\n🔍 KEY INSIGHTS (first 3 projections):")

base_values = {
    'transaction_period_ltv_cac': results_df.iloc[0]['transaction_period_ltv_cac'],
    'gross_revenue': results_df.iloc[0]['gross_revenue'],
    'users': results_df.iloc[0]['users'],
    'm0_ltv_cac': results_df.iloc[0]['m0_ltv_cac']
}

for i in [1, 2, 3]:  # Show first 3 projections (5%, 10%, 15%)
    row = results_df.iloc[i]
    pct = row['increase_percentage']
    
    ltv_change = ((row['transaction_period_ltv_cac'] - base_values['transaction_period_ltv_cac']) / base_values['transaction_period_ltv_cac']) * 100
    revenue_change = ((row['gross_revenue'] - base_values['gross_revenue']) / base_values['gross_revenue']) * 100
    
    print(f"📊 {pct}% spend increase (+{row['month_increment']}mo cohort):")
    print(f"   LTV/CAC: {base_values['transaction_period_ltv_cac']:.4f} → {row['transaction_period_ltv_cac']:.4f} ({ltv_change:+.1f}%)")
    print(f"   Revenue: ${base_values['gross_revenue']:,.0f} → ${row['gross_revenue']:,.0f} ({revenue_change:+.1f}%)")

print(f"\n💾 OUTPUT FILES:")
print(f"📋 Enhanced Results: {output_filename}")

print(f"\n🚀 FEATURES INCLUDED:")
print(f"✅ m0_ltv_cac projections")
print(f"✅ {len(mc_columns)} MC_ channel projections")
print(f"✅ client_actual_spend = total_cac")
print(f"✅ Progressive cohort dates")
print(f"✅ Base cohort included")

print(f"\n🚀 NEXT STEPS:")
print(f"• Use this enhanced dataset for complete scenario modeling")
print(f"• Validate MC_ channel projection accuracy")
print(f"• Compare m0_ltv_cac vs transaction_period_ltv_cac trends")
print(f"• Apply insights to multi-channel spend optimization")

print(f"\n✨ Enhanced experiment completed successfully!")
print("="*60)

if __name__ == "__main__":
    print("Enhanced script execution completed!") 