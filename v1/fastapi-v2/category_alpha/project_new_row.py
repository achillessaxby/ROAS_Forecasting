import pandas as pd
import numpy as np

def load_category_alphas(alpha_file="category_alpha_all_variables.csv"):
    """Load the category alpha values"""
    try:
        df = pd.read_csv(alpha_file)
        return df
    except Exception as e:
        print(f"Error loading alpha file: {e}")
        return None

def project_new_row(base_row, category_alphas, new_spend, category):
    """
    Project a new row based on spend change using alpha values
    
    Parameters:
    - base_row: dict or Series with current values for elapsed_months=1
    - category_alphas: DataFrame with alpha values for each category  
    - new_spend: new spend value (total_cac)
    - category: category of the game ('Casual', 'Core', 'Non-gaming', 'Real-money')
    
    Returns:
    - projected_row: dict with projected values
    """
    
    # Get alpha values for the category
    cat_alphas = category_alphas[category_alphas['category'] == category]
    
    if cat_alphas.empty:
        print(f"No alpha values found for category: {category}")
        return None
        
    cat_alphas = cat_alphas.iloc[0]  # Get first (and only) row
    
    # Calculate spend increase
    base_spend = base_row['total_cac']
    spend_increase = (new_spend - base_spend) / base_spend
    
    projected_row = base_row.copy()
    projected_row['total_cac'] = new_spend
    
    print(f"\nProjecting new row for category: {category}")
    print(f"Base spend: ${base_spend:,.2f}")
    print(f"New spend: ${new_spend:,.2f}")
    print(f"Spend increase: {spend_increase:.2%}")
    print("\nProjected changes:")
    
    # List of variables we calculated alpha for
    variables = [
        'transaction_period_ltv_cac', 'cumulative_ltv_cac', 'day_1_cumulative_ltv_cac',
        'day_7_cumulative_ltv_cac', 'day_14_cumulative_ltv_cac', 'day_30_cumulative_ltv_cac',
        'day_1_payer_retention', 'day_7_payer_retention', 'day_14_payer_retention', 
        'day_30_payer_retention', 'day_1_net_revenue_retention', 'day_7_net_revenue_retention',
        'day_14_net_revenue_retention', 'day_30_net_revenue_retention', 'user_retention',
        'revenue_retention', 'gross_revenue', 'net_revenue', 'avg_gross_revenue',
        'avg_net_revenue', 'users', 'transactions'
    ]
    
    for variable in variables:
        alpha_col = f'{variable}_alpha'
        
        if alpha_col in cat_alphas and not pd.isna(cat_alphas[alpha_col]):
            alpha = cat_alphas[alpha_col]
            base_value = base_row.get(variable, 0)
            
            if base_value > 0:
                # Formula: new_value = base_value / (1 + alpha * spend_increase)
                projected_value = base_value / (1 + alpha * spend_increase)
                projected_row[variable] = projected_value
                
                change_pct = ((projected_value - base_value) / base_value) * 100
                print(f"  {variable}: {base_value:.4f} → {projected_value:.4f} ({change_pct:+.1f}%)")
            else:
                projected_row[variable] = base_value
                print(f"  {variable}: {base_value} (no change - invalid base)")
        else:
            # Keep original value if no alpha available
            projected_row[variable] = base_row.get(variable, 0)
            print(f"  {variable}: No alpha available, keeping original value")
    
    return projected_row

def example_usage():
    """Example of how to use the projection function"""
    
    # Load alpha values
    print("Loading category alpha values...")
    category_alphas = load_category_alphas()
    
    if category_alphas is None:
        print("Cannot proceed without alpha values. Run generate_category_alphas_extended.py first.")
        return
    
    print("Available categories:")
    print(category_alphas[['category', 'games_count']])
    
    # Example base row (you would get this from your actual data)
    example_base_row = {
        'total_cac': 50000,  # Base spend
        'transaction_period_ltv_cac': 1.2,
        'day_30_cumulative_ltv_cac': 0.8,
        'day_30_payer_retention': 15.0,
        'gross_revenue': 25000,
        'net_revenue': 20000,
        'users': 500,
        'transactions': 150,
        'user_retention': 0.3,
        'revenue_retention': 0.25
    }
    
    # Project for different spend levels
    category = 'Casual'  # Example category
    new_spends = [60000, 75000, 100000]  # 20%, 50%, 100% increases
    
    print(f"\n=== Projection Example for {category} Category ===")
    
    for new_spend in new_spends:
        print(f"\n{'='*60}")
        projected_row = project_new_row(example_base_row, category_alphas, new_spend, category)
        
        if projected_row is not None:
            # Show key metrics
            print(f"\nKey projected metrics:")
            key_metrics = ['transaction_period_ltv_cac', 'day_30_cumulative_ltv_cac', 
                          'gross_revenue', 'users', 'day_30_payer_retention']
            
            for metric in key_metrics:
                if metric in projected_row:
                    print(f"  {metric}: {projected_row[metric]:.4f}")

def create_projection_function_for_inference():
    """
    Create a simplified projection function for use in inference/production
    """
    code = '''
def project_metrics_from_spend_change(base_metrics, category_alphas, new_spend, category):
    """
    Project new metrics based on spend change using pre-calculated alpha values
    
    Args:
        base_metrics: dict with current metrics (must include 'total_cac')
        category_alphas: dict or DataFrame row with alpha values for the category
        new_spend: new total_cac value
        category: category name
    
    Returns:
        dict with projected metrics
    """
    base_spend = base_metrics['total_cac']
    spend_increase = (new_spend - base_spend) / base_spend
    
    projected = base_metrics.copy()
    projected['total_cac'] = new_spend
    
    # Variables to project (must match your alpha calculation)
    variables = [
        'transaction_period_ltv_cac', 'cumulative_ltv_cac', 'day_30_cumulative_ltv_cac',
        'gross_revenue', 'net_revenue', 'users', 'transactions', 'day_30_payer_retention'
        # Add other variables as needed
    ]
    
    for variable in variables:
        alpha_key = f'{variable}_alpha'
        if alpha_key in category_alphas and variable in base_metrics:
            alpha = category_alphas[alpha_key]
            if not pd.isna(alpha) and base_metrics[variable] > 0:
                projected[variable] = base_metrics[variable] / (1 + alpha * spend_increase)
    
    return projected

# Example usage:
# projected_row = project_metrics_from_spend_change(current_metrics, alphas['Casual'], 75000, 'Casual')
'''
    
    with open('projection_function.py', 'w') as f:
        f.write(code)
    
    print("Created projection_function.py for easy integration into your inference code!")

if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Create standalone function
    create_projection_function_for_inference()
    
    print("\n" + "="*60)
    print("HOW TO USE IN YOUR INFERENCE CODE:")
    print("1. Load your category alpha values once at startup")
    print("2. For any base row with elapsed_months=1, call project_new_row()")
    print("3. Specify the new spend level you want to project for")
    print("4. Get back a complete projected row with all metrics!")
    print("="*60) 