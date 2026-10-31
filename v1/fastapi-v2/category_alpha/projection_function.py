
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
