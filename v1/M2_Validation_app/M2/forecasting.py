import pandas as pd
import numpy as np

def forecast_cumulative_ltv_cac_variations_m1(game_id, best_model_expected, best_model_bad, best_model_good, df, forecast_months=12):
    """Forecast cumulative LTV/CAC variations for M1 model"""
    game_data = df[df['game_id'] == game_id].copy()
    forecasted_results = []
    sp_ltv_cac = df[df['transaction_period_ltv_cac'] > 0]['transaction_period_ltv_cac'].min()
    
    for cohort in game_data['cohort'].unique():
        cohort_data = game_data[game_data['cohort'] == cohort].copy()
        total_cac = cohort_data['total_cac'].iloc[0]
        last_elapsed_month = int(cohort_data['elapsed_months'].max())
        cumulative_actual = 0

        actual_cumulative = []
        for month in range(0, last_elapsed_month + 1):
            transactional_actual = cohort_data.loc[cohort_data['elapsed_months'] == month, 'transaction_period_ltv_cac'].values
            transactional_actual = transactional_actual[0] if len(transactional_actual) > 0 else 0
            cumulative_actual += transactional_actual
            actual_cumulative.append({
                'month': month,
                'transactional_actual': transactional_actual,
                'cumulative_actual': cumulative_actual,
                'absolute_revenue_actual': total_cac * transactional_actual,
                'total_cac': total_cac
            })

        
        lag_1 = np.nan
        lag_2 = np.nan
        lag_3 = np.nan
        lag_1b = np.nan
        lag_1g = np.nan

        cumulative_values = {
            'cumulative_bad': 0,
            'cumulative_expected': 0,
            'cumulative_good': 0
        }
        
        static_features = cohort_data.loc[cohort_data['elapsed_months'] == last_elapsed_month,
                                          best_model_expected.__dict__["feature_names_in_"]].iloc[0].copy()
        for col in ['elapsed_months', 'current_month', 'ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3']:
            if col in static_features:
                static_features = static_features.drop(col)

        
        for i in range(0, last_elapsed_month + forecast_months + 1):
            elapsed_months = i
            features = static_features.copy()
            features['elapsed_months'] = elapsed_months

            if 'current_month' in best_model_expected.__dict__["feature_names_in_"]:
                cohort_start_date = pd.to_datetime(cohort_data['cohort'].iloc[0])
                current_date = cohort_start_date + pd.DateOffset(months=elapsed_months)
                features['current_month'] = current_date.month
            else:
                current_date = pd.to_datetime(cohort_data['cohort'].iloc[0]) + pd.DateOffset(months=elapsed_months)

            features1 = features.copy()
            features2 = features.copy()
            
            features['ltv_lag_1'] = lag_1
            features1['ltv_lag_1'] = lag_1b
            features2['ltv_lag_1'] = lag_1g
            features['ltv_lag_2'] = lag_2
            features1['ltv_lag_2'] = lag_2
            features2['ltv_lag_2'] = lag_2
            features['ltv_lag_3'] = lag_3
            features1['ltv_lag_3'] = lag_3
            features2['ltv_lag_3'] = lag_3

            features_df = pd.DataFrame([features])
            features_df1 = pd.DataFrame([features1])
            features_df2 = pd.DataFrame([features2])

            predicted_val_bad = max(0, best_model_bad.predict(features_df1)[0])
            predicted_val_expected = max(0, best_model_expected.predict(features_df)[0])
            predicted_val_good = max(0, best_model_good.predict(features_df2)[0])
            x = [predicted_val_bad, predicted_val_expected, predicted_val_good]
            x.sort()
            predicted_val_bad, predicted_val_expected, predicted_val_good = x

            if predicted_val_good <= 0:
                predicted_val_good = sp_ltv_cac
            if predicted_val_expected <= 0:
                predicted_val_expected = predicted_val_good / 2
            if predicted_val_bad <= 0:
                predicted_val_bad = predicted_val_expected / 2

            lag_3 = lag_2
            lag_2 = lag_1
            lag_1 = predicted_val_expected
            lag_1b = predicted_val_bad
            lag_1g = predicted_val_good

            row_result = {
                'game_id': game_id,
                'cohort': cohort,
                'elapsed_months': elapsed_months,
                'current_month': current_date.strftime('%Y-%m')
            }
            actual_data = next((ac for ac in actual_cumulative if ac['month'] == elapsed_months), None)
            row_result['transactional_actual'] = actual_data['transactional_actual'] if actual_data else None
            row_result['cumulative_actual'] = actual_data['cumulative_actual'] if actual_data else None
            row_result['absolute_revenue_actual'] = actual_data['absolute_revenue_actual'] if actual_data else None
            row_result['total_cac'] = actual_data['total_cac'] if actual_data else total_cac

            for model_type, predicted_val in zip(['bad', 'expected', 'good'],
                                                   [predicted_val_bad, predicted_val_expected, predicted_val_good]):
                cumulative_key = f"cumulative_{model_type}"
                cumulative_values[cumulative_key] += predicted_val
                row_result[f"transactional_{model_type}"] = predicted_val
                row_result[cumulative_key] = cumulative_values[cumulative_key]
                row_result[f"absolute_revenue_{model_type}"] = total_cac * predicted_val

            forecasted_results.append(row_result)

    forecast_df = pd.DataFrame(forecasted_results)
    return forecast_df

def forecast_cumulative_ltv_cac_variations_m2(game_id, best_model_expected, best_model_bad, best_model_good, df, forecast_months=12, epsilon=0.001):
    """Forecast cumulative LTV/CAC variations for M2 model"""
    game_data = df[df['game_id'] == game_id].copy()

    forecasted_results = []
    sp_ltv_cac = df[df['transaction_period_ltv_cac'] > 0]['transaction_period_ltv_cac'].min()

    for cohort in game_data['cohort'].unique():
        cohort_data = game_data[game_data['cohort'] == cohort].copy()
        m0_ltv_cac = cohort_data['m0_ltv_cac'].iloc[0]
        total_cac = cohort_data['client_actual_spend'].iloc[0]
        last_elapsed_month = int(cohort_data['elapsed_months'].max())
        
        cumulative_actual = m0_ltv_cac
        actual_cumulative = []
        for month in range(1, last_elapsed_month + 1):
            historical_data = cohort_data.loc[cohort_data['elapsed_months'] == month, 'transaction_period_ltv_cac']
            transactional_actual = historical_data.values[0] if not historical_data.empty else 0
            cumulative_actual += transactional_actual
            actual_cumulative.append({
                'month': month,
                'transactional_actual': transactional_actual,
                'cumulative_actual': cumulative_actual,
                'absolute_revenue_actual': total_cac * transactional_actual,
                'total_cac': total_cac
            })

        lag_1 = np.nan; lag_2 = np.nan; lag_3 = np.nan
        lag_1b = np.nan; lag_1g = np.nan; lag_2b = np.nan; lag_2g = np.nan; lag_3b = np.nan; lag_3g = np.nan
        cumulative_values = {key: m0_ltv_cac for key in ['cumulative_bad', 'cumulative_expected', 'cumulative_good']}
        
        static_features = cohort_data.loc[cohort_data['elapsed_months'] == last_elapsed_month, best_model_expected.__dict__["feature_names_in_"]].iloc[0].copy()
        for col in ['elapsed_months', 'current_month', 'ltv_lag_1', 'ltv_lag_2', 'ltv_lag_3']:
            if col in static_features:
                static_features = static_features.drop(col)

        for i in range(1, last_elapsed_month + forecast_months + 1):
            elapsed_months = i
            features = static_features.copy()
            features['elapsed_months'] = elapsed_months
            
            if 'current_month' in best_model_expected.__dict__["feature_names_in_"]:
                cohort_start_date = pd.to_datetime(cohort_data['cohort'].iloc[0])
                current_date = cohort_start_date + pd.DateOffset(months=elapsed_months)
                features['current_month'] = current_date.month
            else:
                current_date = pd.to_datetime(cohort_data['cohort'].iloc[0]) + pd.DateOffset(months=elapsed_months)

            if i <= last_elapsed_month:
                # Historical months: use actual values for LAG FEATURES ONLY
                historical_data = cohort_data.loc[cohort_data['elapsed_months'] == i, 'transaction_period_ltv_cac']
                if historical_data.empty:
                    lag_value = epsilon  # Missing data
                else:
                    historical_value = historical_data.values[0]
                    lag_value = historical_value if historical_value != 0 else epsilon  # Replace zero with epsilon
                
                # Update lag features with actual historical data
                lag_1 = lag_value
                lag_1b = lag_value
                lag_1g = lag_value
                
                # BUT STILL USE MODELS TO PREDICT (don't set predictions = actual)
                features1 = features.copy()
                features2 = features.copy()
                features['ltv_lag_1'] = lag_1
                features1['ltv_lag_1'] = lag_1b
                features2['ltv_lag_1'] = lag_1g
                features['ltv_lag_2'] = lag_2
                features1['ltv_lag_2'] = lag_2b
                features2['ltv_lag_2'] = lag_2g
                features['ltv_lag_3'] = lag_3
                features1['ltv_lag_3'] = lag_3b
                features2['ltv_lag_3'] = lag_3g
                
                features_df = pd.DataFrame([features])
                features_df1 = pd.DataFrame([features1])
                features_df2 = pd.DataFrame([features2])
                
                predicted_val_bad = max(0, best_model_bad.predict(features_df1)[0])
                predicted_val_expected = max(0, best_model_expected.predict(features_df)[0])
                predicted_val_good = max(0, best_model_good.predict(features_df2)[0])
                
                x = [predicted_val_bad, predicted_val_expected, predicted_val_good]
                x.sort()
                predicted_val_bad, predicted_val_expected, predicted_val_good = x
                if predicted_val_good <= 0:
                    predicted_val_good = sp_ltv_cac
                if predicted_val_expected <= 0:
                    predicted_val_expected = predicted_val_good / 2
                if predicted_val_bad <= 0:
                    predicted_val_bad = predicted_val_expected / 2
                
            else:
                # Future months: use model predictions for both predictions AND lag features
                features1 = features.copy()
                features2 = features.copy()
                features['ltv_lag_1'] = lag_1
                features1['ltv_lag_1'] = lag_1b
                features2['ltv_lag_1'] = lag_1g
                features['ltv_lag_2'] = lag_2
                features1['ltv_lag_2'] = lag_2b
                features2['ltv_lag_2'] = lag_2g
                features['ltv_lag_3'] = lag_3
                features1['ltv_lag_3'] = lag_3b
                features2['ltv_lag_3'] = lag_3g
                
                features_df = pd.DataFrame([features])
                features_df1 = pd.DataFrame([features1])
                features_df2 = pd.DataFrame([features2])
                
                predicted_val_bad = max(0, best_model_bad.predict(features_df1)[0])
                predicted_val_expected = max(0, best_model_expected.predict(features_df)[0])
                predicted_val_good = max(0, best_model_good.predict(features_df2)[0])
                
                x = [predicted_val_bad, predicted_val_expected, predicted_val_good]
                x.sort()
                predicted_val_bad, predicted_val_expected, predicted_val_good = x
                if predicted_val_good <= 0:
                    predicted_val_good = sp_ltv_cac
                if predicted_val_expected <= 0:
                    predicted_val_expected = predicted_val_good / 2
                if predicted_val_bad <= 0:
                    predicted_val_bad = predicted_val_expected / 2
                
                # Update lag features with predicted values for next iteration
                lag_1b = predicted_val_bad
                lag_1 = predicted_val_expected
                lag_1g = predicted_val_good

            # Update lag variables
            lag_3 = lag_2
            lag_2 = lag_1
            lag_3b = lag_2b
            lag_2b = lag_1b
            lag_3g = lag_2g
            lag_2g = lag_1g

            # Construct result row
            row_result = {
                'game_id': game_id,
                'cohort': cohort,
                'elapsed_months': elapsed_months,
                'current_month': current_date.strftime('%Y-%m'),
            }
            actual_data = next((ac for ac in actual_cumulative if ac['month'] == elapsed_months), None)
            row_result['transactional_actual'] = actual_data['transactional_actual'] if actual_data else None
            row_result['cumulative_actual'] = actual_data['cumulative_actual'] if actual_data else None
            row_result['absolute_revenue_actual'] = actual_data['absolute_revenue_actual'] if actual_data else None
            row_result['total_cac'] = actual_data['total_cac'] if actual_data else total_cac
            
            for model_type, predicted_val in zip(['bad', 'expected', 'good'], [predicted_val_bad, predicted_val_expected, predicted_val_good]):
                cumulative_key = f"cumulative_{model_type}"
                cumulative_values[cumulative_key] += predicted_val
                row_result[f"transactional_{model_type}"] = predicted_val
                row_result[cumulative_key] = cumulative_values[cumulative_key]
                row_result[f"absolute_revenue_{model_type}"] = total_cac * predicted_val
            
            forecasted_results.append(row_result)

    forecast_df = pd.DataFrame(forecasted_results)
    return forecast_df 