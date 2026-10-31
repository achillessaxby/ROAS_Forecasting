import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import cloudpickle
import os



class ModelFactory:
    """Factory class to create different types of models with their parameter distributions"""
    
    @staticmethod
    def get_model_configs():
        """Returns configuration for all available models"""
        return {
            'xgboost': {
                'model_class': xgb.XGBRegressor,
                'model_params': {'enable_categorical': True, 'random_state': 42},
                'param_dist': {
                    'model__n_estimators': randint(50, 500),
                    'model__learning_rate': uniform(0.01, 0.09),
                    'model__max_depth': randint(3, 10),
                    'model__subsample': uniform(0.6, 0.4),
                    'model__colsample_bytree': uniform(0.6, 0.4),
                    'model__min_child_weight': randint(1, 10),
                    'model__gamma': uniform(0, 0.3),
                    'model__reg_alpha': uniform(0, 1),
                    'model__reg_lambda': uniform(1, 2),
                }
            },
            'random_forest': {
                'model_class': RandomForestRegressor,
                'model_params': {'random_state': 42, 'n_jobs': -1},
                'param_dist': {
                    'model__n_estimators': randint(50, 300),
                    'model__max_depth': randint(5, 20),
                    'model__min_samples_split': randint(2, 20),
                    'model__min_samples_leaf': randint(1, 10),
                    'model__max_features': ['sqrt', 'log2', None],
                    'model__bootstrap': [True, False],
                }
            },
            'lightgbm': {
                'model_class': lgb.LGBMRegressor,
                'model_params': {'random_state': 42, 'verbosity': -1},
                'param_dist': {
                    'model__n_estimators': randint(50, 500),
                    'model__learning_rate': uniform(0.01, 0.09),
                    'model__max_depth': randint(3, 15),
                    'model__num_leaves': randint(10, 300),
                    'model__subsample': uniform(0.6, 0.4),
                    'model__colsample_bytree': uniform(0.6, 0.4),
                    'model__min_child_samples': randint(5, 100),
                    'model__reg_alpha': uniform(0, 1),
                    'model__reg_lambda': uniform(0, 1),
                }
            },
            'catboost': {
                'model_class': cb.CatBoostRegressor,
                'model_params': {'random_state': 42, 'verbose': False, 'loss_function': 'RMSE'},
                'param_dist': {
                    'model__iterations': randint(100, 1000),
                    'model__learning_rate': uniform(0.01, 0.09),
                    'model__depth': randint(4, 10),
                    'model__l2_leaf_reg': loguniform(1, 10),
                    'model__border_count': randint(32, 255),
                    'model__bagging_temperature': uniform(0, 1),
                }
            },
            'ridge': {
                'model_class': Ridge,
                'model_params': {},
                'param_dist': {
                    'model__alpha': loguniform(0.01, 100),
                    'model__fit_intercept': [True, False],
                    'model__normalize': [True, False],
                }
            }
        }
    
    @staticmethod
    def create_model(model_type):
        """Create a model instance of the specified type"""
        configs = ModelFactory.get_model_configs()
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = configs[model_type]
        return config['model_class'](**config['model_params'])
    
    @staticmethod
    def get_param_distribution(model_type):
        """Get parameter distribution for hyperparameter tuning"""
        configs = ModelFactory.get_model_configs()
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return configs[model_type]['param_dist']

def create_pipeline(model, X_train):
    """Create a preprocessing pipeline for the model"""
    # Identify numerical columns
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
        ],
        remainder='passthrough'
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def train_and_evaluate_model(model_type, X_train, y_train, X_test, y_test, n_iter=100):
    """
    Train and evaluate a single model with hyperparameter tuning
    
    Loss Functions Used:
    - XGBoost: MSE (default regression objective)
    - Random Forest: MSE (built-in for regression)
    - LightGBM: MSE (default regression objective)
    - CatBoost: RMSE (default regression objective) 
    """
    try:
        # Create model
        model = ModelFactory.create_model(model_type)
        
        # Create pipeline
        pipeline = create_pipeline(model, X_train)
        
        # Get parameter distribution
        param_dist = ModelFactory.get_param_distribution(model_type)
        
        # Enhanced hyperparameter tuning with more iterations for fast models
        if model_type in ['catboost', 'lightgbm']:
            # Fast models can handle more iterations
            n_iter_adj = min(n_iter * 1.5, 150)
        else:
            # Tree-based models (XGBoost, Random Forest)
            n_iter_adj = max(n_iter, 50)
        
        # Hyperparameter tuning with cross-validation
        random_search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist,
            n_iter=int(n_iter_adj), 
            scoring='neg_mean_squared_error',  # Using MSE as base loss for consistency
            cv=5, 
            verbose=0, 
            n_jobs=-1, 
            random_state=42,
            error_score='raise'  # Better error handling
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Store training feature names for later forecasting
        best_model.__dict__["feature_names_in_"] = X_train.columns
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate RMSE
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        return {
            "model": best_model,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "best_params": random_search.best_params_,
            "cv_score": -random_search.best_score_  # Convert back to positive MSE
        }
        
    except Exception as e:
        print(f"Error training {model_type}: {str(e)}")
        return None

def train_multiple_models(X_train, y_train, X_test, y_test, model_types=None, n_iter=100):
    """Train multiple models and return results"""
    if model_types is None:
        model_types = ['xgboost', 'random_forest', 'lightgbm', 'catboost']
    
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type}...")
        result = train_and_evaluate_model(model_type, X_train, y_train, X_test, y_test, n_iter)
        if result:
            results[model_type] = result
    
    return results

def save_all_models(game_id, model_results, model_suffix=""):
    """Save all trained models and return information about saved models"""
    if not model_results:
        return None
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    saved_models = {}
    best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['rmse_test'])
    
    # Save all models
    for model_name, result in model_results.items():
        filename = f'models/{game_id}_{model_name}{model_suffix}.pkl'
        
        # Save the model
        with open(filename, 'wb') as f:
            cloudpickle.dump(result['model'], f)
        
        saved_models[model_name] = {
            'filename': filename,
            'rmse_test': result['rmse_test'],
            'rmse_train': result['rmse_train'],
            'is_best': model_name == best_model_name
        }
    
    return {
        'saved_models': saved_models,
        'best_model': best_model_name,
        'best_rmse': model_results[best_model_name]['rmse_test'],
        'total_models_saved': len(saved_models)
    }

def save_best_model(game_id, model_results, model_suffix=""):
    """Save the best performing model based on test RMSE - DEPRECATED: Use save_all_models instead"""
    if not model_results:
        return None
    
    # Find best model based on test RMSE
    best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['rmse_test'])
    best_result = model_results[best_model_name]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create filename with model type indication
    filename = f'models/{game_id}_{best_model_name}{model_suffix}.pkl'
    
    # Save the best model
    with open(filename, 'wb') as f:
        cloudpickle.dump(best_result['model'], f)
    
    return {
        'filename': filename,
        'model_type': best_model_name,
        'rmse_test': best_result['rmse_test'],
        'rmse_train': best_result['rmse_train']
    }

def load_specific_model(game_id, model_type, model_suffix="_M2"):
    """Load a specific model for a game"""
    filename = f'models/{game_id}_{model_type}{model_suffix}.pkl'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            model = cloudpickle.load(f)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {filename}: {str(e)}")

def get_available_models_for_game(game_id, model_suffix="_M2"):
    """Get list of available trained models for a specific game"""
    models_dir = 'models'
    available_models = []
    
    if not os.path.exists(models_dir):
        return available_models
    
    # Look for model files matching the pattern
    import glob
    pattern = f"{models_dir}/{game_id}_*{model_suffix}.pkl"
    model_files = glob.glob(pattern)
    
    for model_file in model_files:
        # Extract model type from filename
        filename = os.path.basename(model_file)
        # Remove game_id prefix and suffix to get model type
        model_type = filename.replace(f"{game_id}_", "").replace(f"{model_suffix}.pkl", "")
        available_models.append(model_type)
    
    return sorted(available_models) 