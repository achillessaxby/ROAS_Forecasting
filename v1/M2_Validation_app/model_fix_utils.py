"""
Model Loading Utilities with sklearn Compatibility Fixes
========================================================

This module provides fallback methods for loading M2 models when there are
sklearn version compatibility issues.
"""

import cloudpickle
import pickle
import warnings
import os

def load_model_with_fallback(model_path):
    """
    Try multiple methods to load a pickled model, with fallbacks for sklearn issues
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        tuple: (model, success_flag, error_message)
    """
    
    # Method 1: Try cloudpickle (normal method)
    try:
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        return model, True, None
    except Exception as e1:
        error1 = str(e1)
        
        # Method 2: Try regular pickle
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, True, f"Loaded with pickle fallback (cloudpickle failed: {error1[:100]})"
        except Exception as e2:
            error2 = str(e2)
            
            # Method 3: Try with sklearn compatibility patches
            try:
                model = load_with_sklearn_patches(model_path)
                return model, True, f"Loaded with sklearn patches (both pickle methods failed)"
            except Exception as e3:
                error3 = str(e3)
                
                return None, False, f"All methods failed: cloudpickle({error1[:50]}), pickle({error2[:50]}), patches({error3[:50]})"

def load_with_sklearn_patches(model_path):
    """
    Attempt to load model with sklearn compatibility patches
    """
    try:
        # Patch 1: Try to monkey-patch the missing _RemainderColsList
        import sklearn.compose._column_transformer as ct
        if not hasattr(ct, '_RemainderColsList'):
            # Create a dummy class for backward compatibility
            class _RemainderColsList(list):
                pass
            ct._RemainderColsList = _RemainderColsList
        
        # Now try loading
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        
        return model
        
    except Exception as e:
        # Patch 2: Try importing from different sklearn versions
        try:
            # Sometimes the class moved locations
            from sklearn.compose import _RemainderColsList
            import sklearn.compose._column_transformer as ct
            ct._RemainderColsList = _RemainderColsList
            
            with open(model_path, 'rb') as f:
                model = cloudpickle.load(f)
            return model
            
        except Exception:
            raise e

def get_model_info_safe(model):
    """
    Safely extract information from a loaded model without triggering errors
    """
    info = {
        'type': str(type(model).__name__),
        'has_predict': hasattr(model, 'predict'),
        'has_steps': hasattr(model, 'steps'),
        'has_feature_names': hasattr(model, 'feature_names_in_'),
        'is_pipeline': False,
        'preprocessing_steps': []
    }
    
    try:
        if hasattr(model, 'steps'):
            info['is_pipeline'] = True
            info['preprocessing_steps'] = [step[0] for step in model.steps]
            
        if hasattr(model, 'feature_names_in_'):
            info['feature_count'] = len(model.feature_names_in_)
        elif hasattr(model, 'n_features_in_'):
            info['feature_count'] = model.n_features_in_
        else:
            info['feature_count'] = 'unknown'
            
    except Exception as e:
        info['extraction_error'] = str(e)
    
    return info

def test_model_prediction(model, feature_names=None, sample_size=10):
    """
    Test if the model can make predictions with dummy data
    """
    try:
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        if feature_names:
            n_features = len(feature_names)
            X_test = pd.DataFrame(
                np.random.randn(sample_size, n_features), 
                columns=feature_names
            )
        else:
            # Try with a reasonable number of features
            n_features = 20
            X_test = pd.DataFrame(np.random.randn(sample_size, n_features))
        
        # Try prediction
        predictions = model.predict(X_test)
        
        return True, f"Successfully predicted {len(predictions)} samples"
        
    except Exception as e:
        return False, f"Prediction test failed: {str(e)}"

def diagnose_model_compatibility(model_path):
    """
    Comprehensive diagnosis of model compatibility issues
    """
    diagnosis = {
        'file_exists': os.path.exists(model_path),
        'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        'load_success': False,
        'model_info': None,
        'prediction_test': None,
        'recommendations': []
    }
    
    if not diagnosis['file_exists']:
        diagnosis['recommendations'].append("Check file path - model file does not exist")
        return diagnosis
    
    # Try loading
    model, success, error_msg = load_model_with_fallback(model_path)
    diagnosis['load_success'] = success
    diagnosis['load_error'] = error_msg
    
    if success and model is not None:
        # Get model info
        diagnosis['model_info'] = get_model_info_safe(model)
        
        # Test prediction
        pred_success, pred_msg = test_model_prediction(model)
        diagnosis['prediction_test'] = {'success': pred_success, 'message': pred_msg}
        
        if pred_success:
            diagnosis['recommendations'].append("✅ Model appears to be working correctly")
        else:
            diagnosis['recommendations'].append("⚠️ Model loads but prediction fails - check feature compatibility")
    else:
        diagnosis['recommendations'].append("❌ Model loading failed - likely sklearn version incompatibility")
        diagnosis['recommendations'].append("🔧 Try using requirements.txt with pinned sklearn version")
        diagnosis['recommendations'].append("🔄 Consider retraining models in current environment")
    
    return diagnosis 