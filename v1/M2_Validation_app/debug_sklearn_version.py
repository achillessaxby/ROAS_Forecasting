#!/usr/bin/env python3
"""
Debug Script: Identify sklearn version compatibility for M2 models
==================================================================

This script helps identify the sklearn version that was used to train
your M2 models and provides troubleshooting steps for the deployment.

Run this script in your development environment where the models were trained.
"""

import cloudpickle
import os
import sys
import subprocess

def check_sklearn_version():
    """Check current sklearn version"""
    try:
        import sklearn
        print(f"✅ Current sklearn version: {sklearn.__version__}")
        return sklearn.__version__
    except ImportError:
        print("❌ sklearn not installed")
        return None

def check_model_sklearn_version(model_path):
    """Try to load a model and identify sklearn version requirements"""
    try:
        print(f"\n🔍 Analyzing model: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        
        print("✅ Model loaded successfully")
        
        # Check if it's a pipeline
        if hasattr(model, 'steps'):
            print("📋 Model is a sklearn Pipeline")
            for step_name, step_obj in model.steps:
                print(f"  - {step_name}: {type(step_obj).__name__}")
        
        # Check for ColumnTransformer
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'transformers'):
                print("🔧 Found ColumnTransformer in preprocessor")
                for name, transformer, columns in preprocessor.transformers:
                    print(f"  - {name}: {type(transformer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        if "_RemainderColsList" in str(e):
            print("🚨 This is the sklearn version compatibility issue!")
        return False

def find_m2_models():
    """Find all M2 model files"""
    model_files = []
    
    # Check M2/new_models directory
    if os.path.exists("M2/new_models"):
        for file in os.listdir("M2/new_models"):
            if file.endswith("_M2.pkl"):
                model_files.append(os.path.join("M2/new_models", file))
    
    # Check new_models directory (if models were moved)
    if os.path.exists("new_models"):
        for file in os.listdir("new_models"):
            if file.endswith("_M2.pkl"):
                model_files.append(os.path.join("new_models", file))
    
    return model_files

def generate_requirements():
    """Generate requirements.txt with compatible versions"""
    
    # Get current package versions
    packages = {
        'streamlit': '1.28.0',
        'pandas': '2.0.3', 
        'numpy': '1.24.3',
        'scikit-learn': '1.3.0',  # Known compatible version
        'xgboost': '1.7.6',
        'lightgbm': '4.0.0',
        'catboost': '1.2',
        'plotly': '5.15.0',
        'cloudpickle': '2.2.1',
        'scipy': '1.11.1'
    }
    
    print("\n📝 Recommended requirements.txt:")
    print("=" * 50)
    for package, version in packages.items():
        print(f"{package}=={version}")
    print("=" * 50)
    
    # Write to file
    with open("requirements_fixed.txt", "w") as f:
        for package, version in packages.items():
            f.write(f"{package}=={version}\n")
    
    print("💾 Saved as requirements_fixed.txt")

def check_docker_environment():
    """Check if running in Docker and provide Docker-specific advice"""
    if os.path.exists("/.dockerenv"):
        print("\n🐳 Running in Docker environment")
        print("💡 Rebuild your Docker image with fixed requirements.txt")
    else:
        print("\n💻 Running in development environment")

def main():
    print("🔍 M2 Model sklearn Compatibility Checker")
    print("=" * 50)
    
    # Check current sklearn version
    current_version = check_sklearn_version()
    
    # Find M2 models
    model_files = find_m2_models()
    
    if not model_files:
        print("\n❌ No M2 models found!")
        print("📁 Expected locations:")
        print("  - M2/new_models/")
        print("  - new_models/")
        return
    
    print(f"\n📊 Found {len(model_files)} M2 model files")
    
    # Test loading a few models
    success_count = 0
    for i, model_path in enumerate(model_files[:3]):  # Test first 3 models
        if check_model_sklearn_version(model_path):
            success_count += 1
    
    # Provide recommendations
    print(f"\n📈 Results: {success_count}/{min(3, len(model_files))} models loaded successfully")
    
    if success_count == 0:
        print("\n🚨 ISSUE DETECTED: sklearn version incompatibility")
        print("\n🔧 SOLUTIONS:")
        print("1. Use the fixed requirements.txt (will be generated below)")
        print("2. Rebuild your Docker image")
        print("3. If still failing, retrain models in current environment")
    elif success_count < 3:
        print("\n⚠️  PARTIAL COMPATIBILITY: Some models may have issues")
        print("🔧 Recommend using fixed requirements.txt")
    else:
        print("\n✅ All tested models compatible!")
    
    # Generate fixed requirements
    generate_requirements()
    
    # Docker advice
    check_docker_environment()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Replace your requirements.txt with requirements_fixed.txt")
    print("2. Rebuild Docker image: docker build -t your-app .")
    print("3. Test the application")
    print("4. If still failing, you may need to retrain models")

if __name__ == "__main__":
    main() 