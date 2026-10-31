#!/usr/bin/env python3
"""
Deployment Fix Script for M2 Models sklearn Compatibility
=========================================================

This script automates the process of fixing sklearn compatibility issues
in your M2 forecasting tool deployment.

Usage:
    python fix_deployment.py

This will:
1. Diagnose the current environment
2. Check model compatibility
3. Generate fixed requirements.txt
4. Provide deployment instructions
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_current_environment():
    """Check current Python and package versions"""
    print("🔍 Checking Current Environment")
    print("=" * 50)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Check key packages
    packages_to_check = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm', 'catboost', 'plotly', 'cloudpickle'
    ]
    
    versions = {}
    for package in packages_to_check:
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            versions[package] = version
            print(f"✅ {package}: {version}")
        except ImportError:
            versions[package] = 'not installed'
            print(f"❌ {package}: not installed")
    
    return versions

def find_model_files():
    """Find M2 model files in the project"""
    print("\n📁 Searching for M2 Model Files")
    print("=" * 50)
    
    model_locations = [
        "M2/new_models/",
        "new_models/",
        "models_m2/",
        "M2/models/"
    ]
    
    found_models = []
    
    for location in model_locations:
        if os.path.exists(location):
            files = [f for f in os.listdir(location) if f.endswith('_M2.pkl')]
            if files:
                print(f"📦 Found {len(files)} models in {location}")
                found_models.extend([os.path.join(location, f) for f in files[:3]])  # Sample first 3
            else:
                print(f"📂 {location} exists but no M2 models found")
        else:
            print(f"❌ {location} does not exist")
    
    return found_models

def test_model_loading(model_files):
    """Test loading M2 models"""
    print("\n🧪 Testing Model Loading")
    print("=" * 50)
    
    if not model_files:
        print("❌ No M2 model files found to test")
        return False
    
    success_count = 0
    sklearn_errors = 0
    
    for model_path in model_files:
        print(f"\n🔍 Testing: {os.path.basename(model_path)}")
        try:
            # Try the fallback loading method
            from model_fix_utils import load_model_with_fallback, get_model_info_safe
            
            model, success, error_msg = load_model_with_fallback(model_path)
            
            if success:
                print(f"✅ Loaded successfully")
                info = get_model_info_safe(model)
                print(f"   Type: {info['type']}")
                print(f"   Pipeline: {info['is_pipeline']}")
                success_count += 1
            else:
                print(f"❌ Failed to load: {error_msg}")
                if "_RemainderColsList" in str(error_msg) or "sklearn" in str(error_msg).lower():
                    sklearn_errors += 1
                    
        except ImportError:
            # Fallback method not available, try direct loading
            try:
                import cloudpickle
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
                print(f"✅ Loaded successfully (direct)")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to load: {str(e)}")
                if "_RemainderColsList" in str(e) or "sklearn" in str(e).lower():
                    sklearn_errors += 1
    
    print(f"\n📊 Results: {success_count}/{len(model_files)} models loaded successfully")
    print(f"🚨 sklearn compatibility errors: {sklearn_errors}")
    
    return success_count == len(model_files), sklearn_errors > 0

def generate_fixed_requirements():
    """Generate requirements.txt with compatible versions"""
    print("\n📝 Generating Fixed requirements.txt")
    print("=" * 50)
    
    # Recommended versions that work together
    requirements = {
        'streamlit': '1.28.0',
        'pandas': '2.0.3',
        'numpy': '1.24.3',
        'scikit-learn': '1.3.0',  # Key version for compatibility
        'xgboost': '1.7.6',
        'lightgbm': '4.0.0',
        'catboost': '1.2',
        'plotly': '5.15.0',
        'cloudpickle': '2.2.1',
        'scipy': '1.11.1'
    }
    
    # Write requirements.txt
    with open('requirements_fixed.txt', 'w') as f:
        for package, version in requirements.items():
            f.write(f"{package}=={version}\n")
    
    print("✅ Generated requirements_fixed.txt")
    
    # Show contents
    print("\n📋 Contents:")
    for package, version in requirements.items():
        print(f"   {package}=={version}")
    
    return True

def generate_deployment_instructions():
    """Generate step-by-step deployment instructions"""
    print("\n🚀 Deployment Instructions")
    print("=" * 50)
    
    instructions = """
🔧 FIXING SKLEARN COMPATIBILITY ISSUE

Step 1: Update Requirements
---------------------------
1. Replace your current requirements.txt with requirements_fixed.txt:
   
   mv requirements_fixed.txt requirements.txt

Step 2: Update Dockerfile (if using Docker)
-------------------------------------------
1. Your Dockerfile should look like this:

FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential gcc g++ \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \\
    STREAMLIT_SERVER_PORT=8080 \\
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "forecast_tool_m2.py", "--server.address=0.0.0.0", "--server.port=8080"]

Step 3: Rebuild and Deploy
--------------------------
1. Rebuild Docker image:
   docker build -t m2-forecasting .

2. Test in development:
   docker run -p 8080:8080 m2-forecasting

3. Deploy to GCP:
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/m2-forecasting
   gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/m2-forecasting --platform managed

Step 4: Verify Deployment
-------------------------
1. Check if models load correctly in the deployed app
2. Test M2 forecasting functionality
3. Monitor logs for any remaining errors

ALTERNATIVE SOLUTIONS (if above doesn't work):
---------------------------------------------
1. Retrain models: If sklearn compatibility issues persist, you may need to 
   retrain your M2 models in an environment with sklearn==1.3.0

2. Use fallback loading: The provided model_fix_utils.py includes fallback 
   methods that might work around some compatibility issues

3. Version matching: Find the exact sklearn version used for training and 
   use that in requirements.txt
"""
    
    print(instructions)
    
    # Write to file
    with open('DEPLOYMENT_INSTRUCTIONS.md', 'w') as f:
        f.write(f"# M2 Forecasting Tool - Deployment Fix Instructions\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(instructions)
    
    print("💾 Instructions saved to DEPLOYMENT_INSTRUCTIONS.md")

def main():
    """Main deployment fix process"""
    print("🔧 M2 Forecasting Tool - Deployment Fix Script")
    print("=" * 60)
    print("This script will help fix sklearn compatibility issues\n")
    
    # Step 1: Check environment
    versions = check_current_environment()
    
    # Step 2: Find models
    model_files = find_model_files()
    
    # Step 3: Test model loading
    all_good, has_sklearn_errors = test_model_loading(model_files)
    
    # Step 4: Generate fixes
    generate_fixed_requirements()
    
    # Step 5: Generate instructions
    generate_deployment_instructions()
    
    # Summary
    print("\n🎯 SUMMARY")
    print("=" * 50)
    
    if all_good and not has_sklearn_errors:
        print("✅ Everything looks good! Your models should work.")
        print("💡 Use the generated requirements_fixed.txt to ensure consistency")
    elif has_sklearn_errors:
        print("🚨 sklearn compatibility issues detected!")
        print("🔧 Use the generated requirements_fixed.txt and rebuild")
        print("📖 Follow DEPLOYMENT_INSTRUCTIONS.md for step-by-step guidance")
    else:
        print("⚠️  Some issues detected. Check the analysis above.")
        print("📖 Follow DEPLOYMENT_INSTRUCTIONS.md for guidance")
    
    print(f"\n📁 Generated files:")
    print(f"   - requirements_fixed.txt")
    print(f"   - DEPLOYMENT_INSTRUCTIONS.md")
    if os.path.exists('model_fix_utils.py'):
        print(f"   - model_fix_utils.py (fallback loading)")
    if os.path.exists('debug_sklearn_version.py'):
        print(f"   - debug_sklearn_version.py (debugging)")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. mv requirements_fixed.txt requirements.txt")
    print(f"   2. docker build -t m2-forecasting .")
    print(f"   3. Deploy to GCP")

if __name__ == "__main__":
    main() 