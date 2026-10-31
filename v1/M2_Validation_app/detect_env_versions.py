#!/usr/bin/env python3
"""
Environment Version Detector
==================================

This script detects the exact package versions in your development environment
where the M2 models work fine, so we can replicate them in Docker.

Run this in your development environment where models work.
"""

import sys
import pkg_resources

def get_installed_versions():
    """Get all installed package versions"""
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 60)
    
    # Key packages for M2 forecasting
    key_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm', 'catboost', 'plotly', 
        'cloudpickle', 'scipy'
    ]
    
    versions = {}
    
    print("📦 Key Package Versions:")
    print("-" * 40)
    
    for package in key_packages:
        try:
            # Try to get version using pkg_resources
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
            print(f"✅ {package:15} : {version}")
        except pkg_resources.DistributionNotFound:
            try:
                # Fallback: try importing and checking __version__
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                versions[package] = version
                print(f"✅ {package:15} : {version}")
            except ImportError:
                versions[package] = 'NOT_INSTALLED'
                print(f"❌ {package:15} : NOT INSTALLED")
    
    return versions

def test_model_loading_in_env():
    """Test if M2 models can be loaded in current environment"""
    print("\n🧪 Testing M2 Model Loading in Development Environment")
    print("=" * 60)
    
    import os
    import cloudpickle
    
    # Look for M2 models
    model_paths = []
    if os.path.exists("M2/new_models"):
        models = [f for f in os.listdir("M2/new_models") if f.endswith("_M2.pkl")]
        model_paths = [os.path.join("M2/new_models", m) for m in models[:3]]
    
    if not model_paths:
        print("❌ No M2 models found in M2/new_models/")
        return False
    
    success_count = 0
    for model_path in model_paths:
        try:
            print(f"🔍 Testing: {os.path.basename(model_path)}")
            with open(model_path, 'rb') as f:
                model = cloudpickle.load(f)
            print(f"   ✅ Loaded successfully")
            
            # Check model structure
            if hasattr(model, 'steps'):
                print(f"   📋 Pipeline with {len(model.steps)} steps")
            if hasattr(model, 'feature_names_in_'):
                print(f"   🔧 {len(model.feature_names_in_)} features")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
    
    print(f"\n📊 Result: {success_count}/{len(model_paths)} models loaded successfully")
    return success_count == len(model_paths)

def generate_exact_requirements(versions):
    """Generate requirements.txt with exact development-environment versions"""
    print("\n📝 Generating Exact Requirements")
    print("=" * 60)
    
    # Only include packages that are actually installed
    installed_versions = {k: v for k, v in versions.items() if v != 'NOT_INSTALLED'}
    
    # Write requirements_exact.txt
    with open('requirements_exact.txt', 'w') as f:
        for package, version in installed_versions.items():
            f.write(f"{package}=={version}\n")
    
    print("✅ Generated requirements_exact.txt with your development-environment versions:")
    print("-" * 50)
    for package, version in installed_versions.items():
        print(f"{package}=={version}")
    print("-" * 50)
    
    return True

def main():
    print("🔍 Environment Version Detector")
    print("=" * 60)
    print("This will detect the exact versions where your M2 models work!\n")
    
    # Step 1: Get versions
    versions = get_installed_versions()
    
    # Step 2: Test model loading
    models_work = test_model_loading_in_env()
    
    # Step 3: Generate exact requirements
    generate_exact_requirements(versions)
    
    # Summary and recommendations
    print("\n🎯 SUMMARY & NEXT STEPS")
    print("=" * 60)
    
    if models_work:
        print("✅ M2 models work in your development environment!")
        print("🔧 Use requirements_exact.txt for your Docker deployment:")
        print("   1. cp requirements_exact.txt requirements.txt")
        print("   2. docker build -t m2-forecasting .")
        print("   3. Deploy to GCP")
        
        # Show the critical sklearn version
        sklearn_version = versions.get('scikit-learn', 'unknown')
        print(f"\n🔑 Key: Your working sklearn version is {sklearn_version}")
        
    else:
        print("❌ M2 models don't work in your development environment either!")
        print("🔧 You may need to:")
        print("   1. Check if M2 models exist in M2/new_models/")
        print("   2. Verify models were trained correctly")
        print("   3. Check for missing dependencies")
    
    print(f"\n📁 Files generated:")
    print(f"   - requirements_exact.txt (exact development-environment versions)")

if __name__ == "__main__":
    main() 