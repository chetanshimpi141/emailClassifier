#!/usr/bin/env python3
"""
Installation checker for the email classifier.
This script verifies that all required dependencies are installed.
"""

import sys
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a dependency is installed."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} is NOT installed")
        return False

def main():
    """Check all required dependencies."""
    print("🔍 Checking Email Classifier Dependencies")
    print("=" * 40)
    
    dependencies = [
        ("pandas", "pandas"),
        ("nltk", "NLTK"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
    ]
    
    all_installed = True
    for module, name in dependencies:
        if not check_dependency(module, name):
            all_installed = False
    
    print("\n" + "=" * 40)
    
    if all_installed:
        print("🎉 All dependencies are installed!")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    else:
        print("❌ Some dependencies are missing!")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    
    # Check NLTK data
    print("\n🔍 Checking NLTK data...")
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK stopwords data is available")
    except LookupError:
        print("❌ NLTK stopwords data is missing")
        print("Run: python -c \"import nltk; nltk.download('stopwords')\"")
    
    # Check project structure
    print("\n🔍 Checking project structure...")
    import os
    
    required_dirs = ["src", "data", "models"]
    required_files = ["src/preprocess.py", "src/train.py", "src/predict.py"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' is missing")
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' is missing")

if __name__ == "__main__":
    main() 