#!/usr/bin/env python3
"""
Verification script to confirm all fixes are working.
"""

import os

def check_file_consistency():
    """Check that all files use consistent naming."""
    print("🔍 Checking file naming consistency...")
    
    # Check train.py saves to correct path
    with open("src/train.py", "r") as f:
        train_content = f.read()
        if "../models/classifier.pkl" in train_content:
            print("✅ train.py saves to 'classifier.pkl'")
        else:
            print("❌ train.py has incorrect model file name")
    
    # Check predict.py loads from correct path
    with open("src/predict.py", "r") as f:
        predict_content = f.read()
        if "../models/classifier.pkl" in predict_content:
            print("✅ predict.py loads from 'classifier.pkl'")
        else:
            print("❌ predict.py has incorrect model file name")
    
    # Check test script looks for correct files
    with open("test_email_classifier.py", "r") as f:
        test_content = f.read()
        if "models/classifier.pkl" in test_content:
            print("✅ test script checks for 'classifier.pkl'")
        else:
            print("❌ test script has incorrect model file name")

def check_project_structure():
    """Check that all required files and directories exist."""
    print("\n🔍 Checking project structure...")
    
    required_items = [
        "src/preprocess.py",
        "src/train.py", 
        "src/predict.py",
        "data/spam",
        "data/easy_ham",
        "models",
        "requirements.txt",
        "README.md"
    ]
    
    for item in required_items:
        if os.path.exists(item):
            print(f"✅ {item} exists")
        else:
            print(f"❌ {item} missing")

def check_requirements():
    """Check that requirements.txt has all needed dependencies."""
    print("\n🔍 Checking requirements.txt...")
    
    with open("requirements.txt", "r") as f:
        requirements = f.read()
        
    needed_deps = ["pandas", "nltk", "scikit-learn", "joblib"]
    
    for dep in needed_deps:
        if dep in requirements:
            print(f"✅ {dep} in requirements.txt")
        else:
            print(f"❌ {dep} missing from requirements.txt")

def main():
    """Run all verification checks."""
    print("🚀 Email Classifier Fix Verification")
    print("=" * 40)
    
    check_file_consistency()
    check_project_structure()
    check_requirements()
    
    print("\n" + "=" * 40)
    print("🎉 Verification complete!")
    print("\nTo test the system:")
    print("1. python check_installation.py")
    print("2. python src/train.py")
    print("3. python test_email_classifier.py")

if __name__ == "__main__":
    main() 