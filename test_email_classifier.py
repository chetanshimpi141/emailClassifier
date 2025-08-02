#!/usr/bin/env python3
"""
Test script for the email classifier.
This script tests both training and prediction functionality.
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.append('src')

def test_preprocessing():
    """Test the preprocessing functionality."""
    print("🧪 Testing preprocessing...")
    
    from preprocess import clean_text, load_and_preprocess_data
    
    # Test text cleaning
    test_text = "Hello! This is a TEST email with UPPERCASE and numbers 123."
    cleaned = clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test data loading (with a small subset)
    try:
        # Create temporary directories with a few sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            spam_dir = os.path.join(temp_dir, "spam")
            ham_dir = os.path.join(temp_dir, "ham")
            os.makedirs(spam_dir)
            os.makedirs(ham_dir)
            
            # Copy a few sample files
            spam_files = os.listdir("data/spam")[:3]
            ham_files = os.listdir("data/easy_ham")[:3]
            
            for file in spam_files:
                shutil.copy(os.path.join("data/spam", file), spam_dir)
            for file in ham_files:
                shutil.copy(os.path.join("data/easy_ham", file), ham_dir)
            
            df = load_and_preprocess_data(spam_dir, ham_dir)
            print(f"✅ Loaded {len(df)} emails")
            print(f"Spam: {len(df[df['label'] == 1])}, Ham: {len(df[df['label'] == 0])}")
            
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False
    
    print("✅ Preprocessing test passed!")
    return True

def test_training():
    """Test the training functionality."""
    print("\n🧪 Testing training...")
    
    try:
        from train import train_and_save_model
        
        # Create temporary directories with sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            spam_dir = os.path.join(temp_dir, "spam")
            ham_dir = os.path.join(temp_dir, "ham")
            os.makedirs(spam_dir)
            os.makedirs(ham_dir)
            
            # Copy a few sample files for testing
            spam_files = os.listdir("data/spam")[:5]
            ham_files = os.listdir("data/easy_ham")[:5]
            
            for file in spam_files:
                shutil.copy(os.path.join("data/spam", file), spam_dir)
            for file in ham_files:
                shutil.copy(os.path.join("data/easy_ham", file), ham_dir)
            
            # Train model
            train_and_save_model(spam_dir, ham_dir)
            
            # Check if model files were created
            if os.path.exists("models/classifier.pkl") and os.path.exists("models/vectorizer.pkl"):
                print("✅ Model files created successfully!")
                return True
            else:
                print("❌ Model files not found!")
                return False
                
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def test_prediction():
    """Test the prediction functionality."""
    print("\n🧪 Testing prediction...")
    
    try:
        from predict import predict_email
        
        # Test predictions
        test_emails = [
            "Hello, this is a legitimate email from your bank.",
            "URGENT: You've won $1,000,000! Click here to claim your prize!",
            "Meeting reminder for tomorrow at 2 PM."
        ]
        
        for email in test_emails:
            prediction, probability = predict_email(email)
            label = "SPAM" if prediction == 1 else "HAM"
            confidence = probability[1] if prediction == 1 else probability[0]
            print(f"Email: {email[:30]}... -> {label} (confidence: {confidence:.2f})")
        
        print("✅ Prediction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Email Classifier Tests")
    print("=" * 50)
    
    # Test preprocessing
    if not test_preprocessing():
        print("❌ Preprocessing test failed!")
        return
    
    # Test training
    if not test_training():
        print("❌ Training test failed!")
        return
    
    # Test prediction
    if not test_prediction():
        print("❌ Prediction test failed!")
        return
    
    print("\n🎉 All tests passed! Email classifier is working correctly.")

if __name__ == "__main__":
    main() 