#!/usr/bin/env python3
"""
Test script for the Email Spam Classifier FastAPI.
"""

import requests
import json
import time
import os
import tempfile
import shutil

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_info():
    """Test the API info endpoint."""
    print("\n🧪 Testing API info endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API info: {data['message']} v{data['version']}")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def test_training():
    """Test the training endpoint."""
    print("\n🧪 Testing training endpoint...")
    
    # Check if we have sample data
    if not os.path.exists("data/spam") or not os.path.exists("data/easy_ham"):
        print("❌ Sample data not found. Skipping training test.")
        return False
    
    try:
        # Get a few sample files
        spam_files = os.listdir("data/spam")[:3]
        ham_files = os.listdir("data/easy_ham")[:3]
        
        if not spam_files or not ham_files:
            print("❌ No sample files found. Skipping training test.")
            return False
        
        # Prepare files for upload
        files = []
        
        # Add spam files
        for filename in spam_files:
            file_path = os.path.join("data/spam", filename)
            with open(file_path, 'rb') as f:
                files.append(('spam_files', (filename, f.read(), 'text/plain')))
        
        # Add ham files
        for filename in ham_files:
            file_path = os.path.join("data/easy_ham", filename)
            with open(file_path, 'rb') as f:
                files.append(('ham_files', (filename, f.read(), 'text/plain')))
        
        # Make training request
        response = requests.post(f"{API_BASE}/train", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training successful: {data['training_stats']}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint."""
    print("\n🧪 Testing prediction endpoint...")
    
    # Check if model exists
    if not os.path.exists("models/classifier.pkl"):
        print("❌ Model not found. Skipping prediction test.")
        return False
    
    test_emails = [
        "Hello, this is a legitimate email from your bank.",
        "URGENT: You've won $1,000,000! Click here to claim your prize!",
        "Meeting reminder for tomorrow at 2 PM."
    ]
    
    try:
        for i, email in enumerate(test_emails):
            data = {'email_text': email}
            response = requests.post(f"{API_BASE}/predict", data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Email {i+1}: {result['prediction']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"❌ Prediction {i+1} failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def test_batch_prediction():
    """Test the batch prediction endpoint."""
    print("\n🧪 Testing batch prediction endpoint...")
    
    # Check if model exists
    if not os.path.exists("models/classifier.pkl"):
        print("❌ Model not found. Skipping batch prediction test.")
        return False
    
    test_emails = [
        "Hello, this is a legitimate email from your bank.",
        "URGENT: You've won $1,000,000! Click here to claim your prize!",
        "Meeting reminder for tomorrow at 2 PM."
    ]
    
    try:
        data = {}
        for i, email in enumerate(test_emails):
            data[f'emails'] = email
        
        response = requests.post(f"{API_BASE}/predict-batch", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful: {result['total_emails']} emails processed")
            for pred in result['predictions']:
                print(f"  - Email {pred['email_id']+1}: {pred['prediction']} (confidence: {pred['confidence']:.2f})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Batch prediction test error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n🧪 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_BASE}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Email Spam Classifier API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("API Info", test_api_info),
        ("Model Info", test_model_info),
        ("Training", test_training),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎉 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! API is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 