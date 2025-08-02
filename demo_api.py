#!/usr/bin/env python3
"""
Demo script showing how to use the Email Spam Classifier FastAPI.
"""

import requests
import json
import os

API_BASE = "http://localhost:8000"

def demo_health_check():
    """Demo the health check endpoint."""
    print("🔍 Checking API health...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def demo_train_model():
    """Demo training the model with sample data."""
    print("🚀 Training model with sample data...")
    
    # Check if we have sample data
    if not os.path.exists("data/spam") or not os.path.exists("data/easy_ham"):
        print("❌ Sample data not found. Skipping training demo.")
        return False
    
    # Get a few sample files
    spam_files = os.listdir("data/spam")[:2]
    ham_files = os.listdir("data/easy_ham")[:2]
    
    if not spam_files or not ham_files:
        print("❌ No sample files found. Skipping training demo.")
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
        print("✅ Training successful!")
        print(f"Training stats: {data['training_stats']}")
        print()
        return True
    else:
        print(f"❌ Training failed: {response.status_code}")
        print(f"Error: {response.text}")
        print()
        return False

def demo_single_prediction():
    """Demo single email prediction."""
    print("🔮 Making single email prediction...")
    
    test_emails = [
        "Hello, this is a legitimate email from your bank.",
        "URGENT: You've won $1,000,000! Click here to claim your prize!",
        "Meeting reminder for tomorrow at 2 PM."
    ]
    
    for i, email in enumerate(test_emails, 1):
        print(f"Email {i}: {email[:50]}...")
        
        data = {'email_text': email}
        response = requests.post(f"{API_BASE}/predict", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Ham prob: {result['probabilities']['ham']:.2f}")
            print(f"  Spam prob: {result['probabilities']['spam']:.2f}")
        else:
            print(f"  ❌ Prediction failed: {response.status_code}")
        print()

def demo_batch_prediction():
    """Demo batch email prediction."""
    print("📊 Making batch predictions...")
    
    test_emails = [
        "Hello, this is a legitimate email from your bank.",
        "URGENT: You've won $1,000,000! Click here to claim your prize!",
        "Meeting reminder for tomorrow at 2 PM."
    ]
    
    data = {}
    for i, email in enumerate(test_emails):
        data[f'emails'] = email
    
    response = requests.post(f"{API_BASE}/predict-batch", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch prediction successful: {result['total_emails']} emails processed")
        
        for pred in result['predictions']:
            print(f"  Email {pred['email_id']+1}: {pred['prediction'].upper()} "
                  f"(confidence: {pred['confidence']:.2f})")
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(f"Error: {response.text}")
    print()

def demo_model_info():
    """Demo getting model information."""
    print("ℹ️ Getting model information...")
    
    response = requests.get(f"{API_BASE}/model-info")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Model info retrieved:")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Model files: {data['model_files']}")
        if 'model_type' in data:
            print(f"  Model type: {data['model_type']}")
    else:
        print(f"❌ Failed to get model info: {response.status_code}")
    print()

def main():
    """Run all demos."""
    print("🎬 Email Spam Classifier API Demo")
    print("=" * 50)
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding properly.")
            print("Please start the server first: python start_server.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server.")
        print("Please start the server first: python start_server.py")
        return
    
    print("✅ API server is running!")
    print()
    
    # Run demos
    demo_health_check()
    demo_model_info()
    
    # Try to train if no model exists
    if not os.path.exists("models/classifier.pkl"):
        print("📝 No trained model found. Training with sample data...")
        demo_train_model()
    
    # Run predictions
    demo_single_prediction()
    demo_batch_prediction()
    
    print("🎉 Demo completed!")
    print("\n💡 Try the web interface at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 