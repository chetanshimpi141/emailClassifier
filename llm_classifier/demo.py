#!/usr/bin/env python3
"""
Demo script for the LLM Email Classifier.
Tests the classifier with sample emails and compares with ML model.
"""

import os
import sys
sys.path.append('..')

from llm_predictor import LLMEmailClassifier
from src.predict import predict_email
import json

async def test_emails():
    """Test the LLM classifier with sample emails."""
    
    # Sample emails for testing
    test_emails = [
        # Obvious spam
        """From fort@bluemail.dk  Thu Aug 22 18:28:10 2002
Subject: FORTUNE 500 COMPANY HIRING, AT HOME REPS.

Help wanted. We are a 14 year old fortune 500 company, that is growing at a tremendous rate. We are looking for individuals who want to work from home. This is an opportunity to make an excellent income. No experience is required. We will train you. So if you are looking to be employed from home with a career that has vast opportunities, then go: http://www.basetel.com/wealthnow We are looking for energetic and self motivated people. If that is you than click on the link and fill out the form, and one of our employement specialist will contact you. To be removed from our link simple go to: http://www.basetel.com/remove.html 4139vOLW7-758DoDY1425FRhM1-764SMFc8513fCsLl40""",
        
        # Legitimate email
        """From: john.doe@company.com
Subject: Meeting Reminder - Project Review

Hi team,

This is a reminder that we have our weekly project review meeting tomorrow at 2 PM in Conference Room A. Please come prepared with updates on your respective tasks.

Agenda:
1. Progress updates
2. Blockers discussion
3. Next week planning

Best regards,
John Doe
Project Manager""",
        
        # Another spam
        """Subject: URGENT: You've won $1,000,000!

CONGRATULATIONS! You are the lucky winner of our $1,000,000 sweepstakes! 

ACT NOW - Limited time offer! Click here to claim your prize: http://fake-sweepstakes.com/claim

This is a once-in-a-lifetime opportunity. Don't miss out!""",
        
        # Professional email
        """From: support@microsoft.com
Subject: Your Microsoft Account Security Alert

Dear User,

We detected unusual activity on your Microsoft account. For your security, we recommend reviewing your recent sign-in activity.

To review your account activity, please visit: https://account.microsoft.com/security

If you did not sign in from this location, please secure your account immediately.

Microsoft Security Team"""
    ]
    
    # Initialize LLM classifier
    try:
        llm_classifier = LLMEmailClassifier()
        llm_classifier.load_ml_insights("ml_insights.json")
        print("✅ Gemini Classifier initialized with ML insights")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini classifier: {e}")
        return
    
    print("\n🧪 Testing Gemini Email Classifier")
    print("=" * 60)
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n📧 Email {i}:")
        print("-" * 40)
        
        # Get email preview
        preview = email[:100] + "..." if len(email) > 100 else email
        print(f"Preview: {preview}")
        
        # Gemini classification
        print("\n🤖 Gemini Classification:")
        try:
            llm_result = await llm_classifier.classify_email(email)
            print(f"   Classification: {llm_result['classification']}")
            print(f"   Confidence: {(llm_result['confidence'] * 100):.1f}% ({llm_result.get('confidence_level', 'MEDIUM')})")
            print(f"   Reasons: {', '.join(llm_result['reasons'][:3])}")
            if llm_result['suspicious_elements']:
                print(f"   Suspicious: {', '.join(llm_result['suspicious_elements'][:3])}")
        except Exception as e:
            print(f"   ❌ Gemini classification failed: {e}")
        
        # ML classification (for comparison)
        print("\n🤖 ML Classification:")
        try:
            ml_prediction, ml_probability = predict_email(email)
            ml_label = "SPAM" if ml_prediction == 1 else "HAM"
            ml_confidence = ml_probability[1] if ml_prediction == 1 else ml_probability[0]
            print(f"   Classification: {ml_label}")
            print(f"   Confidence: {ml_confidence:.3f}")
        except Exception as e:
            print(f"   ❌ ML classification failed: {e}")
        
        print("-" * 40)

async def test_batch_classification():
    """Test batch classification."""
    print("\n📊 Testing Batch Classification")
    print("=" * 60)
    
    # Sample emails for batch testing
    batch_emails = [
        "Hello, this is a legitimate business email.",
        "URGENT: Make money fast! Click here now!",
        "Meeting reminder for tomorrow at 3 PM.",
        "FREE VIAGRA! CLICK HERE TO ORDER NOW!",
        "Please review the attached quarterly report."
    ]
    
    try:
        llm_classifier = LLMEmailClassifier()
        llm_classifier.load_ml_insights("ml_insights.json")
        
        results = await llm_classifier.batch_classify(batch_emails)
        
        for i, result in enumerate(results):
            confidence_pct = (result['confidence'] * 100)
            print(f"\nEmail {i+1}: {batch_emails[i][:50]}...")
            print(f"  LLM: {result['classification']} ({confidence_pct:.1f}%)")
            print(f"  Reasons: {', '.join(result['reasons'][:2])}")
            
    except Exception as e:
        print(f"❌ Batch classification failed: {e}")

async def main():
    """Run the demo."""
    print("🚀 Gemini Email Classifier Demo")
    print("=" * 60)
    
    # Check if insights file exists (optional)
    if not os.path.exists("ml_insights.json"):
        print("ℹ️  No ML insights file found. LLM will work with default prompts.")
    else:
        print("✅ ML insights found and will be used.")
    
    # Test individual emails
    await test_emails()
    
    # Test batch classification
    await test_batch_classification()
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 