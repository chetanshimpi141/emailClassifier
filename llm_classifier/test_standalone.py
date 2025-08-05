#!/usr/bin/env python3
"""
Standalone test script for the LLM Email Classifier.
Tests the LLM classifier without any ML model dependencies.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from llm_predictor import LLMEmailClassifier

async def test_standalone_llm():
    """Test the LLM classifier standalone."""
    
    print("🧠 Testing Standalone LLM Classifier")
    print("=" * 50)
    
    # Test emails
    test_emails = [
        # Obvious spam
        """Subject: URGENT: You've won $1,000,000!
        
        CONGRATULATIONS! You are the lucky winner of our sweepstakes! 
        ACT NOW - Limited time offer! Click here to claim your prize: 
        http://fake-sweepstakes.com/claim
        
        This is a once-in-a-lifetime opportunity. Don't miss out!""",
        
        # Legitimate email
        """Subject: Meeting Reminder - Project Review
        
        Hi team,
        
        This is a reminder that we have our weekly project review meeting 
        tomorrow at 2 PM in Conference Room A. Please come prepared with 
        updates on your respective tasks.
        
        Agenda:
        1. Progress updates
        2. Blockers discussion
        3. Next week planning
        
        Best regards,
        John Doe
        Project Manager""",
        
        # Another spam
        """Subject: FORTUNE 500 COMPANY HIRING, AT HOME REPS.
        
        Help wanted. We are a 14 year old fortune 500 company, that is 
        growing at a tremendous rate. We are looking for individuals who 
        want to work from home. This is an opportunity to make an excellent 
        income. No experience is required. We will train you.
        
        Click here: http://www.basetel.com/wealthnow""",
        
        # Professional email
        """Subject: Your Microsoft Account Security Alert
        
        Dear User,
        
        We detected unusual activity on your Microsoft account. For your 
        security, we recommend reviewing your recent sign-in activity.
        
        To review your account activity, please visit: 
        https://account.microsoft.com/security
        
        If you did not sign in from this location, please secure your 
        account immediately.
        
        Microsoft Security Team"""
    ]
    
    try:
        # Initialize LLM classifier
        print("🔧 Initializing LLM classifier...")
        classifier = LLMEmailClassifier()
        
        # Try to load insights (optional)
        if os.path.exists("ml_insights.json"):
            classifier.load_ml_insights("ml_insights.json")
            print("✅ Loaded ML insights")
        else:
            print("ℹ️  No ML insights found. Using default prompts.")
        
        print("✅ LLM classifier initialized successfully!")
        print()
        
        # Test each email
        for i, email in enumerate(test_emails, 1):
            print(f"📧 Testing Email {i}:")
            print("-" * 40)
            
            # Get email preview
            preview = email[:100] + "..." if len(email) > 100 else email
            print(f"Preview: {preview}")
            
            # Classify email
            print("\n🤖 LLM Classification:")
            try:
                result = await classifier.classify_email(email)
                
                print(f"   Classification: {result['classification']}")
                print(f"   Confidence: {(result['confidence'] * 100):.1f}% ({result.get('confidence_level', 'MEDIUM')})")
                print(f"   Reasons: {', '.join(result['reasons'][:3])}")
                
                if result['suspicious_elements']:
                    print(f"   Suspicious: {', '.join(result['suspicious_elements'][:3])}")
                
                if result['explanation']:
                    print(f"   Explanation: {result['explanation']}")
                    
            except Exception as e:
                print(f"   ❌ Classification failed: {e}")
            
            print("-" * 40)
            print()
        
        # Test batch classification
        print("📊 Testing Batch Classification:")
        print("-" * 40)
        
        try:
            results = await classifier.batch_classify(test_emails)
            
            for i, result in enumerate(results):
                confidence_pct = (result['confidence'] * 100)
                print(f"Email {i+1}: {result['classification']} ({confidence_pct:.1f}%)")
                
        except Exception as e:
            print(f"❌ Batch classification failed: {e}")
        
        print("\n✅ Standalone LLM test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Set GEMINI_API_KEY in .env file")
        print("   2. Installed required packages: pip install -r requirements.txt")
        print("   3. Valid Gemini API key")

async def main():
    """Run the standalone test."""
    await test_standalone_llm()

if __name__ == "__main__":
    asyncio.run(main()) 