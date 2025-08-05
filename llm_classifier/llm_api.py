from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_predictor import LLMEmailClassifier

app = FastAPI(
    title="Gemini Email Spam Classifier API",
    description="Gemini-based email spam classification with ML insights",
    version="1.0.0"
)

# Global classifier instance
llm_classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the Gemini classifier on startup."""
    global llm_classifier
    try:
        llm_classifier = LLMEmailClassifier()
        
        # Try to load ML insights (optional)
        if os.path.exists("ml_insights.json"):
            llm_classifier.load_ml_insights("ml_insights.json")
            print("✅ Gemini Classifier initialized with ML insights")
        else:
            print("ℹ️  No ML insights found. LLM will work with default prompts.")
            
    except Exception as e:
        print(f"❌ Failed to initialize LLM classifier: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Gemini Email Spam Classifier API",
        "version": "1.0.0",
        "status": "ready" if llm_classifier else "error"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if llm_classifier else "unhealthy",
        "classifier_loaded": llm_classifier is not None,
        "ml_insights_loaded": llm_classifier.training_insights is not None if llm_classifier else False
    }

@app.post("/classify")
async def classify_email(email_text: str = Form(...)):
    """
    Classify a single email using the LLM.
    
    Args:
        email_text: The email text to classify
        
    Returns:
        Classification result with confidence and explanations
    """
    if not llm_classifier:
        raise HTTPException(status_code=500, detail="Classifier not initialized")
    
    try:
        result = await llm_classifier.classify_email(email_text)
        
        return {
            "email_preview": email_text[:100] + "..." if len(email_text) > 100 else email_text,
            "classification": result["classification"],
            "confidence": result["confidence"],
            "confidence_level": result.get("confidence_level", "MEDIUM"),
            "reasons": result["reasons"],
            "suspicious_elements": result["suspicious_elements"],
            "explanation": result["explanation"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify-batch")
async def classify_batch(emails: List[str] = Form(...)):
    """
    Classify multiple emails at once.
    
    Args:
        emails: List of email texts to classify
        
    Returns:
        List of classification results
    """
    if not llm_classifier:
        raise HTTPException(status_code=500, detail="Classifier not initialized")
    
    try:
        results = await llm_classifier.batch_classify(emails)
        
        # Format results for API response
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "email_id": i,
                "email_preview": emails[i][:100] + "..." if len(emails[i]) > 100 else emails[i],
                "classification": result["classification"],
                "confidence": result["confidence"],
                "confidence_level": result.get("confidence_level", "MEDIUM"),
                "reasons": result["reasons"][:3],  # Limit to top 3 reasons
                "suspicious_elements": result["suspicious_elements"][:3]  # Limit to top 3
            })
        
        return {
            "total_emails": len(emails),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@app.get("/insights")
async def get_insights():
    """Get information about the ML insights used by the LLM."""
    if not llm_classifier or not llm_classifier.training_insights:
        raise HTTPException(status_code=404, detail="ML insights not available")
    
    insights = llm_classifier.training_insights
    
    return {
        "model_accuracy": insights.model_accuracy,
        "total_samples": insights.total_samples,
        "spam_count": insights.spam_count,
        "ham_count": insights.ham_count,
        "top_spam_keywords": insights.spam_keywords[:10],
        "top_ham_keywords": insights.ham_keywords[:10],
        "spam_patterns": insights.spam_patterns[:5]
    }



if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run(app, host=host, port=port) 