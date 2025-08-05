from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import shutil
from typing import List, Optional
import json

# Import our email classifier modules
from src.train import train_and_save_model
from src.predict import predict_email, predict_batch
from src.preprocess import load_and_preprocess_data

app = FastAPI(
    title="Email Spam Classifier API",
    description="A FastAPI-based email spam classification service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/unified_interface.html")

@app.get("/original")
async def original_interface():
    """Serve the original HTML page."""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Email Spam Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "train": "/train",
            "predict": "/predict",
            "predict-batch": "/predict-batch",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": os.path.exists("models/classifier.pkl") and os.path.exists("models/vectorizer.pkl")
    }

@app.post("/train")
async def train_model(
    spam_files: List[UploadFile] = File(...),
    ham_files: List[UploadFile] = File(...)
):
    """
    Train the email classifier with uploaded spam and ham files.
    
    Args:
        spam_files: List of spam email files
        ham_files: List of ham (legitimate) email files
    
    Returns:
        Training results and model performance metrics
    """
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            spam_dir = os.path.join(temp_dir, "spam")
            ham_dir = os.path.join(temp_dir, "ham")
            os.makedirs(spam_dir)
            os.makedirs(ham_dir)
            
            # Save uploaded spam files
            for file in spam_files:
                if file.filename:
                    file_path = os.path.join(spam_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
            
            # Save uploaded ham files
            for file in ham_files:
                if file.filename:
                    file_path = os.path.join(ham_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
            
            # Train the model in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, train_and_save_model, spam_dir, ham_dir)
            
            # Load data to get statistics
            df = await loop.run_in_executor(None, load_and_preprocess_data, spam_dir, ham_dir)
            
            return {
                "message": "Model trained successfully",
                "training_stats": {
                    "total_emails": len(df),
                    "spam_count": len(df[df['label'] == 1]),
                    "ham_count": len(df[df['label'] == 0]),
                    "spam_files_uploaded": len(spam_files),
                    "ham_files_uploaded": len(ham_files)
                },
                "model_files": {
                    "classifier": "models/classifier.pkl",
                    "vectorizer": "models/vectorizer.pkl"
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict_single_email(email_text: str = Form(...)):
    """
    Predict whether a single email is spam or ham.
    
    Args:
        email_text: The email text to classify
    
    Returns:
        Prediction result with confidence score
    """
    try:
        if not os.path.exists("models/classifier.pkl"):
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using /train endpoint."
            )
        
        # Run prediction in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        prediction, probability = await loop.run_in_executor(None, predict_email, email_text)
        
        return {
            "email_text": email_text[:100] + "..." if len(email_text) > 100 else email_text,
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence": float(probability[1] if prediction == 1 else probability[0]),
            "probabilities": {
                "ham": float(probability[0]),
                "spam": float(probability[1])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_multiple_emails(emails: List[str] = Form(...)):
    """
    Predict multiple emails at once.
    
    Args:
        emails: List of email texts to classify
    
    Returns:
        List of prediction results
    """
    try:
        if not os.path.exists("models/classifier.pkl"):
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first using /train endpoint."
            )
        
        # Run batch prediction in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, predict_batch, emails)
        
        predictions = []
        for i, (email, (prediction, probability)) in enumerate(zip(emails, results)):
            predictions.append({
                "email_id": i,
                "email_text": email[:100] + "..." if len(email) > 100 else email,
                "prediction": "spam" if prediction == 1 else "ham",
                "confidence": float(probability[1] if prediction == 1 else probability[0]),
                "probabilities": {
                    "ham": float(probability[0]),
                    "spam": float(probability[1])
                }
            })
        
        return {
            "total_emails": len(emails),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model."""
    try:
        model_exists = os.path.exists("models/classifier.pkl")
        vectorizer_exists = os.path.exists("models/vectorizer.pkl")
        
        info = {
            "model_loaded": model_exists and vectorizer_exists,
            "model_files": {
                "classifier": model_exists,
                "vectorizer": vectorizer_exists
            }
        }
        
        if model_exists:
            import joblib
            import asyncio
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, joblib.load, "models/classifier.pkl")
            info["model_type"] = type(model).__name__
            info["model_params"] = str(model.get_params())
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 