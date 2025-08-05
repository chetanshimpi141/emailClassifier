import joblib
import os
import sys
sys.path.append('..')
from src.preprocess import clean_text

def load_model_and_vectorizer():
    """Load the trained model and vectorizer from the models directory."""
    model_path = "models/classifier.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model files not found. Please run train.py first.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

def predict_email(email_text):
    """
    Predict whether an email is spam (1) or ham (0).
    
    Args:
        email_text (str): The email text to classify
        
    Returns:
        tuple: (prediction, probability)
            - prediction: 1 for spam, 0 for ham
            - probability: confidence score
    """
    model, vectorizer = load_model_and_vectorizer()
    
    # Clean the text
    clean_email = clean_text(email_text)
    
    # Vectorize the text
    email_vector = vectorizer.transform([clean_email])
    
    # Make prediction
    prediction = model.predict(email_vector)[0]
    probability = model.predict_proba(email_vector)[0]
    
    return prediction, probability

def predict_batch(email_texts):
    """
    Predict multiple emails at once.
    
    Args:
        email_texts (list): List of email texts to classify
        
    Returns:
        list: List of tuples (prediction, probability) for each email
    """
    model, vectorizer = load_model_and_vectorizer()
    
    # Clean all texts
    clean_texts = [clean_text(text) for text in email_texts]
    
    # Vectorize all texts
    email_vectors = vectorizer.transform(clean_texts)
    
    # Make predictions
    predictions = model.predict(email_vectors)
    probabilities = model.predict_proba(email_vectors)
    
    return [(pred, prob) for pred, prob in zip(predictions, probabilities)]

if __name__ == "__main__":
    # Example usage
    sample_emails = [
        "Hello, this is a legitimate email from your bank.",
        "URGENT: You've won $1,000,000! Click here to claim your prize!",
        "Meeting reminder for tomorrow at 2 PM."
    ]
    
    print("Email Classification Results:")
    print("=" * 50)
    
    for i, email in enumerate(sample_emails, 1):
        prediction, probability = predict_email(email)
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        print(f"Email {i}: {label} (confidence: {confidence:.2f})")
        print(f"Text: {email[:50]}...")
        print("-" * 50)
