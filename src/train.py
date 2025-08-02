import os
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_and_save_model(spam_dir, ham_dir):
    # Load and preprocess data
    df = load_and_preprocess_data(spam_dir, ham_dir)

    # Features and labels
    X = df['clean_text']
    y = df['label']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/classifier.pkl")
    joblib.dump(vectorizer, "../models/vectorizer.pkl")
    print("✅ Model and vectorizer saved to '../models/'")

if __name__ == "__main__":
    spam_path = "../data/spam"
    ham_path = "../data/easy_ham"
    train_and_save_model(spam_path, ham_path)
