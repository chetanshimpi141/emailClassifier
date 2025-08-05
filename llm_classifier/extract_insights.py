import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from collections import Counter
import re
from typing import Dict, List, Tuple

async def extract_ml_insights(spam_dir: str = "data/spam", ham_dir: str = "data/easy_ham") -> Dict:
    """
    Extract insights from the ML model training data.
    
    Args:
        spam_dir: Directory containing spam emails
        ham_dir: Directory containing ham emails
        
    Returns:
        Dictionary containing training insights
    """
    print("🔍 Extracting ML training insights...")
    
    # Load the trained model
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, joblib.load, "models/classifier.pkl")
        vectorizer = await loop.run_in_executor(None, joblib.load, "models/vectorizer.pkl")
        print("✅ Loaded trained model")
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first.")
        return {}
    
    # Load and preprocess data
    from src.preprocess import load_and_preprocess_data
    
    df = await loop.run_in_executor(None, load_and_preprocess_data, spam_dir, ham_dir)
    print(f"✅ Loaded {len(df)} emails ({len(df[df['label']==1])} spam, {len(df[df['label']==0])} ham)")
    
    # Extract feature names and importance
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = {}
    
    # Get feature importance from Naive Bayes
    if hasattr(model, 'feature_log_prob_'):
        # For MultinomialNB, we can use the log probabilities
        spam_log_probs = model.feature_log_prob_[1]  # Spam class
        ham_log_probs = model.feature_log_prob_[0]   # Ham class
        
        # Calculate importance as difference between spam and ham probabilities
        importance_scores = spam_log_probs - ham_log_probs
        
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = float(importance_scores[i])
    
    # Extract keywords by class
    spam_texts = df[df['label'] == 1]['clean_text'].tolist()
    ham_texts = df[df['label'] == 0]['clean_text'].tolist()
    
    # Get most important features for each class
    spam_keywords = extract_keywords(spam_texts, feature_importance, top_n=50)
    ham_keywords = extract_keywords(ham_texts, feature_importance, top_n=50)
    
    # Extract patterns
    spam_patterns = extract_patterns(spam_texts)
    ham_patterns = extract_patterns(ham_texts)
    
    # Calculate model accuracy (approximate)
    X = vectorizer.transform(df['clean_text'])
    y = df['label']
    accuracy = model.score(X, y)
    
    # Create insights dictionary
    insights = {
        "spam_keywords": spam_keywords,
        "ham_keywords": ham_keywords,
        "spam_patterns": spam_patterns,
        "ham_patterns": ham_patterns,
        "feature_importance": dict(sorted(feature_importance.items(), 
                                        key=lambda x: abs(x[1]), reverse=True)[:100]),
        "model_accuracy": accuracy,
        "total_samples": len(df),
        "spam_count": len(df[df['label'] == 1]),
        "ham_count": len(df[df['label'] == 0])
    }
    
    print(f"✅ Extracted insights:")
    print(f"   - Model accuracy: {accuracy:.2%}")
    print(f"   - Top spam keywords: {', '.join(spam_keywords[:10])}")
    print(f"   - Top ham keywords: {', '.join(ham_keywords[:10])}")
    
    return insights

def extract_keywords(texts: List[str], feature_importance: Dict[str, float], top_n: int = 50) -> List[str]:
    """Extract top keywords from texts based on feature importance."""
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    # Combine frequency with feature importance
    keyword_scores = {}
    for word, count in word_counts.items():
        if word in feature_importance:
            # Combine frequency with importance score
            score = count * abs(feature_importance[word])
            keyword_scores[word] = score
    
    # Return top keywords
    return [word for word, _ in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def extract_patterns(texts: List[str]) -> List[str]:
    """Extract common patterns from texts."""
    patterns = []
    
    # Common spam patterns
    spam_patterns = [
        r'\b(?:urgent|act now|limited time|offer|discount|free|money|earn|income|fortune)\b',
        r'\b(?:work from home|home based|opportunity|career)\b',
        r'\b(?:click here|visit|website|link)\b',
        r'\b(?:remove|unsubscribe|opt out)\b',
        r'\b(?:guaranteed|promise|assure|guarantee)\b',
        r'\b(?:investment|stock|trading|profit)\b',
        r'\b(?:loan|credit|debt|mortgage)\b',
        r'\b(?:pharmacy|medication|prescription)\b',
        r'\b(?:casino|gambling|poker|lottery)\b',
        r'\b(?:dating|romance|relationship)\b'
    ]
    
    # Count pattern occurrences
    pattern_counts = Counter()
    for text in texts:
        text_lower = text.lower()
        for pattern in spam_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                pattern_counts[pattern] += len(matches)
    
    # Return most common patterns
    return [pattern for pattern, _ in pattern_counts.most_common(10)]

def save_insights(insights: Dict, output_file: str = "ml_insights.json"):
    """Save insights to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(insights, f, indent=2)
    print(f"✅ Saved insights to {output_file}")

async def main():
    """Extract and save ML insights."""
    print("🚀 ML Insights Extractor")
    print("=" * 50)
    
    # Extract insights
    insights = await extract_ml_insights()
    
    if insights:
        # Save insights
        save_insights(insights)
        
        # Print summary
        print("\n📊 Insights Summary:")
        print(f"   - Model Accuracy: {insights['model_accuracy']:.2%}")
        print(f"   - Total Samples: {insights['total_samples']}")
        print(f"   - Spam Count: {insights['spam_count']}")
        print(f"   - Ham Count: {insights['ham_count']}")
        print(f"   - Top Spam Keywords: {', '.join(insights['spam_keywords'][:5])}")
        print(f"   - Top Ham Keywords: {', '.join(insights['ham_keywords'][:5])}")
        print(f"   - Spam Patterns: {', '.join(insights['spam_patterns'][:3])}")
        
        print("\n✅ Insights extraction completed successfully!")
    else:
        print("❌ Failed to extract insights.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 