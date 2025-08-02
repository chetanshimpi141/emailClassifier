import os
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return ' '.join(words)


def read_emails_from_folder(folder_path, label):
    """
    Reads all email files from the specified folder, extracts the body, and returns a list of dicts.
    Each dict contains 'text' and 'label'.
    """
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()

            # Split by first empty line to remove headers
            parts = content.split('\n\n', 1)
            body = parts[1] if len(parts) > 1 else parts[0]
            body = body.strip()

            if body:  # Skip empty bodies
                data.append({'text': body, 'label': label})
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    return data

def load_and_preprocess_data(spam_dir, ham_dir):
    spam_data = read_emails_from_folder(spam_dir, label=1)
    ham_data = read_emails_from_folder(ham_dir, label=0)

    all_data = spam_data + ham_data
    df = pd.DataFrame(all_data)

    # Apply cleaning
    df['clean_text'] = df['text'].apply(clean_text)

    return df

# Example usage
if __name__ == "__main__":
    spam_path = "../data/spam"
    ham_path = "../data/easy_ham"

    df = load_and_preprocess_data(spam_path, ham_path)
    print("Sample data:")
    print(df.sample(5))
    print(f"\nTotal records: {len(df)}")
    print(df['label'].value_counts())
