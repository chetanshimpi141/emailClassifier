# Email Spam Classifier

A machine learning-based email spam classifier built with Python, scikit-learn, and NLTK. This project can classify emails as either spam or legitimate (ham) using a Naive Bayes classifier. Now available as a FastAPI web service with a beautiful web interface!

## Features

- **Text Preprocessing**: Cleans email text by removing HTML tags, URLs, email addresses, and stopwords
- **Machine Learning Model**: Uses Multinomial Naive Bayes for classification
- **FastAPI Web Service**: RESTful API with automatic documentation
- **Beautiful Web Interface**: Modern, responsive HTML frontend
- **Easy-to-use API**: Simple functions for training and prediction
- **Batch Processing**: Support for classifying multiple emails at once
- **Model Persistence**: Saves trained models for reuse
- **File Upload Support**: Train models by uploading email files
- **Real-time Predictions**: Instant email classification results

## Project Structure

```
emailClassifier/
├── src/
│   ├── preprocess.py      # Text preprocessing functions
│   ├── train.py          # Model training script
│   └── predict.py        # Prediction functions
├── data/
│   ├── spam/             # Spam email files
│   └── easy_ham/         # Legitimate email files
├── models/               # Saved model files
├── static/
│   └── index.html        # Web interface
├── app.py                # FastAPI application
├── start_server.py       # Server startup script
├── requirements.txt      # Python dependencies
├── test_email_classifier.py  # Test script
├── test_api.py           # API test script
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emailClassifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start (Web Interface)

1. Start the FastAPI server:
```bash
python start_server.py
```

2. Open your browser and go to: `http://localhost:8000`

3. Use the web interface to:
   - Upload spam and ham email files to train the model
   - Classify individual emails
   - Perform batch predictions on multiple emails

## API Documentation

Once the server is running, you can access:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative API Docs**: `http://localhost:8000/redoc`
- **API Info**: `http://localhost:8000/api`

## Usage

### Web Interface (Recommended)

The easiest way to use the email classifier is through the web interface:

1. Start the server: `python start_server.py`
2. Open `http://localhost:8000` in your browser
3. Upload email files and start classifying!

### Programmatic Usage

#### Training the Model

To train the model with your email data:

```python
from src.train import train_and_save_model

# Train the model
train_and_save_model("data/spam", "data/easy_ham")
```

Or run the training script directly:
```bash
python src/train.py
```

### Making Predictions

```python
from src.predict import predict_email

# Classify a single email
email_text = "Hello, this is a legitimate email from your bank."
prediction, probability = predict_email(email_text)

if prediction == 1:
    print("SPAM")
else:
    print("HAM")
print(f"Confidence: {probability.max():.2f}")
```

### Batch Predictions

```python
from src.predict import predict_batch

emails = [
    "Hello, this is a legitimate email.",
    "URGENT: You've won $1,000,000!",
    "Meeting reminder for tomorrow."
]

results = predict_batch(emails)
for email, (pred, prob) in zip(emails, results):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{email[:30]}... -> {label}")
```

## Data Format

The classifier expects email data in the following format:

- **Spam emails**: Place spam email files in `data/spam/`
- **Ham emails**: Place legitimate email files in `data/easy_ham/`

Each email file should contain the email body text. The system automatically:
- Removes email headers (everything before the first empty line)
- Cleans the text (removes HTML, URLs, special characters)
- Removes stopwords

## Model Performance

The model typically achieves:
- **Accuracy**: 95%+ on test data
- **Precision**: High precision for both spam and ham detection
- **Recall**: Good recall for both classes

## Testing

### Test the Core Functionality

Run the test script to verify everything works:

```bash
python test_email_classifier.py
```

This will test:
- Text preprocessing
- Model training
- Prediction functionality

### Test the API

Test the FastAPI endpoints:

```bash
python test_api.py
```

This will test:
- Health check endpoint
- API information
- Model training via API
- Single and batch predictions
- Model information endpoint

## Dependencies

- `pandas`: Data manipulation
- `nltk`: Natural language processing
- `scikit-learn`: Machine learning algorithms
- `joblib`: Model serialization
- `fastapi`: Web framework for building APIs
- `uvicorn`: ASGI server for running FastAPI
- `python-multipart`: File upload support

## Customization

### Adding New Features

To add new text features, modify the `clean_text()` function in `src/preprocess.py`:

```python
def clean_text(text):
    # Add your custom preprocessing steps here
    # ...
    return cleaned_text
```

### Using Different Models

To use a different classifier, modify `src/train.py`:

```python
from sklearn.ensemble import RandomForestClassifier

# Replace MultinomialNB with your preferred model
model = RandomForestClassifier()
```

## Troubleshooting

### NLTK Data Not Found
If you encounter NLTK data errors, the script will automatically download required data. If issues persist:

```python
import nltk
nltk.download('stopwords')
```

### Model Files Not Found
Ensure you've run the training script first:
```bash
python src/train.py
```

### API Server Issues
If the FastAPI server won't start:
1. Check if port 8000 is available
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify all required files exist: `python start_server.py`

### Memory Issues
For large datasets, consider:
- Using a subset of data for training
- Implementing data streaming
- Using more memory-efficient vectorizers

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on the project repository. 