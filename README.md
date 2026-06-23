# Sentiment Analysis Dashboard

## System Design
This project provides an end-to-end NLP pipeline that classifies Amazon Alexa reviews as Positive or Negative. The system uses a machine learning model, stores results in an SQLite database, and provides a dashboard for visualization.

## Dataset
Amazon Alexa Reviews dataset (Kaggle). Contains text reviews and binary sentiment labels.

## NLP Pipeline & Model Selection
- **Preprocessing:** Regex cleaning, lowercase conversion, NLTK stopword removal, Tokenization.
- **Feature Extraction:** TF-IDF Vectorizer.
- **Models:** Compared Logistic Regression, Naive Bayes, and SVM. Logistic Regression was selected as the best model based on F1-score performance.

## Training Process
The models were trained on 80% of the data, evaluated with Accuracy, Precision, Recall, and F1-Score metrics.

## Installation
1. Install requirements: `pip install -r requirements.txt`
2. Run training: `python3 src/train.py`
3. Run dashboard: `streamlit run app.py`

## Features
- Interactive sentiment prediction.
- Real-time SQL logging of predictions.
- Sentiment distribution visualization.