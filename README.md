# Sentiment Analysis Dashboard

## System Design
This project is an end-to-end NLP system built to classify Amazon Alexa reviews. The pipeline includes text preprocessing, feature extraction using TF-IDF, and machine learning classification. All predictions are logged into an SQLite database for trend analysis.

## Dataset Description
The system utilizes the **Amazon Alexa Reviews** dataset (Kaggle), consisting of customer reviews and binary sentiment labels.

## Model Selection & Training Process
We implemented and compared three models: **Logistic Regression**, **Naive Bayes**, and **Support Vector Machine (SVM)**. 
- **Preprocessing:** Text cleaning, lowercase conversion, NLTK stopword removal, and tokenization.
- **Feature Extraction:** TF-IDF Vectorizer.
- **Optimization:** Used `class_weight='balanced'` to handle class imbalance, significantly improving F1-score for negative reviews.

## Evaluation Results (Logistic Regression)
| Metric | Class 0 (Negative) | Class 1 (Positive) |
| :--- | :--- | :--- |
| Precision | 0.52 | 0.97 |
| Recall | 0.78 | 0.92 |
| F1-Score | 0.62 | 0.95 |

## Installation Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python3 src/train.py`
3. Run the dashboard: `streamlit run app.py`

## Dashboard Screenshots
![Streamlit-Dashboard](image.png)