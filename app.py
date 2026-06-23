import streamlit as st
import sqlite3
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Models not found! Please run `python3 src/train.py` first.")
    st.stop()

def init_db():
    conn = sqlite3.connect('sentiment_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            predicted_sentiment TEXT,
            confidence_score REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_prediction(text, sentiment, confidence):
    conn = sqlite3.connect('sentiment_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (review_text, predicted_sentiment, confidence_score, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (text, sentiment, confidence, timestamp))
    conn.commit()
    conn.close()

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("📊 NLP Project: Sentiment Analysis Dashboard")
st.write("Analyze customer reviews, visualize sentiment trends, and store predictions in an SQL database.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Test a New Review")
    user_input = st.text_area("Customer Review:", "This product is absolutely amazing, highly recommended!")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            text_vec = vectorizer.transform([user_input])
            prediction = model.predict(text_vec)[0]
            
            try:
                probabilities = model.predict_proba(text_vec)[0]
                confidence = max(probabilities) * 100
            except AttributeError:
                confidence = 100.0
            
            sentiment_label = "Positive" if prediction == 1 else "Negative"
            
            save_prediction(user_input, sentiment_label, round(confidence, 2))
            
            if sentiment_label == "Positive":
                st.success(f"**Sentiment:** {sentiment_label} | **Confidence:** {confidence:.2f}%")
            else:
                st.error(f"**Sentiment:** {sentiment_label} | **Confidence:** {confidence:.2f}%")
        else:
            st.warning("Please enter a review first.")

with col2:
    st.subheader("Model Comparison")
    models_data = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'SVM'],
        'Accuracy': [0.91, 0.90, 0.91]
    }
    df_models = pd.DataFrame(models_data)
    fig_models, ax_models = plt.subplots(figsize=(5, 3))
    ax_models.bar(df_models['Model'], df_models['Accuracy'], color=['blue', 'orange', 'green'])
    ax_models.set_ylim(0.85, 1.0)
    ax_models.set_ylabel('Accuracy')
    st.pyplot(fig_models)

st.markdown("---")
st.subheader("Sentiment Statistics (From Database)")

conn = sqlite3.connect('sentiment_data.db')
df_history = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

if not df_history.empty:
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.write("Recent Database Records")
        st.dataframe(df_history.tail(5))
        
    with col4:
        st.write("Sentiment Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        df_history['predicted_sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax_pie, colors=['lightgreen', 'salmon'])
        ax_pie.set_ylabel('')
        st.pyplot(fig_pie)
else:
    st.info("No predictions in the database yet. Test a review to see statistics.")