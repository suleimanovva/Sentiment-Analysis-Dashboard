import streamlit as st
import joblib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

conn = sqlite3.connect('sentiment.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS results (text TEXT, sentiment TEXT, confidence REAL, timestamp TEXT)')
conn.commit()

def main():
    st.title("Sentiment Analysis Dashboard")
    
    vectorizer = joblib.load('models/vectorizer.pkl')
    model = joblib.load('models/best_model.pkl')
    
    user_input = st.text_input("Enter a review:")
    
    if st.button("Predict"):
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        confidence = 0.95 
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
      
        c.execute("INSERT INTO results VALUES (?,?,?,?)", (user_input, sentiment, confidence, datetime.now()))
        conn.commit()
        st.write(f"Prediction: {sentiment}")

    df = pd.read_sql_query("SELECT * FROM results", conn)
    if not df.empty:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        st.pyplot(fig)

if __name__ == "__main__":
    main()