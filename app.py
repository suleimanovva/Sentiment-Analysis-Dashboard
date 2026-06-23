import streamlit as st
import joblib

def main():
    st.title("Sentiment Analysis Dashboard")
    
    vectorizer = joblib.load('models/vectorizer.pkl')
    model = joblib.load('models/best_model.pkl')
    
    user_input = st.text_input("Enter a review:")
    
    if st.button("Predict"):
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")

if __name__ == "__main__":
    main()