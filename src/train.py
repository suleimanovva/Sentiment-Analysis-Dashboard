import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_models():
    if not os.path.exists('models'):
        os.makedirs('models')
        
    df = pd.read_csv('data/amazon_alexa.tsv', sep='\t')
    df = df.dropna()
    
    X = df['verified_reviews']
    y = df['feedback']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(class_weight='balanced', probability=True)
    }
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        print(f"--- {name} ---")
        print(classification_report(y_test, predictions))
        
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(models["Logistic Regression"], 'models/best_model.pkl')
    print("Models saved successfully!")

if __name__ == "__main__":
    train_models()