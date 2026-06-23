import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def preprocess_data(path):
    df = pd.read_csv(path, sep='\t')
    df = df.dropna()
    df['cleaned_review'] = df['verified_reviews'].apply(clean_text)
    print("Preprocessing complete!")
    print(df[['verified_reviews', 'cleaned_review']].head())
    return df

if __name__ == "__main__":
    file_path = 'data/amazon_alexa.tsv'
    if os.path.exists(file_path):
        df = preprocess_data(file_path)
    else:
        print(f"File not found: {file_path}")