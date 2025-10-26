import pandas as pd
import numpy as np
from utils import clean_url, getTokens 
import random
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


def TL():
    """Trains the model and saves it to disk."""
    allurls = './data/data.csv' 
    try:
        allurlscsv = pd.read_csv(allurls, delimiter=',', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {allurls}")
        print("Please make sure 'data.csv' is in a folder named 'data'")
        return None, None

    allurlsdata = pd.DataFrame(allurlscsv)
    allurlsdata = np.array(allurlsdata)
    random.shuffle(allurlsdata)

    y = [d[1] for d in allurlsdata]
    corpus_raw = [d[0] for d in allurlsdata]
    
    print(f"Loaded {len(corpus_raw)} URLs from {allurls}")

    # --- THIS IS THE NEW STEP ---
    # Clean every URL in the corpus before training
    print("Cleaning and normalizing URLs...")
    corpus_clean = [clean_url(url) for url in corpus_raw]
    
    print("Vectorizing data using TfidfVectorizer...")
    vectorizer = TfidfVectorizer(tokenizer=getTokens)
    X = vectorizer.fit_transform(corpus_clean) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    lgs = LogisticRegression(max_iter=1000) 
    lgs.fit(X_train, y_train)
    
    accuracy = lgs.score(X_test, y_test)
    print(f"MODEL ACCURACY: {accuracy*100:.2f}%")
    
    return vectorizer, lgs

if __name__ == "__main__":
    print("Starting model training...")
    vectorizer, lgs = TL()
    
    if vectorizer and lgs:
        # Save the vectorizer and model
        joblib.dump(vectorizer, 'vectorizer.pkl')
        print("Vectorizer saved to vectorizer.pkl")
        joblib.dump(lgs, 'model.pkl')
        print("Model saved to model.pkl")
        print("\nTraining complete.")

