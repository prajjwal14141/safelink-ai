import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import math
from collections import Counter
import os

app = Flask(__name__, static_folder=".", static_url_path="")  # Serve all files from main directory
CORS(app)  # Enable CORS for handling requests from other origins

# Function to calculate entropy
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

# Custom tokenizer function
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot += tempTokens
        allTokens += tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

# Function to train and save the model (only run once)
def train_and_save_model():
    if os.path.exists('logistic_regression_model.pkl') and os.path.exists('vectorizer.pkl'):
        print("Model already trained and saved.")
        return

    allurls = './data/data.csv'  # path to the CSV file
    allurlscsv = pd.read_csv(allurls, delimiter=',', on_bad_lines='skip')  # reading the file
    allurlsdata = np.array(allurlscsv)  # converting to a NumPy array
    random.shuffle(allurlsdata)  # shuffling the data

    y = [d[1] for d in allurlsdata]  # extracting labels (good or bad)
    corpus = [d[0] for d in allurlsdata]  # extracting URLs

    vectorizer = TfidfVectorizer(tokenizer=getTokens)  # using custom tokenizer
    X = vectorizer.fit_transform(corpus)  # vectorizing the URLs

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)  # increased max_iter for convergence
    model.fit(X_train, y_train)
    print(f"Model accuracy: {model.score(X_test, y_test)}")  # print model accuracy

    # Save the trained model and vectorizer to disk
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the pre-trained model and vectorizer
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Route for predicting URL classification
@app.route('/predict', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL not provided'}), 400
        
        # Transform the URL using the pre-trained vectorizer
        X_predict = [str(url)]
        X_predict = vectorizer.transform(X_predict)
        
        # Predict using the pre-trained model
        y_predict = model.predict(X_predict)
        
        response = {
            'url': url,
            'prediction': y_predict[0],
            'entropy': entropy(url)
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for serving the index page (your HTML/JS front-end)
@app.route('/')
def index():
    return send_from_directory(".", "index.html")  # Serve from the main directory

# Route to serve all static files (CSS, JS, images, linked pages)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(".", filename)

# Run the application
if __name__ == "__main__":
    train_and_save_model()  # Train and save the model only once
    model, vectorizer = load_model()  # Load the pre-trained model and vectorizer
    app.run(host='0.0.0.0', port=5000, debug=True)
