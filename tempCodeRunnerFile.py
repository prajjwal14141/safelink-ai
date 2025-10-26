import os
import joblib
import math
import re
from collections import Counter
from flask import Flask, request, render_template, jsonify
import numpy as np

# --- 1. Initialize Flask App ---
app = Flask(__name__, static_url_path='/static')

# --- 2. DEFINE HELPER FUNCTIONS ---

def clean_url(url):
    """
    Cleans a URL by removing protocol (http/https), 'www.',
    and trailing slashes.
    """
    try:
        url = str(url)
        url = re.sub(r'^(https?|ftp)://', '', url)
        url = re.sub(r'^www\.', '', url)
        url = re.sub(r'/$', '', url)
        return url
    except Exception as e:
        print(f"Error cleaning URL {url}: {e}")
        return ""

def getTokens(input):
    """
    A custom tokenizer that splits URLs by '/', '-', and '.'
    """
    try:
        tokensBySlash = str(input).split('/')
        allTokens = []
        for i in tokensBySlash:
            tokens = str(i).split('-')
            tokensByDot = []
            for j in range(0, len(tokens)):
                tempTokens = str(tokens[j]).split('.')
                tokensByDot = tokensByDot + tempTokens
            allTokens = allTokens + tokens + tokensByDot
        
        allTokens = list(set(allTokens))
        common_tokens = ['com', 'www', 'http', 'httpsV', 'org', 'net']
        allTokens = [t for t in allTokens if t not in common_tokens]
        return allTokens
    except Exception as e:
        print(f"Error tokenizing input {input}: {e}")
        return []

def entropy(s):
    """Calculates the Shannon entropy of a string."""
    p, lns = Counter(s), float(len(s))
    if lns == 0:
        return 0.0
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


# --- 3. Load Pre-trained Model and Vectorizer ---
try:
    vectorizer = joblib.load('vectorizer.pkl')
    lgs = joblib.load('model.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    vectorizer = None
    lgs = None
except Exception as e:
    print(f"Error loading models: {e}")
    vectorizer = None
    lgs = None


# --- 4. Define App Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze a URL."""
    if not lgs or not vectorizer:
        return jsonify({'error': 'AI model is not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json()
        url_raw = data.get('url')
        if not url_raw:
            return jsonify({'error': 'No URL provided.'}), 400

        # --- 1. Our AI Model Prediction ---
        url_clean = clean_url(url_raw)
        if not url_clean:
             return jsonify({'error': 'Invalid URL provided.'}), 400

        url_entropy = entropy(url_clean)
        X_predict = [url_clean] 
        X_predict_vec = vectorizer.transform(X_predict)
        
        y_Predict = lgs.predict(X_predict_vec)
        ai_prediction = str(y_Predict[0])

        # --- 2. Final Verdict ---
        is_malicious = (ai_prediction == 'bad')

        return jsonify({
            'url': url_raw,
            'ai_prediction': ai_prediction,
            'entropy': f"{url_entropy:.4f}",
            'is_malicious': is_malicious
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/how-it-works')
def how_it_works():
    """Serves the explanation page."""
    return render_template('how-it-works.html')


# --- 5. Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

