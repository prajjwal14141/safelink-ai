import os
import joblib
import math
import re
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime # To add timestamps
from collections import Counter
from flask import Flask, request, render_template, jsonify
import numpy as np
from flask_cors import CORS
import requests
# --- 1. Initialize Flask App ---
app = Flask(__name__, static_url_path='/static')

# --- Initialize Firebase Admin SDK ---
try:
    # Path to your service account key file
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client() # Get Firestore client instance
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    db = None # Set db to None if initialization fails

# --- 2. Enable CORS for the extension ---
CORS(app, resources={
    r"/analyze": {"origins": "chrome-extension://*"},
    r"/api/expand": {"origins": "*"} # Allow web page to call this
})
# --- 3. DEFINE HIGH-RISK TOKENS ---
# We can expand this list over time
HIGH_RISK_TOKENS = [
    'exe', 'php', 'install', 'toolbar', 'crack', 'spider', 'lucky',
    'admin', 'login', 'secure', 'account', 'password', 'key', 'download',
    'free', 'gift', 'prize', 'winner', 'click'
]

# --- 4. DEFINE HELPER FUNCTIONS ---

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


# --- 5. Load Pre-trained Model and Vectorizer ---
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


# --- 6. Define App Routes ---

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
        url_tokens = getTokens(url_clean) # Get tokens for the report
        
        X_predict = [url_clean] 
        X_predict_vec = vectorizer.transform(X_predict)
        
        y_Predict = lgs.predict(X_predict_vec)
        ai_prediction = str(y_Predict[0])
        is_malicious = (ai_prediction == 'bad')

        # --- 2. Build Threat Report ---
        threat_report = []
        if is_malicious:
            # Find which high-risk tokens are in this URL
            found_bad_tokens = [token for token in url_tokens if token in HIGH_RISK_TOKENS]
            for token in found_bad_tokens:
                threat_report.append(f"Contains suspicious token: '{token}'")
            
            # Check entropy
            if url_entropy > 4.0:
                threat_report.append(f"High randomness score: {url_entropy:.2f}")
            
            # Fallback if no specific tokens found
            if not threat_report:
                threat_report.append("Matches a general malicious URL pattern.")
        
        return jsonify({
            'url': url_raw,
            'ai_prediction': ai_prediction,
            'entropy': f"{url_entropy:.4f}",
            'is_malicious': is_malicious,
            'threat_report': threat_report # Send the new report
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/how-it-works')
def how_it_works():
    """Serves the explanation page."""
    return render_template('how-it-works.html')

# --- 7. NEW URL EXPANDER ROUTES ---

@app.route('/expander')
def expander_page():
    """Serves the URL Expander tool page."""
    return render_template('expander.html')


@app.route('/api/expand', methods=['POST'])
def api_expand():
    """API endpoint to expand a shortened URL."""
    try:
        data = request.get_json()
        short_url = data.get('url')
        if not short_url:
            return jsonify({'error': 'No URL provided.'}), 400

        # Add http:// if no protocol is present
        # Use regex for better matching
        if not re.match(r'^(?:http|ftp)s?://', short_url):
             short_url = 'http://' + short_url
             print(f"[Expander Debug] Prepended http:// to: {short_url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        print(f"[Expander Debug] Attempting to expand URL: {short_url}")

        response = requests.get(short_url, allow_redirects=True, timeout=10, headers=headers, stream=True)

        print(f"[Expander Debug] Status Code after redirects: {response.status_code}")
        print(f"[Expander Debug] Final URL from response.url: {response.url}")
        print(f"[Expander Debug] Redirect History Count: {len(response.history)}")

        response.close()

        # Check if the request was successful
        if not response.ok:
             print(f"[Expander Error] Request failed with status: {response.status_code}")
             return jsonify({'error': f'Request failed with status: {response.status_code}'}), response.status_code

        final_url = response.url

        # Compare after cleaning protocols for robustness
        cleaned_input = re.sub(r'^(?:http|ftp)s?://', '', short_url).strip('/')
        cleaned_final = re.sub(r'^(?:http|ftp)s?://', '', final_url).strip('/')
        if cleaned_input == cleaned_final and len(response.history) == 0:
             print("[Expander Warning] Final URL is same as input and no redirects occurred. Maybe not a short link?")
             # Return an error message indicating it couldn't be expanded
             return jsonify({'error': 'Could not expand the URL. It might not be a shortened link or the service blocked the request.'}), 400

        return jsonify({'final_url': final_url})

    except requests.exceptions.Timeout:
        print("[Expander Error] Request timed out.")
        return jsonify({'error': 'The request timed out. The server might be down or slow.'}), 504
    except requests.exceptions.ConnectionError as e:
        print(f"[Expander Error] Connection error - {e}")
        return jsonify({'error': 'Could not connect to the URL. Please check the link.'}), 500
    except requests.exceptions.TooManyRedirects:
         print("[Expander Error] Too many redirects.")
         return jsonify({'error': 'Too many redirects. The link may be a loop.'}), 500
    except requests.exceptions.RequestException as e: # Catch other requests errors
        print(f"[Expander Error] Request exception - {e}")
        return jsonify({'error': f'An error occurred while trying to reach the URL: {e}'}), 500
    except Exception as e:
        print(f"[Expander Error] Unknown exception - {e}")
        return jsonify({'error': 'An unknown error occurred while expanding the URL.'}), 500

# --- 8. NEW HISTORY PAGE ROUTE ---

@app.route('/history')
def history_page():
    """Serves the user's analysis history page."""
    return render_template('history.html')

# --- 9. NEW REPORT INCORRECT PREDICTION ROUTES ---

@app.route('/report')
def report_page():
    """Serves the 'Report Incorrect Prediction' page."""
    return render_template('report.html')

@app.route('/api/submit_report', methods=['POST'])
def api_submit_report():
    """API endpoint to receive user feedback and save to Firestore."""
    if not db: 
             print("[Firestore Error] 'db' object is None. Firebase initialization likely failed.") 
             return jsonify({'error': 'Database connection is not configured.'}), 500

    try:
        data = request.get_json()
        report_url = data.get('url')
        feedback = data.get('feedback') 
        comments = data.get('comments', '') 

        if not report_url or not feedback:
            return jsonify({'error': 'URL and feedback are required.'}), 400

        # Define the data to save
        report_data = {
            'url': report_url,
            'feedback': feedback,
            'comments': comments,
            'timestamp': datetime.now() 
        }

        
        reports_ref = db.collection('feedback_reports')
        reports_ref.add(report_data)

        print(f"Feedback report saved to Firestore: {report_url} ({feedback})") # Log success to server console

        # Send a success response back to the user
        return jsonify({'message': 'Thank you for your feedback! It has been recorded.'})

    except Exception as e:
        print(f"Error processing report or saving to Firestore: {e}")
        return jsonify({'error': 'An error occurred while submitting your report.'}), 500
    
# --- 10. Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)