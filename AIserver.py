import os
import joblib
import math
import re
from collections import Counter
from flask import Flask, request, render_template, jsonify
import numpy as np
from flask_cors import CORS
import requests
import json                     # <-- Import json
import firebase_admin           # <-- Import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# --- 1. Initialize Flask App ---
app = Flask(__name__, static_url_path='/static')

# --- 2. Initialize Firebase Admin SDK (Vercel Environment Variable Method) ---
try:
    # Get the JSON credentials string from the environment variable
    creds_json_str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not creds_json_str:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set.")

    # Parse the JSON string into a dictionary
    creds_dict = json.loads(creds_json_str)
    cred = credentials.Certificate(creds_dict)

    # Initialize Firebase app only if it hasn't been initialized yet (important for serverless)
    if not firebase_admin._apps:
         firebase_admin.initialize_app(cred)
    else:
         firebase_admin.get_app() # Get default app if already initialized

    db = firestore.client() # Get Firestore client instance
    print("Firebase Admin SDK initialized successfully via Environment Variable.")

except FileNotFoundError: # Should not happen with env var method, but good practice
     print("Error: Could not find credentials. Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON is set.")
     db = None
except ValueError as e: # Catch missing env var or invalid JSON
     print(f"Error initializing Firebase: {e}")
     db = None
except Exception as e:
    print(f"Unexpected error initializing Firebase Admin SDK: {e}")
    db = None

# --- 3. Enable CORS ---
CORS(app, resources={
    r"/analyze": {"origins": "chrome-extension://*"},
    r"/api/expand": {"origins": "*"},
    r"/api/submit_report": {"origins": "*"} # Allow web page to call this too
})

# --- 4. DEFINE HIGH-RISK TOKENS ---
HIGH_RISK_TOKENS = [
    'exe', 'php', 'install', 'toolbar', 'crack', 'spider', 'lucky',
    'admin', 'login', 'secure', 'account', 'password', 'key', 'download',
    'free', 'gift', 'prize', 'winner', 'click'
]

# --- 5. DEFINE HELPER FUNCTIONS ---

def clean_url(url):
    """ Cleans URL: removes protocol, www., trailing slash """
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
    """ Custom tokenizer: splits by '/', '-', '.' """
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
        common_tokens = ['com', 'www', 'http', 'https', 'org', 'net'] # Corrected https typo
        allTokens = [t for t in allTokens if t not in common_tokens and len(t) > 1]
        return allTokens
    except Exception as e:
        print(f"Error tokenizing input {input}: {e}")
        return []

def entropy(s):
    """ Calculates Shannon entropy """
    p, lns = Counter(s), float(len(s))
    if lns == 0: return 0.0
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

# --- 6. Load Pre-trained Model and Vectorizer ---
MODEL_URL = "https://github.com/prajjwal14141/safelink-ai/releases/download/v1.0.0/model.pkl"
VECTORIZER_URL = "https://github.com/prajjwal14141/safelink-ai/releases/download/v1.0.0/vectorizer.pkl"
MODEL_PATH = "/tmp/model.pkl"
VECTORIZER_PATH = "/tmp/vectorizer.pkl"

vectorizer = None
lgs = None

# --- Function to download a file ---
def download_file(url, destination):
    print(f"Downloading {os.path.basename(destination)} from {url}...")
    try:
        # Use User-Agent to avoid potential blocks from GitHub/CDNs
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        with requests.get(url, stream=True, timeout=90, headers=headers) as r: # Increased timeout further
            r.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)
            with open(destination, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
        print(f"{os.path.basename(destination)} downloaded successfully ({downloaded_size / (1024*1024):.2f} MB).")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {os.path.basename(destination)}: {e}")
        # Attempt to delete partial file if download failed
        if os.path.exists(destination):
            try: os.remove(destination)
            except OSError: pass # Ignore error if file cannot be deleted
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download of {os.path.basename(destination)}: {e}")
        if os.path.exists(destination):
            try: os.remove(destination)
            except OSError: pass
        return False

# --- Load or Download Logic ---
try:
    # Check if files already exist in /tmp (for warm starts)
    model_exists = os.path.exists(MODEL_PATH)
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)

    # Download Vectorizer if missing
    if not vectorizer_exists:
        if not download_file(VECTORIZER_URL, VECTORIZER_PATH):
             raise RuntimeError(f"Failed to download vectorizer from {VECTORIZER_URL}")
    print("Loading vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Download Model if missing
    if not model_exists:
         if not download_file(MODEL_URL, MODEL_PATH):
             raise RuntimeError(f"Failed to download model from {MODEL_URL}")
    print("Loading model...")
    lgs = joblib.load(MODEL_PATH)

    if vectorizer and lgs:
         print("Model and vectorizer ready.")
    else:
         # This case should ideally be caught by exceptions during download/load
         raise RuntimeError("Model or vectorizer failed to load after download attempt.")

except RuntimeError as e: # Catch specific download/load failures
    print(f"FATAL ERROR during model setup: {e}")
    vectorizer = None
    lgs = None
except Exception as e:
    # Catch any other unexpected errors during the process
    print(f"FATAL ERROR loading/downloading models: {e}")
    vectorizer = None
    lgs = None
# --- 7. Define App Routes ---

@app.route('/')
def home():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """ API endpoint to analyze a URL. """
    if not lgs or not vectorizer:
        return jsonify({'error': 'AI model is not loaded. Please check server logs.'}), 500
    try:
        data = request.get_json()
        url_raw = data.get('url')
        if not url_raw: return jsonify({'error': 'No URL provided.'}), 400

        url_clean = clean_url(url_raw)
        if not url_clean: return jsonify({'error': 'Invalid URL provided.'}), 400

        url_entropy = entropy(url_clean)
        url_tokens = getTokens(url_clean)
        X_predict = [url_clean]
        X_predict_vec = vectorizer.transform(X_predict)
        y_Predict = lgs.predict(X_predict_vec)
        ai_prediction = str(y_Predict[0])
        is_malicious = (ai_prediction == 'bad')

        threat_report = []
        if is_malicious:
            found_bad_tokens = [token for token in url_tokens if token in HIGH_RISK_TOKENS]
            for token in found_bad_tokens: threat_report.append(f"Contains suspicious token: '{token}'")
            if url_entropy > 4.0: threat_report.append(f"High randomness score: {url_entropy:.2f}")
            if not threat_report: threat_report.append("Matches a general malicious URL pattern.")

        return jsonify({
            'url': url_raw, 'ai_prediction': ai_prediction,
            'entropy': f"{url_entropy:.4f}", 'is_malicious': is_malicious,
            'threat_report': threat_report
        })
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': 'An internal server error occurred during analysis.'}), 500

@app.route('/how-it-works')
def how_it_works():
    """ Serves the explanation page. """
    return render_template('how-it-works.html')

@app.route('/expander')
def expander_page():
    """ Serves the URL Expander tool page. """
    return render_template('expander.html')

@app.route('/api/expand', methods=['POST'])
def api_expand():
    """ API endpoint to expand a shortened URL. """
    try:
        data = request.get_json()
        short_url = data.get('url')
        if not short_url: return jsonify({'error': 'No URL provided.'}), 400
        if not re.match(r'^(?:http|ftp)s?://', short_url): short_url = 'http://' + short_url
        headers = {'User-Agent': 'Mozilla/5.0 ...'} # Keep your User-Agent
        print(f"[Expander Debug] Attempting to expand URL: {short_url}")
        response = requests.get(short_url, allow_redirects=True, timeout=10, headers=headers, stream=True)
        print(f"[Expander Debug] Status: {response.status_code}, Final URL: {response.url}, History: {len(response.history)}")
        response.close()
        if not response.ok: return jsonify({'error': f'Request failed with status: {response.status_code}'}), response.status_code
        final_url = response.url
        cleaned_input = re.sub(r'^(?:http|ftp)s?://', '', short_url).strip('/')
        cleaned_final = re.sub(r'^(?:http|ftp)s?://', '', final_url).strip('/')
        if cleaned_input == cleaned_final and len(response.history) == 0:
             return jsonify({'error': 'Could not expand URL. May not be a short link or request blocked.'}), 400
        return jsonify({'final_url': final_url})
    except requests.exceptions.Timeout: return jsonify({'error': 'Request timed out.'}), 504
    except requests.exceptions.ConnectionError as e: return jsonify({'error': 'Could not connect.'}), 500
    except requests.exceptions.TooManyRedirects: return jsonify({'error': 'Too many redirects.'}), 500
    except requests.exceptions.RequestException as e: return jsonify({'error': f'Request error: {e}'}), 500
    except Exception as e: print(f"Error expanding URL: {e}"); return jsonify({'error': 'Unknown error expanding URL.'}), 500

@app.route('/history')
def history_page():
    """ Serves the History page. """
    return render_template('history.html')

@app.route('/report')
def report_page():
    """ Serves the 'Report Incorrect Prediction' page. """
    return render_template('report.html')

@app.route('/api/submit_report', methods=['POST'])
def api_submit_report():
    """ API endpoint to receive user feedback and save to Firestore. """
    if not db:
         print("[Firestore Error] 'db' object is None. Firebase initialization likely failed.")
         return jsonify({'error': 'Database connection is not configured.'}), 500
    try:
        data = request.get_json()
        report_url = data.get('url')
        feedback = data.get('feedback')
        comments = data.get('comments', '')
        if not report_url or not feedback: return jsonify({'error': 'URL and feedback are required.'}), 400
        report_data = {'url': report_url, 'feedback': feedback, 'comments': comments, 'timestamp': datetime.now()}
        reports_ref = db.collection('feedback_reports')
        reports_ref.add(report_data)
        print(f"Feedback report saved to Firestore: {report_url} ({feedback})")
        return jsonify({'message': 'Thank you for your feedback! It has been recorded.'})
    except Exception as e:
        print(f"Error processing report or saving to Firestore: {e}")
        return jsonify({'error': 'An error occurred while submitting your report.'}), 500

# --- 8. Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Use debug=True for local testing if needed, but False for Vercel build
    app.run(host='0.0.0.0', port=port, debug=False)