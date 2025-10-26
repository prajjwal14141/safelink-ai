import os
import joblib
import math
import re                      # Keep re import
from collections import Counter # Keep Counter import
from flask import Flask, request, render_template, jsonify
import numpy as np             # Keep numpy import
from flask_cors import CORS
import requests
# DO NOT import json here if only used for Firebase env var method
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# --- Import helper functions from utils.py ---
# Ensure utils.py exists in your project root
try:
    from utils import clean_url, getTokens, entropy, HIGH_RISK_TOKENS
    print("Successfully imported from utils.py")
except ImportError:
    print("FATAL ERROR: Could not import from utils.py. Make sure utils.py exists.")
    # Define fallbacks or exit if utils are absolutely critical at startup
    # For now, let it proceed and potentially fail later if functions are called
    clean_url = lambda x: x # Dummy functions
    getTokens = lambda x: []
    entropy = lambda x: 0
    HIGH_RISK_TOKENS = []


# --- 1. Initialize Flask App ---
app = Flask(__name__, static_url_path='/static')

# --- 2. Initialize Firebase Admin SDK (Render Secret File Method) ---
db = None # Initialize db globally
try:
    # Path to your service account key file (Render places it in the root)
    # Check current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for serviceAccountKey.json in {os.path.abspath('.')}")

    cred_path = "serviceAccountKey.json"
    if not os.path.exists(cred_path):
         # Log a warning but don't crash, allow app to run without Firestore
         print(f"Warning: serviceAccountKey.json not found at '{os.path.abspath(cred_path)}'. Ensure it was added as a Secret File on Render. Firestore features disabled.")
    else:
        print("serviceAccountKey.json found. Initializing Firebase...")
        cred = credentials.Certificate(cred_path)
        # Initialize only if not already done
        if not firebase_admin._apps:
             firebase_admin.initialize_app(cred)
             print("Firebase Admin SDK initialized successfully via Secret File.")
        else:
             firebase_admin.get_app()
             print("Firebase Admin SDK already initialized.")
        db = firestore.client()

except Exception as e:
    # Log the full error for better debugging
    import traceback
    print(f"Unexpected error initializing Firebase Admin SDK: {e}")
    print(traceback.format_exc()) # Print stack trace
    print("Firestore features disabled.")
    db = None

# --- 3. Enable CORS ---
CORS(app, resources={
    r"/analyze": {"origins": "chrome-extension://*"},
    r"/api/expand": {"origins": "*"},
    r"/api/submit_report": {"origins": "*"}
})


MODEL_URL = "https://github.com/prajjwal14141/safelink-ai/releases/download/v1.0.0/model.pkl" 
VECTORIZER_URL = "https://github.com/prajjwal14141/safelink-ai/releases/download/v1.0.0/vectorizer.pkl" 

MODEL_PATH = "/tmp/model.pkl"
VECTORIZER_PATH = "/tmp/vectorizer.pkl"

vectorizer = None
lgs = None

def download_file(url, destination):
    print(f"Downloading {os.path.basename(destination)} from {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        if not url or not url.startswith(('http://', 'https://')): raise ValueError(f"Invalid download URL: {url}")
        with requests.get(url, stream=True, timeout=90, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk); downloaded_size += len(chunk)
        print(f"{os.path.basename(destination)} downloaded successfully ({downloaded_size / (1024*1024):.2f} MB).")
        return True
    except requests.exceptions.RequestException as e: print(f"Error downloading {os.path.basename(destination)}: {e}")
    except ValueError as e: print(f"Error: {e}")
    except Exception as e: print(f"Unexpected error during download of {os.path.basename(destination)}: {e}")
    if os.path.exists(destination):
        try: os.remove(destination);
        except OSError: pass
    return False

# --- Load or Download Logic ---
try:
    # Ensure /tmp exists (usually does on Render, but good check)
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
        print("Created /tmp directory.")

    model_exists = os.path.exists(MODEL_PATH)
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)

    # Download Vectorizer if missing
    if not vectorizer_exists:
        print(f"{VECTORIZER_PATH} not found.")
        if not download_file(VECTORIZER_URL, VECTORIZER_PATH): raise RuntimeError(f"Failed to download vectorizer from {VECTORIZER_URL}")
    print(f"Loading vectorizer from {VECTORIZER_PATH}...")
    try:
        # Explicitly load using joblib
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Vectorizer loaded via joblib.")
    except Exception as load_err:
        print(f"joblib.load failed for vectorizer: {load_err}")
        # Add more debugging: check file size, try pickle directly?
        if os.path.exists(VECTORIZER_PATH):
             print(f"Vectorizer file size: {os.path.getsize(VECTORIZER_PATH)} bytes")
        raise RuntimeError(f"Failed to load vectorizer from {VECTORIZER_PATH}: {load_err}")


    # Download Model if missing
    if not model_exists:
        print(f"{MODEL_PATH} not found.")
        if not download_file(MODEL_URL, MODEL_PATH): raise RuntimeError(f"Failed to download model from {MODEL_URL}")
    print(f"Loading model from {MODEL_PATH}...")
    try:
        lgs = joblib.load(MODEL_PATH)
        print("Model loaded via joblib.")
    except Exception as load_err:
        print(f"joblib.load failed for model: {load_err}")
        if os.path.exists(MODEL_PATH):
             print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
             if os.path.exists(MODEL_PATH): 
                try: os.remove(MODEL_PATH); 
                except OSError: pass # Clean up corrupted?
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {load_err}")

    if vectorizer and lgs: print("Model and vectorizer ready.")
    else: raise RuntimeError("Model or vectorizer failed to load.")

except RuntimeError as e: print(f"FATAL ERROR during model setup: {e}"); vectorizer = None; lgs = None
except Exception as e: print(f"FATAL ERROR during model setup (general exception): {e}"); vectorizer = None; lgs = None


# --- 7. Define App Routes ---
# (Keep all your @app.route definitions for /, /analyze, /how-it-works, etc. below this)
# Make sure they correctly use the imported functions like getTokens, clean_url

@app.route('/')
def home():
    """ Serves the main AI Detector page. """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """ API endpoint to analyze a URL using the AI model. """
    if not lgs or not vectorizer:
        print("Error: /analyze called but model/vectorizer not loaded.")
        return jsonify({'error': 'AI model is not ready. Please check server start-up logs.'}), 503 # Service Unavailable
    try:
        data = request.get_json();
        if not data: return jsonify({'error': 'Invalid JSON payload.'}), 400
        url_raw = data.get('url')
        if not url_raw: return jsonify({'error': 'No URL provided.'}), 400

        url_clean = clean_url(url_raw) # Use imported function
        if not url_clean: return jsonify({'error': 'Invalid URL provided (failed cleaning).'}), 400

        url_entropy = entropy(url_clean) # Use imported function
        url_tokens = getTokens(url_clean) # Use imported function

        X_predict = [url_clean]
        try:
            X_predict_vec = vectorizer.transform(X_predict)
            y_Predict = lgs.predict(X_predict_vec)
        except Exception as pred_err:
             print(f"Error during model prediction/transform: {pred_err}")
             return jsonify({'error': 'Error applying AI model.'}), 500

        ai_prediction = str(y_Predict[0]) if y_Predict else 'error'
        is_malicious = (ai_prediction == 'bad')

        threat_report = []
        if is_malicious:
            # Use imported HIGH_RISK_TOKENS
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

# (Include all other routes: /how-it-works, /expander, /api/expand, /history, /report, /api/submit_report)
# Make sure they are defined correctly below

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/expander')
def expander_page():
    return render_template('expander.html')

@app.route('/api/expand', methods=['POST'])
def api_expand():
    # Keep the implementation using requests from previous correct version
    try:
        data = request.get_json()
        short_url = data.get('url')
        if not short_url: return jsonify({'error': 'No URL provided.'}), 400
        if not re.match(r'^(?:http|ftp)s?://', short_url): short_url = 'http://' + short_url
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        print(f"[Expander Debug] Attempting to expand URL: {short_url}")
        response = requests.head(short_url, allow_redirects=True, timeout=7, headers=headers)
        print(f"[Expander Debug] Status: {response.status_code}, Final URL: {response.url}, History: {len(response.history)}")
        if not response.ok and response.status_code != 405:
             print(f"[Expander Error] Request failed with status: {response.status_code}")
             return jsonify({'error': f'Request failed with status: {response.status_code}'}), response.status_code
        final_url = response.url
        cleaned_input = re.sub(r'^(?:http|ftp)s?://', '', short_url).strip('/')
        cleaned_final = re.sub(r'^(?:http|ftp)s?://', '', final_url).strip('/')
        if cleaned_input == cleaned_final and len(response.history) <= 1:
             print("[Expander Warning] Final URL same as input/protocol change only.")
             return jsonify({'error': 'Could not expand URL. May not be short link or blocked.'}), 400
        return jsonify({'final_url': final_url})
    except requests.exceptions.Timeout: print("[Expander Error] Timed out."); return jsonify({'error': 'Request timed out.'}), 504
    except requests.exceptions.ConnectionError as e: print(f"[Expander Error] Connection error: {e}"); return jsonify({'error': 'Could not connect.'}), 500
    except requests.exceptions.TooManyRedirects: print("[Expander Error] Too many redirects."); return jsonify({'error': 'Too many redirects.'}), 500
    except requests.exceptions.RequestException as e: print(f"[Expander Error] Request exception: {e}"); return jsonify({'error': f'Request error: {e}'}), 500
    except Exception as e: print(f"[Expander Error] Unknown exception: {e}"); return jsonify({'error': 'Unknown error expanding URL.'}), 500

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/api/submit_report', methods=['POST'])
def api_submit_report():
    if not db:
         print("[Firestore Error] 'db' object is None during submit_report.")
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
        return jsonify({'message': 'Thank you! Your feedback helps improve SafeLink AI.'}) # Updated message
    except Exception as e:
        print(f"Error processing report or saving to Firestore: {e}")
        return jsonify({'error': 'An error occurred while submitting your report.'}), 500

# --- 8. Run the App (for local testing, ignored by Render) ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Keep debug=False, especially when dealing with file I/O and external services
    app.run(host='0.0.0.0', port=port, debug=False)