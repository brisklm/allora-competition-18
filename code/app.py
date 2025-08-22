import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, SentimentIntensityAnalyzer, optuna

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))

# Load model and scaler
try:
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
except FileNotFoundError:
    model = None
    scaler = None
    selected_features = []

# VADER setup
if SentimentIntensityAnalyzer is not None:
    sia = SentimentIntensityAnalyzer()
else:
    sia = None

# TOOLS
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning with enhanced hyperparameters for better R2 and directional accuracy. Includes regularization and ensembling.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content after syntax validation.",
        "parameters": {
            "title": {"type": "string", "description": "Filename (e.g., model.py)", "required": True},
            "content": {"type": "string", "description": "Complete source code content", "required": True},
            "artifact_id": {"type": "string", "description": "Artifact UUID", "required": False},
            "artifact_version_id": {"type": "string", "description": "Version UUID", "required": False},
            "contentType": {"type": "string", "description": "Content type (e.g., text/python)", "required": False}
        }
    },
    {
        "name": "commit_to_github",
        "description": "Commits changes to GitHub repository.",
        "parameters": {
            "message": {"type": "string", "description": "Commit message", "required": True},
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}, "required": True}
        }
    }
]

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/optimize', methods=['POST'])
def optimize():
    if optuna is None:
        return jsonify({"error": "Optuna not available"}), 500
    try:
        result = subprocess.run(['python', 'train.py', '--optuna'], capture_output=True, text=True)
        return jsonify({"result": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    features = [data.get(f, np.nan) for f in selected_features]
    # Robust NaN handling
    if NAN_HANDLING == 'mean':
        features = [0 if np.isnan(f) else f for f in features]  # Simplified; use precomputed means in production
    features = np.array(features).reshape(1, -1)
    # Low variance check
    if np.var(features) < LOW_VARIANCE_THRESHOLD:
        return jsonify({"prediction": 0, "warning": "Low variance input"})
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return jsonify({"prediction": prediction})

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    if sia is None:
        return jsonify({"error": "VADER not available"}), 500
    data = request.json
    text = data.get('text', '')
    scores = sia.polarity_scores(text)
    return jsonify(scores)

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)