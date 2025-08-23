import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD

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

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
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

# Add endpoint for tools
@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

# Add optimize endpoint
@app.route('/optimize', methods=['POST'])
def run_optimize():
    try:
        # Run a script for Optuna tuning, assume tune.py exists
        result = subprocess.run(['python', 'tune.py'], capture_output=True, text=True)
        return jsonify({"status": "success", "output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

# Prediction endpoint with NaN handling and low-variance check compatibility
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Load model and scaler
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    # Extract features
    features = np.array([data.get(f, np.nan) for f in selected_features]).reshape(1, -1)
    # Robust NaN handling
    if NAN_HANDLING == 'ffill':
        features = np.nan_to_num(features, nan=0.0)  # Simple replacement, can be extended
    # Low variance check (for logging, not affecting prediction)
    variances = np.var(features, axis=0)
    low_var = variances < LOW_VARIANCE_THRESHOLD
    if np.any(low_var):
        print(f"Warning: Low variance features detected")
    # Scale and predict
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    # Stabilize via simple smoothing (e.g., ensemble-like averaging if multiple models, here placeholder)
    return jsonify({"prediction": float(pred[0])})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)