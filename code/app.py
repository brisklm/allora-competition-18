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

# Load model and scaler
try:
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
except Exception as e:
    model = None
    scaler = None
    selected_features = []

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    features = np.array([data.get(f, np.nan) for f in selected_features])
    # Robust NaN handling
    if NAN_HANDLING == 'fill_median':
        nan_mask = np.isnan(features)
        if np.any(nan_mask):
            medians = np.nanmedian(features)
            features[nan_mask] = medians
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    # Prediction with stabilization (simple ensemble-like averaging if multiple models, but assume single)
    prediction = model.predict(features_scaled)[0]
    # Optional smoothing (e.g., exponential moving average simulation, but for single pred, skip or assume)
    return jsonify({"prediction": prediction})

def run_optimize():
    # Trigger Optuna tuning (assume train.py exists with Optuna integration for hyperparams)
    result = subprocess.run(['python', 'train.py', '--optuna'], capture_output=True)
    return result.stdout.decode()

@app.route('/tool', methods=['POST'])
def tool():
    data = request.json
    name = data['name']
    params = data.get('parameters', {})
    if name == 'optimize':
        return jsonify({"results": run_optimize()})
    elif name == 'write_code':
        title = params['title']
        content = params['content']
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"success": True, "artifact_id": params.get('artifact_id', 'default')})
    elif name == 'commit_to_github':
        message = params['message']
        files = params['files']
        for file in files:
            subprocess.run(['git', 'add', file])
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Unknown tool"}), 400

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)