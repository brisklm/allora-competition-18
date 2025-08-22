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

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/optimize', methods=['POST'])
def optimize():
    from config import optuna
    if optuna is None:
        return jsonify({"error": "Optuna not available"}), 500
    try:
        subprocess.run(['python', 'train.py'], check=True)
        return jsonify({"status": "optimization complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    features = np.array([data.get(feat, np.nan) for feat in selected_features])
    # Robust NaN handling
    if NAN_HANDLING == 'ffill':
        for i in range(1, len(features)):
            if np.isnan(features[i]):
                features[i] = features[i-1]
    elif NAN_HANDLING == 'mean':
        mean_val = np.nanmean(features)
        features = np.nan_to_num(features, nan=mean_val)
    else:
        features = np.nan_to_num(features, nan=0)
    # Low variance check (optional, for logging)
    var = np.var(features)
    if var < LOW_VARIANCE_THRESHOLD:
        return jsonify({"warning": "Low variance features", "prediction": 0.0})
    scaler = joblib.load(scaler_file_path)
    features_scaled = scaler.transform([features])
    model = joblib.load(model_file_path)
    prediction = model.predict(features_scaled)[0]
    # Stabilize with simple smoothing (e.g., ensemble simulation)
    predictions = [prediction]  # could load multiple models
    stable_pred = np.mean(predictions)
    return jsonify({"prediction": stable_pred, "direction": 1 if stable_pred > 0 else -1})

@app.route('/tool/write_code', methods=['POST'])
def write_code():
    params = request.json
    title = params['title']
    content = params['content']
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return jsonify({"error": str(e)}), 400
    with open(title, 'w') as f:
        f.write(content)
    return jsonify({"status": "code written", "file": title})

@app.route('/tool/commit_to_github', methods=['POST'])
def commit_to_github():
    params = request.json
    message = params['message']
    files = params['files']
    try:
        subprocess.run(['git', 'add'] + files, check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return jsonify({"status": "committed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)