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

@app.route('/optimize', methods=['POST'])
def trigger_optimize():
    try:
        result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
        return jsonify({"status": "success", "output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

model = None
scaler = None
selected_features = None
def load_model():
    global model, scaler, selected_features
    if os.path.exists(model_file_path):
        model = joblib.load(model_file_path)
    if os.path.exists(scaler_file_path):
        scaler = joblib.load(scaler_file_path)
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    for key in data:
        if data[key] is None or np.isnan(data[key]):
            if NAN_HANDLING == 'mean':
                data[key] = 0  # Simplified; assume mean=0 or compute from data
    features = [data.get(f, 0) for f in selected_features or []]
    features = np.array(features).reshape(1, -1)
    if scaler:
        features = scaler.transform(features)
    if model:
        prediction = model.predict(features)[0]
        return jsonify({"prediction": prediction})
    else:
        return jsonify({"error": "Model not loaded"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)