import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
import ast
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
model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)
with open(selected_features_path, 'r') as f:
    selected_features = json.load(f)

@app.route('/')
def home():
    return f"MCP Version: {MCP_VERSION}"

@app.route('/health')
def health():
    return "OK"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data.get(f, np.nan) for f in selected_features]
    # Robust NaN handling
    if NAN_HANDLING == 'median':
        fill_value = np.nanmedian(features)
    else:
        fill_value = 0
    features = np.nan_to_num(features, nan=fill_value)
    # Low variance check (skip if variance below threshold, but for prediction, assume features are selected)
    if np.var(features) < LOW_VARIANCE_THRESHOLD:
        return jsonify({'prediction': 0, 'warning': 'Low variance input'})
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    # Stabilize via simple smoothing (e.g., ensemble-like averaging if multiple models, but here placeholder)
    return jsonify({'prediction': prediction})

@app.route('/tools')
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke/<tool_name>', methods=['POST'])
def invoke(tool_name):
    if tool_name == 'optimize':
        # Trigger Optuna-based optimization for LightGBM with regularization
        result = subprocess.run(['python', 'optimize.py'], capture_output=True)  # Assume optimize.py handles Optuna tuning with suggestions
        return jsonify({'result': result.stdout.decode()})
    elif tool_name == 'write_code':
        params = request.json
        title = params['title']
        content = params['content']
        try:
            ast.parse(content)
            with open(title, 'w') as f:
                f.write(content)
            return jsonify({'status': 'success'})
        except SyntaxError as e:
            return jsonify({'status': 'error', 'message': str(e)})
    elif tool_name == 'commit_to_github':
        params = request.json
        message = params['message']
        files = params['files']
        for file in files:
            subprocess.run(['git', 'add', file])
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({'status': 'committed'})
    else:
        return jsonify({'error': 'Tool not found'}), 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT, host='0.0.0.0')