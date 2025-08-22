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

# Functions for tools
def run_optimize():
    try:
        result = subprocess.run(['python', 'tune.py'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def write_code(params):
    title = params['title']
    content = params['content']
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return str(e)
    with open(title, 'w') as f:
        f.write(content)
    return f"Written to {title}"

def commit_to_github(params):
    message = params['message']
    files = params['files']
    try:
        subprocess.run(['git', 'add'] + files, check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return "Committed and pushed"
    except Exception as e:
        return str(e)

# Endpoint for tools
@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def run_tool(name):
    params = request.json
    if name == 'optimize':
        result = run_optimize()
    elif name == 'write_code':
        result = write_code(params)
    elif name == 'commit_to_github':
        result = commit_to_github(params)
    else:
        return jsonify({"error": "Tool not found"}), 404
    return jsonify({"result": result})

# Load model and components
model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)
with open(selected_features_path, 'r') as f:
    selected_features = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data.get(f, 0) for f in selected_features]
    # Robust NaN handling
    if NAN_HANDLING == 'mean':
        features = [x if not np.isnan(x) else 0 for x in features]  # Use 0 as placeholder; improve with precomputed means if needed
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    # Stabilize via simple clipping/smoothing
    prediction = np.clip(prediction, -0.1, 0.1)
    return jsonify({"prediction": prediction})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)