import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "65")
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
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}}
        }
    }
]

# In-memory cache for inference
MODEL = None
SCALER = None
SELECTED_FEATURES = None
def load_model():
    global MODEL, SCALER, SELECTED_FEATURES
    try:
        MODEL = joblib.load(model_file_path)
        with open(selected_features_path, 'r') as f:
            SELECTED_FEATURES = json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
load_model()

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/call_tool', methods=['POST'])
def call_tool():
    data = request.json
    tool_name = data.get('name')
    params = data.get('parameters', {})
    if tool_name == 'optimize':
        return optimize_model(params)
    elif tool_name == 'write_code':
        return write_code(params)
    elif tool_name == 'commit_to_github':
        return commit_to_github(params)
    else:
        return jsonify({"error": "Unknown tool"}), 400
def optimize_model(params):
    try:
        from config import optuna, OPTUNA_TRIALS
        # Placeholder for Optuna optimization logic
        study = optuna.create_study(direction='minimize')
        # Assume objective function defined elsewhere
        study.optimize(lambda trial: 0.1, n_trials=OPTUNA_TRIALS)  # Dummy
        return jsonify({"result": study.best_params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def write_code(params):
    title = params.get('title')
    content = params.get('content')
    contentType = params.get('contentType', 'text/python')
    if contentType == 'text/python':
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {e}"}), 400
    with open(title, 'w') as f:
        f.write(content)
    artifact_id = params.get('artifact_id', 'generated')
    artifact_version_id = params.get('artifact_version_id', 'v1')
    return jsonify({"success": True, "artifact_id": artifact_id, "artifact_version_id": artifact_version_id})
def commit_to_github(params):
    message = params.get('message')
    files = params.get('files', [])
    try:
        subprocess.run(['git', 'add'] + files, check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data.get(f, np.nan) for f in SELECTED_FEATURES]).reshape(1, -1)
    if NAN_HANDLING == 'ffill':
        features = np.nan_to_num(features, nan=0.0)  # Simple handling
    try:
        pred = MODEL.predict(features)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)