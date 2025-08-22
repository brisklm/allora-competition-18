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
def optimize_model():
    from config import optuna
    if optuna is None:
        return jsonify({"error": "Optuna not available", "suggestion": "Install Optuna for tuning"})
    try:
        # Assuming a train.py with tune_model function using Optuna for LightGBM with added params
        from train import tune_model  # Assume exists
        results = tune_model()  # Runs Optuna to optimize for R2 > 0.1 with regularization
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/write_code', methods=['POST'])
def write_code():
    data = request.json
    title = data.get('title')
    content = data.get('content')
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return jsonify({"error": "Syntax error: " + str(e)})
    with open(title, 'w') as f:
        f.write(content)
    return jsonify({"success": True, "message": f"File {title} written"})

@app.route('/commit_to_github', methods=['POST'])
def commit_to_github():
    data = request.json
    message = data.get('message')
    files = data.get('files')
    try:
        for file in files:
            subprocess.run(['git', 'add', file], check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    features = np.array([data.get(feat, np.nan) for feat in selected_features]).reshape(1, -1)
    if NAN_HANDLING == 'fillna_mean':
        features = np.nan_to_num(features, nan=np.nanmean(features))
    features = scaler.transform(features)
    pred = model.predict(features)
    return jsonify({"prediction": pred[0]})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)