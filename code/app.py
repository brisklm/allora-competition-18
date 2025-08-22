import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, MODEL_PARAMS, OPTUNA_TRIALS

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

# Load model
try:
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
except:
    model = None
    scaler = None

def handle_nan(data):
    if NAN_HANDLING == 'fill_median':
        return data.fillna(data.median())
    return data

def remove_low_variance(features):
    variances = np.var(features, axis=0)
    return features[:, variances > LOW_VARIANCE_THRESHOLD]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assume data processing, NaN handling, low variance removal, scaling, prediction
    # For stabilization, simple smoothing example (mock previous pred)
    pred = 0.0  # model.predict(...)
    smoothed_pred = (pred + 0.0) / 2  # mock ensembling/averaging
    return jsonify({'prediction': smoothed_pred})

def run_optimize():
    if optuna is None:
        return {"error": "Optuna not available"}
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        }
        # Mock training and score (aim R2 >0.1, directional acc >0.6, correlation >0.25)
        score = 0.15
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return study.best_params

@app.route('/optimize', methods=['POST'])
def optimize():
    result = run_optimize()
    return jsonify(result)

@app.route('/write_code', methods=['POST'])
def write_code():
    data = request.json
    title = data['title']
    content = data['content']
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return jsonify({"error": str(e)})
    with open(title, 'w') as f:
        f.write(content)
    return jsonify({"success": True})

@app.route('/commit_to_github', methods=['POST'])
def commit_to_github():
    data = request.json
    message = data['message']
    files = data['files']
    for file in files:
        subprocess.run(['git', 'add', file])
    subprocess.run(['git', 'commit', '-m', message])
    subprocess.run(['git', 'push'])
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)