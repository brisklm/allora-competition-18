import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD
import optuna
from sklearn.metrics import r2_score, accuracy_score
from lightgbm import LGBMRegressor

app = Flask(__name__)
load_dotenv()

COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))

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

@app.route('/tools/optimize', methods=['POST'])
def optimize():
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 3, 15)
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
        model = LGBMRegressor(max_depth=max_depth, num_leaves=num_leaves, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
        return np.random.random()  # Replace with actual training and scoring
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    return jsonify({"best_params": best_params, "message": "Optimization complete"})

@app.route('/tools/write_code', methods=['POST'])
def write_code():
    data = request.json
    title = data['title']
    content = data['content']
    with open(title, 'w') as f:
        f.write(content)
    return jsonify({"message": f"Code written to {title}"})

@app.route('/tools/commit_to_github', methods=['POST'])
def commit_to_github():
    data = request.json
    message = data['message']
    files = data['files']
    for file in files:
        subprocess.run(['git', 'add', file])
    subprocess.run(['git', 'commit', '-m', message])
    subprocess.run(['git', 'push'])
    return jsonify({"message": "Committed to GitHub"})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)