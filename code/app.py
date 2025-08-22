import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, FEATURES, LGBM_PARAMS, optuna, OPTUNA_TRIALS, SentimentIntensityAnalyzer

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

def compute_sentiment(text):
    if SentimentIntensityAnalyzer is None:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def run_optimize():
    if optuna is None:
        return {"error": "Optuna not available"}
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        # Placeholder for training and scoring (replace with actual data loading and evaluation for R2 > 0.1, directional acc > 0.6)
        score = np.random.random()  # Aim for R2
        dir_acc = 0.5 + np.random.random() * 0.2  # Aim for >0.6
        return -score
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return {"best_params": study.best_params, "best_score": -study.best_value}

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<tool_name>', methods=['POST'])
def call_tool(tool_name):
    if tool_name == 'optimize':
        result = run_optimize()
        return jsonify(result)
    elif tool_name == 'write_code':
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
    elif tool_name == 'commit_to_github':
        data = request.json
        message = data['message']
        files = data['files']
        for file in files:
            subprocess.run(['git', 'add', file])
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Tool not found"}), 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT)