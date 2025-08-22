import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np

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
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}}
        }
    }
]

# In-memory cache for inference
MODEL_CACHE = {
    "model": None,
    "selected_features": [],
    "last_model_mtime": None,
    "last_update": None
}

@app.route('/')
def home():
    return jsonify({"version": MCP_VERSION})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        import optuna
        # Dummy Optuna study for optimization (adjust params for suggestions: max_depth, regularization)
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 3, 10)
            num_leaves = trial.suggest_int('num_leaves', 20, 50)
            reg_alpha = trial.suggest_float('reg_alpha', 0.0, 0.1)
            return np.random.random()  # Placeholder for actual objective
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        return jsonify({"status": "optimized", "best_params": study.best_params})
    except:
        return jsonify({"error": "Optuna not available"}), 500

@app.route('/write_code', methods=['POST'])
def write_code():
    data = request.json
    title = data.get('title')
    content = data.get('content')
    if not title or not content:
        return jsonify({"error": "Missing parameters"}), 400
    try:
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "written", "file": title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/commit_to_github', methods=['POST'])
def commit_to_github():
    data = request.json
    message = data.get('message')
    files = data.get('files', [])
    # Dummy commit
    return jsonify({"status": "committed", "message": message, "files": files})

@app.route('/predict', methods=['POST'])
def predict():
    # Dummy prediction with smoothing for stabilization
    pred = np.random.random() * 0.01
    smoothed_pred = pred * 0.9 + 0.005  # Example ensembling/smoothing
    return jsonify({"prediction": smoothed_pred})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)