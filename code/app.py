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

@app.route('/tool/<name>', methods=['POST'])
def execute_tool(name):
    data = request.get_json()
    if name == 'optimize':
        from config import optuna, OPTUNA_TRIALS, LGBM_PARAMS
        if optuna is None:
            return jsonify({"error": "Optuna not available"})
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
            r2 = np.random.random()  # Dummy; replace with actual training and evaluation
            return r2
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        model = "dummy model"  # Dummy; replace with actual model training
        joblib.dump(model, model_file_path)
        return jsonify({"best_params": best_params, "best_r2": study.best_value})
    elif name == 'write_code':
        title = data['title']
        content = data['content']
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": str(e)})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "Code written successfully"})
    elif name == 'commit_to_github':
        message = data['message']
        files = data['files']
        try:
            subprocess.check_call(['git', 'add'] + files)
            subprocess.check_call(['git', 'commit', '-m', message])
            subprocess.check_call(['git', 'push'])
            return jsonify({"status": "Committed and pushed successfully"})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Tool not found"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    input_data = np.array([data.get(feat, np.nan) for feat in selected_features]).reshape(1, -1)
    if NAN_HANDLING == 'ffill':
        input_data = np.nan_to_num(input_data, nan=0)
    scaler = joblib.load(scaler_file_path)
    input_scaled = scaler.transform(input_data)
    if np.var(input_scaled) < LOW_VARIANCE_THRESHOLD:
        return jsonify({"prediction": [0.0]})
    model = joblib.load(model_file_path)
    pred = model.predict(input_scaled)
    return jsonify({"prediction": pred.tolist()})

@app.route('/')
def health():
    return "OK"

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)