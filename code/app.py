import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, OPTUNA_TRIALS, LGBM_PARAMS, FEATURES

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

@app.route('/')
def home():
    return "MCP App running"

@app.route('/tools')
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke/<tool_name>', methods=['POST'])
def invoke(tool_name):
    if tool_name == 'optimize':
        try:
            import optuna
            from lightgbm import LGBMRegressor
            # Dummy data for illustration; replace with actual data loading
            X = np.random.rand(100, len(FEATURES))
            y = np.random.rand(100)
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                    'n_estimators': 200,
                    'learning_rate': 0.01
                }
                model = LGBMRegressor(**params)
                model.fit(X, y)
                preds = model.predict(X)
                r2 = np.corrcoef(y, preds)[0,1]**2  # Aim for R2 > 0.1
                return -r2  # Minimize negative R2
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            return jsonify({"best_params": study.best_params, "best_r2": -study.best_value})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif tool_name == 'write_code':
        data = request.json
        title = data.get('title')
        content = data.get('content')
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": "Syntax error: " + str(e)})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"success": True, "file": title})
    elif tool_name == 'commit_to_github':
        data = request.json
        message = data.get('message')
        files = data.get('files')
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Unknown tool"})

def load_model():
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    return model, scaler, selected_features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model, scaler, selected_features = load_model()
    # Robust NaN handling
    for key in selected_features:
        if key not in data or np.isnan(data[key]):
            if NAN_HANDLING == 'drop':
                return jsonify({"error": "Missing feature"})
            elif NAN_HANDLING == 'mean':
                data[key] = 0.0  # Simplified; use precomputed mean in production
            elif NAN_HANDLING == 'median':
                data[key] = 0.0
    features = np.array([data[f] for f in selected_features]).reshape(1, -1)
    # Low variance check skipped for prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    # Simple smoothing for stability (e.g., average with previous, but here dummy)
    smoothed_prediction = prediction * 0.9  # Placeholder for ensembling/smoothing
    return jsonify({"prediction": smoothed_prediction})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)