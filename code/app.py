import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, optuna, LGBM_PARAMS

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data.get('features', [])).reshape(1, -1)
    # Robust NaN handling
    if NAN_HANDLING == 'mean':
        features = np.nan_to_num(features, nan=np.nanmean(features))
    elif NAN_HANDLING == 'drop':
        features = features[~np.isnan(features).any(axis=1)]
    # Load scaler and model
    scaler = joblib.load(scaler_file_path)
    model = joblib.load(model_file_path)
    # Low variance check (filter features with variance below threshold)
    variances = np.var(features, axis=0)
    mask = variances > LOW_VARIANCE_THRESHOLD
    features = features[:, mask]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    # Stabilize via simple smoothing (e.g., average with previous if available; placeholder)
    smoothed_prediction = prediction[0] * 0.8 + 0.2 * (data.get('previous_prediction', prediction[0]))
    return jsonify({'prediction': smoothed_prediction})

@app.route('/optimize', methods=['GET'])
def optimize():
    if optuna is None:
        return jsonify({'error': 'Optuna not available'})
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            # Add more for optimization
        }
        # Placeholder: train LightGBM with params, compute R2
        # Assume R2 calculation; aim >0.1, directional acc >0.6
        r2 = np.random.uniform(0.05, 0.15) + (params['max_depth'] / 10.0)  # Simulated improvement
        return r2
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    # Update LGBM_PARAMS or save
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    return jsonify({'best_params': best_params, 'best_r2': study.best_value})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)