import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, LGBM_PARAMS, ENSEMBLE_SIZE, CORRELATION_THRESHOLD, FEATURES

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

# Load model and scaler
model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)
with open(selected_features_path, 'r') as f:
    selected_features = json.load(f)

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results. Tuned for R2 > 0.1, directional accuracy > 0.6, with regularization.",
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
    # Assume data is dict with features
    features = np.array([data[f] for f in selected_features]).reshape(1, -1)
    # NaN handling
    if NAN_HANDLING == 'mean':
        features = np.nan_to_num(features, nan=np.nanmean(features))
    # Scale
    features_scaled = scaler.transform(features)
    # Ensemble prediction (simple bagging simulation)
    predictions = []
    for _ in range(ENSEMBLE_SIZE):
        pred = model.predict(features_scaled)
        predictions.append(pred)
    smoothed_pred = np.mean(predictions)  # Averaging for stabilization
    return jsonify({'prediction': smoothed_pred})

@app.route('/optimize', methods=['POST'])
def optimize():
    # Trigger Optuna tuning (assume tune.py exists with custom objective for R2, dir acc, correlation >0.25)
    result = subprocess.run(['python', 'tune.py'], capture_output=True, text=True)
    return jsonify({'result': result.stdout})

@app.route('/write_code', methods=['POST'])
def write_code():
    params = request.json
    title = params['title']
    content = params['content']
    # Simple syntax check
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return jsonify({'error': str(e)}), 400
    with open(title, 'w') as f:
        f.write(content)
    return jsonify({'status': 'written'})

@app.route('/commit_to_github', methods=['POST'])
def commit_to_github():
    params = request.json
    message = params['message']
    files = params['files']
    subprocess.run(['git', 'add'] + files)
    subprocess.run(['git', 'commit', '-m', message])
    subprocess.run(['git', 'push'])
    return jsonify({'status': 'committed'})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)