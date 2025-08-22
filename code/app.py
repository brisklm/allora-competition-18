import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD

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

@app.route('/mcp/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/mcp/tool/optimize', methods=['POST'])
def run_optimize():
    # Assuming a script or function for Optuna optimization
    try:
        result = subprocess.run(['python', 'train.py', '--optimize'], capture_output=True, text=True)  # Example
        return jsonify({'result': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add prediction endpoint for compatibility
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Load model and scaler
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    # Assume input features, handle NaN, low variance (simplified)
    features = np.array(data['features'])
    if NAN_HANDLING == 'mean':
        features = np.nan_to_num(features, nan=np.nanmean(features))
    # Low variance check (example, not full impl)
    var = np.var(features, axis=0)
    mask = var > LOW_VARIANCE_THRESHOLD
    features = features[:, mask]
    scaled = scaler.transform([features])
    pred = model.predict(scaled)
    # Stabilize with simple smoothing (example: average with prev if available)
    if 'prev_pred' in data:
        pred = (pred + data['prev_pred']) / 2
    return jsonify({'prediction': pred.tolist()})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)