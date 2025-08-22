import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
import config

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "65")
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
    "last_modified": None
}

def load_model():
    if MODEL_CACHE["model"] is None or (datetime.now() - MODEL_CACHE["last_modified"]).days > 0:
        MODEL_CACHE["model"] = joblib.load(config.model_file_path)
        with open(config.selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = datetime.now()

def preprocess_input(data):
    data = np.array(data)
    if np.var(data) < config.LOW_VARIANCE_THRESHOLD:
        raise ValueError("Input data has low variance")
    if np.any(np.isnan(data)):
        mask = np.isnan(data)
        idx = np.where(~mask)[0]
        data[mask] = np.interp(np.flatnonzero(mask), idx, data[~mask])
    return data

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    input_data = request.json.get('data', [])
    processed_data = preprocess_input(input_data)
    prediction = MODEL_CACHE["model"].predict([processed_data])[0]
    # Simple smoothing example (assuming predictions are in cache or something; for single, pass through)
    return jsonify({'prediction': prediction})

@app.route('/optimize', methods=['GET'])
def optimize():
    # Trigger Optuna optimization (assume tune_model.py exists and uses config.OPTUNA_TRIALS)
    subprocess.run(['python', 'tune_model.py'])
    load_model()
    return jsonify({'status': 'Model optimized', 'version': MCP_VERSION})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)