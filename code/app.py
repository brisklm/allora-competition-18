import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess

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
    if MODEL_CACHE['model'] is None or os.path.getmtime('data/model.pkl') > MODEL_CACHE['last_modified']:
        MODEL_CACHE['model'] = joblib.load('data/model.pkl')
        with open('data/selected_features.json', 'r') as f:
            MODEL_CACHE['selected_features'] = json.load(f)
        MODEL_CACHE['last_modified'] = os.path.getmtime('data/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    features_list = [data.get(f, np.nan) for f in MODEL_CACHE['selected_features']]
    features = np.array(features_list).reshape(1, -1)
    # Robust NaN handling
    if np.any(np.isnan(features)):
        features = np.nan_to_num(features, nan=0.0)  # or use mean from scaler if available
    # Low-variance check (optional, log if variance low)
    if np.var(features) < 0.001:
        print("Warning: Low variance in input features")
    pred = MODEL_CACHE['model'].predict(features)[0]
    # Optional smoothing if enabled
    return jsonify({'prediction': pred})

def run_optimize():
    subprocess.run(['python', 'optimize.py'])  # Assume optimize.py handles Optuna, VADER, etc.
    return {'status': 'optimized'}

@app.route('/tools/optimize', methods=['POST'])
def optimize():
    result = run_optimize()
    return jsonify(result)

def write_code(params):
    title = params['title']
    content = params['content']
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return {'status': 'error', 'message': str(e)}
    with open(title, 'w') as f:
        f.write(content)
    return {'status': 'success'}

@app.route('/tools/write_code', methods=['POST'])
def write_code_route():
    params = request.json
    result = write_code(params)
    return jsonify(result)

def commit_to_github(params):
    message = params['message']
    files = params.get('files', [])
    for file in files:
        subprocess.run(['git', 'add', file])
    subprocess.run(['git', 'commit', '-m', message])
    subprocess.run(['git', 'push'])
    return {'status': 'committed'}

@app.route('/tools/commit_to_github', methods=['POST'])
def commit_github_route():
    params = request.json
    result = commit_to_github(params)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)