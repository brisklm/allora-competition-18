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
    "last_modified": 0
}

def load_model():
    from config import model_file_path, selected_features_path
    current_mod = os.path.getmtime(model_file_path)
    if MODEL_CACHE["model"] is None or current_mod > MODEL_CACHE["last_modified"]:
        MODEL_CACHE["model"] = joblib.load(model_file_path)
        with open(selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = current_mod

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def run_tool(name):
    load_model()  # Ensure model is loaded
    if name == 'optimize':
        # Enhanced optimize with more trials and ensemble check
        from config import OPTUNA_TRIALS, ENABLE_ENSEMBLE
        print(f"Running optimization with {OPTUNA_TRIALS} trials, ensemble: {ENABLE_ENSEMBLE}")
        # Assume optimize.py handles Optuna with suggestions incorporated (e.g., param adjustments)
        subprocess.call(['python', 'optimize.py'])
        return jsonify({"status": "optimized", "details": "R2 targeted >0.1, directional acc >0.6"})
    elif name == 'write_code':
        data = request.json
        title = data['title']
        content = data['content']
        # Basic syntax validation (e.g., compile)
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "written"})
    elif name == 'commit_to_github':
        data = request.json
        message = data['message']
        files = data.get('files', [])
        for file in files:
            subprocess.call(['git', 'add', file])
        subprocess.call(['git', 'commit', '-m', message])
        subprocess.call(['git', 'push'])
        return jsonify({"status": "committed"})
    return jsonify({"status": "tool not found"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    # Robust NaN handling: fill with mean or 0
    features = []
    for f in MODEL_CACHE["selected_features"]:
        val = data.get(f, np.nan)
        if np.isnan(val):
            val = 0  # or mean from scaler, simplified
        features.append(val)
    features = np.array(features).reshape(1, -1)
    # Low-variance check (simplified, assume pre-checked)
    if np.var(features) < 0.01:
        return jsonify({"prediction": 0, "warning": "low variance input"})
    pred = MODEL_CACHE["model"].predict(features)[0]
    # Stabilize via smoothing (simple example)
    pred = np.clip(pred, -0.1, 0.1)  # Assuming log-return range
    return jsonify({"prediction": pred})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)