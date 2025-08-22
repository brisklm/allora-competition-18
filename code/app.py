import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING

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
    mod_time = os.path.getmtime(model_file_path)
    if MODEL_CACHE["model"] is None or MODEL_CACHE["last_modified"] != mod_time:
        MODEL_CACHE["model"] = joblib.load(model_file_path)
        with open(selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = mod_time
    return MODEL_CACHE["model"], MODEL_CACHE["selected_features"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model, features = load_model()
    input_data = np.array([data.get(f, np.nan) for f in features]).reshape(1, -1)
    # Robust NaN handling
    if np.any(np.isnan(input_data)):
        if NAN_HANDLING == 'impute_mean':
            input_data = np.nan_to_num(input_data, nan=np.nanmean(input_data))
        else:
            return jsonify({"error": "Input contains NaN values"}), 400
    # Low-variance check (simple example, can be expanded)
    if np.var(input_data) < 0.01:
        return jsonify({"warning": "Low variance input, predictions may be unstable"})
    prediction = model.predict(input_data)[0]
    # Stabilize via simple smoothing (example)
    smoothed_prediction = prediction * 0.9  # Placeholder for smoothing or ensembling
    return jsonify({"prediction": smoothed_prediction})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def run_tool(name):
    if name == 'optimize':
        # Trigger Optuna optimization (assume optimize.py exists)
        result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
        return jsonify({"result": result.stdout})
    elif name == 'write_code':
        params = request.json
        title = params['title']
        content = params['content']
        # Simple syntax validation (e.g., for Python)
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "File written"})
    elif name == 'commit_to_github':
        params = request.json
        message = params['message']
        files = params.get('files', [])
        subprocess.run(['git', 'add'] + files)
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({"status": "Committed"})
    return jsonify({"error": "Tool not found"}), 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT)