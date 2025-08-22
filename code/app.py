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

import config

def load_model():
    model_path = config.model_file_path
    if not os.path.exists(model_path):
        return None
    mod_time = os.path.getmtime(model_path)
    if MODEL_CACHE["model"] is None or MODEL_CACHE["last_modified"] != mod_time:
        MODEL_CACHE["model"] = joblib.load(model_path)
        with open(config.selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = mod_time
    return MODEL_CACHE["model"], MODEL_CACHE["selected_features"]

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke/<tool_name>', methods=['POST'])
def invoke_tool(tool_name):
    data = request.json
    if tool_name == "optimize":
        try:
            result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
            return jsonify({"status": "success", "output": result.stdout})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    elif tool_name == "write_code":
        title = data.get("title")
        content = data.get("content")
        contentType = data.get("contentType", "text/python")
        if contentType == "text/python":
            try:
                compile(content, title, 'exec')
            except SyntaxError as e:
                return jsonify({"status": "error", "message": f"Syntax error: {str(e)}"})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "success", "message": f"File {title} written"})
    elif tool_name == "commit_to_github":
        message = data.get("message")
        files = data.get("files", [])
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Unknown tool"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model, features = load_model()
    if model is None:
        return jsonify({"error": "Model not loaded"})
    input_data = np.array([[data.get(f, 0) for f in features]])
    if np.any(np.isnan(input_data)):
        input_data = np.nan_to_num(input_data)  # Robust NaN handling
    if np.var(input_data) < 1e-6:  # Low-variance check
        return jsonify({"error": "Input has low variance, unstable prediction"})
    prediction = model.predict(input_data)[0]
    # Stabilize via simple smoothing (e.g., average with 0 for demo; in practice, use ensemble or history)
    smoothed_prediction = (prediction + 0) / 2  # Placeholder for smoothing
    return jsonify({"prediction": smoothed_prediction})

@app.route('/health')
def health():
    return "OK"

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)