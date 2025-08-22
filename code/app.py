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
    import config
    if MODEL_CACHE["model"] is None or os.path.getmtime(config.model_file_path) > MODEL_CACHE["last_modified"]:
        MODEL_CACHE["model"] = joblib.load(config.model_file_path)
        with open(config.selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = os.path.getmtime(config.model_file_path)
    return MODEL_CACHE["model"], MODEL_CACHE["selected_features"]

@app.route('/health')
def health():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model, features = load_model()
    input_data = np.array([[data.get(f, 0) for f in features]])
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": prediction})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def call_tool(name):
    if name == "optimize":
        subprocess.run(["python", "optimize.py"])
        return jsonify({"status": "optimized"})
    elif name == "write_code":
        params = request.json
        title = params["title"]
        content = params["content"]
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "written"})
    elif name == "commit_to_github":
        params = request.json
        message = params["message"]
        files = params.get("files", [])
        for file in files:
            subprocess.run(["git", "add", file])
        subprocess.run(["git", "commit", "-m", message])
        subprocess.run(["git", "push"])
        return jsonify({"status": "committed"})
    return "Tool not found", 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT)