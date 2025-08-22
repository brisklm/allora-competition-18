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
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}}
        }
    }
]

# In-memory cache for inference
MODEL_CACHE = {
    "model": None,
    "selected_features": [],
    "last_model_mtime": None,
    "last_update": None
}

def load_model():
    from config import model_file_path, selected_features_path
    if not os.path.exists(model_file_path):
        return None
    current_mtime = os.path.getmtime(model_file_path)
    if MODEL_CACHE['model'] is None or current_mtime > MODEL_CACHE['last_model_mtime']:
        MODEL_CACHE['model'] = joblib.load(model_file_path)
        with open(selected_features_path, 'r') as f:
            MODEL_CACHE['selected_features'] = json.load(f)
        MODEL_CACHE['last_model_mtime'] = current_mtime
        MODEL_CACHE['last_update'] = datetime.utcnow()
    return MODEL_CACHE['model']

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke/<tool_name>', methods=['POST'])
def invoke_tool(tool_name):
    if tool_name == "optimize":
        # Assuming there's a train.py or similar that runs Optuna
        try:
            subprocess.run(["python", "train.py"])  # Run optimization script
            return jsonify({"status": "optimized", "message": "Model optimized using Optuna"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif tool_name == "write_code":
        params = request.json
        title = params.get('title')
        content = params.get('content')
        if not title or not content:
            return jsonify({"error": "Missing title or content"}), 400
        try:
            compile(content, title, 'exec')  # Syntax validation
            with open(title, 'w') as f:
                f.write(content)
            return jsonify({"status": "written", "file": title})
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
    elif tool_name == "commit_to_github":
        params = request.json
        message = params.get('message')
        files = params.get('files', [])
        if not message:
            return jsonify({"error": "Missing commit message"}), 400
        try:
            for file in files:
                subprocess.run(["git", "add", file], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
            return jsonify({"status": "committed", "message": message})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unknown tool"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = load_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    # Assume data has features in order of selected_features
    input_data = np.array([data.get(f, 0) for f in MODEL_CACHE['selected_features']]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, host='0.0.0.0', debug=True)