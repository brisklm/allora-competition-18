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
    global MODEL_CACHE
    from config import model_file_path, selected_features_path
    current_mod_time = os.path.getmtime(model_file_path)
    if MODEL_CACHE["model"] is None or MODEL_CACHE["last_modified"] != current_mod_time:
        MODEL_CACHE["model"] = joblib.load(model_file_path)
        with open(selected_features_path, 'r') as f:
            MODEL_CACHE["selected_features"] = json.load(f)
        MODEL_CACHE["last_modified"] = current_mod_time
    return MODEL_CACHE["model"], MODEL_CACHE["selected_features"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model, features = load_model()
    input_list = [data.get(f, 0) for f in features]
    input_data = np.array(input_list).reshape(1, -1)
    if np.isnan(input_data).any():
        input_data = np.nan_to_num(input_data, nan=0)
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": prediction})

@app.route('/tool/<name>', methods=['POST'])
def call_tool(name):
    data = request.json
    if name == "optimize":
        try:
            result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
            return jsonify({"status": "success", "output": result.stdout, "error": result.stderr})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    elif name == "write_code":
        title = data['title']
        content = data['content']
        if title.endswith('.py'):
            try:
                compile(content, title, 'exec')
            except SyntaxError as e:
                return jsonify({"status": "error", "message": str(e)})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "success"})
    elif name == "commit_to_github":
        message = data['message']
        files = data.get('files', [])
        try:
            if files:
                subprocess.run(['git', 'add'] + files, check=True)
            else:
                subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Tool not found"}), 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)