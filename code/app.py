import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, OPTUNA_TRIALS, LGBM_PARAMS, FEATURES

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
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}, "required": True}
        }
    }
]

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/call_tool', methods=['POST'])
def call_tool():
    data = request.json
    tool_name = data.get('tool')
    parameters = data.get('parameters', {})
    if tool_name == 'optimize':
        try:
            # Assume train.py handles Optuna optimization with improved params
            result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
            return jsonify({"status": "success", "output": result.stdout})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    elif tool_name == 'write_code':
        title = parameters.get('title')
        content = parameters.get('content')
        if not title or not content:
            return jsonify({"status": "error", "message": "Missing title or content"})
        try:
            compile(content, title, 'exec')
            with open(title, 'w') as f:
                f.write(content)
            return jsonify({"status": "success"})
        except SyntaxError as e:
            return jsonify({"status": "error", "message": str(e)})
    elif tool_name == 'commit_to_github':
        message = parameters.get('message')
        files = parameters.get('files', [])
        if not message or not files:
            return jsonify({"status": "error", "message": "Missing message or files"})
        try:
            for file in files:
                subprocess.run(['git', 'add', file], check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Unknown tool"})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)