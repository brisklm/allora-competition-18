import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD

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

@app.route('/run_tool', methods=['POST'])
def run_tool():
    data = request.get_json()
    name = data.get('name')
    parameters = data.get('parameters', {})
    if name == 'optimize':
        # Trigger optimization with Optuna, ensuring NaN handling and low-variance checks
        try:
            result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
            if result.returncode != 0:
                return jsonify({'error': result.stderr}), 500
            return jsonify({'result': result.stdout})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif name == 'write_code':
        title = parameters.get('title')
        content = parameters.get('content')
        artifact_id = parameters.get('artifact_id')
        artifact_version_id = parameters.get('artifact_version_id')
        contentType = parameters.get('contentType', 'text/python')
        if not title or not content:
            return jsonify({'error': 'Missing title or content'}), 400
        # Validate syntax for python
        if contentType == 'text/python':
            try:
                compile(content, title, 'exec')
            except SyntaxError as e:
                return jsonify({'error': f'Syntax error: {str(e)}'}), 400
        # Write to file
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({'success': True, 'artifact_id': artifact_id or 'new', 'artifact_version_id': artifact_version_id or 'v1'})
    elif name == 'commit_to_github':
        message = parameters.get('message')
        files = parameters.get('files')
        if not message or not files:
            return jsonify({'error': 'Missing message or files'}), 400
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({'success': True})
        except subprocess.CalledProcessError as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Unknown tool'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)