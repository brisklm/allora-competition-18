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
    model_path = 'data/model.pkl'
    features_path = 'data/selected_features.json'
    current_mod = os.path.getmtime(model_path)
    if MODEL_CACHE['model'] is None or MODEL_CACHE['last_modified'] != current_mod:
        MODEL_CACHE['model'] = joblib.load(model_path)
        with open(features_path, 'r') as f:
            MODEL_CACHE['selected_features'] = json.load(f)
        MODEL_CACHE['last_modified'] = current_mod
    return MODEL_CACHE['model'], MODEL_CACHE['selected_features']

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/version', methods=['GET'])
def version():
    return jsonify({"version": MCP_VERSION})

@app.route('/tools', methods=['GET'])
def tools():
    return jsonify(TOOLS)

@app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    tool_name = data.get('tool')
    params = data.get('parameters', {})
    if tool_name == 'optimize':
        try:
            result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
            return jsonify({"result": result.stdout, "error": result.stderr})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif tool_name == 'write_code':
        title = params.get('title')
        content = params.get('content')
        if not title or not content:
            return jsonify({"error": "Missing parameters"}), 400
        try:
            with open(title, 'w') as f:
                f.write(content)
            compile(content, title, 'exec')
            return jsonify({"success": True})
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif tool_name == 'commit_to_github':
        message = params.get('message')
        files = params.get('files', [])
        if not message:
            return jsonify({"error": "Missing message"}), 400
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"success": True})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unknown tool"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    features = np.array(data['features'])
    model, selected = MODEL_CACHE['model'], MODEL_CACHE['selected_features']
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)