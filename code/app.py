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
MODEL = None
SCALER = None
SELECTED_FEATURES = []
def load_model():
    global MODEL, SCALER, SELECTED_FEATURES
    try:
        MODEL = joblib.load(model_file_path)
        SCALER = joblib.load(scaler_file_path)
        with open(selected_features_path, 'r') as f:
            SELECTED_FEATURES = json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    for key in data:
        if np.isnan(data[key]):
            if NAN_HANDLING == 'ffill':
                data[key] = 0
    features_array = np.array([data.get(f, 0) for f in SELECTED_FEATURES])
    if np.var(features_array) < LOW_VARIANCE_THRESHOLD:
        return jsonify({'prediction': 0.0})
    scaled = SCALER.transform(features_array.reshape(1, -1))
    pred = MODEL.predict(scaled)[0]
    return jsonify({'prediction': pred})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/call_tool', methods=['POST'])
def call_tool():
    req = request.json
    tool_name = req['name']
    params = req.get('parameters', {})
    if tool_name == 'optimize':
        try:
            result = subprocess.run(['python', 'optimize.py'], capture_output=True, text=True)
            return jsonify({'result': result.stdout})
        except Exception as e:
            return jsonify({'error': str(e)})
    elif tool_name == 'write_code':
        title = params['title']
        content = params['content']
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({'error': f"Syntax error: {e}"})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({'success': True})
    elif tool_name == 'commit_to_github':
        message = params['message']
        files = params.get('files', [])
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Unknown tool'})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)