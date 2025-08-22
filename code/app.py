import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
import ast

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
    model_path = os.path.join('data', 'model.pkl')
    current_mod = os.path.getmtime(model_path)
    if MODEL_CACHE['model'] is None or MODEL_CACHE['last_modified'] != current_mod:
        MODEL_CACHE['model'] = joblib.load(model_path)
        with open(os.path.join('data', 'selected_features.json'), 'r') as f:
            MODEL_CACHE['selected_features'] = json.load(f)
        MODEL_CACHE['last_modified'] = current_mod

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def run_tool(name):
    if name == "optimize":
        import optimize as opt
        results = opt.run_optimization()  # Assume optimize.py exists with run_optimization using Optuna
        return jsonify(results)
    elif name == "write_code":
        args = request.json
        title = args['title']
        content = args['content']
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "success"})
    elif name == "commit_to_github":
        args = request.json
        message = args['message']
        files = args.get('files', [])
        for file in files:
            subprocess.run(["git", "add", file])
        subprocess.run(["git", "commit", "-m", message])
        subprocess.run(["git", "push"])
        return jsonify({"status": "success"})
    return jsonify({"error": "Tool not found"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    features = [data.get(f, 0) for f in MODEL_CACHE['selected_features']]  # Impute missing with 0
    input_data = np.array([features])
    if np.any(np.isnan(input_data)):
        input_data = np.nan_to_num(input_data)  # Robust NaN handling
    if np.var(input_data) < 1e-6:
        return jsonify({"prediction": 0, "warning": "Low variance input"})
    prediction = MODEL_CACHE['model'].predict(input_data)[0]
    return jsonify({"prediction": float(prediction)})

if __name__ == '__main__':
    app.run(port=FLASK_PORT, host='0.0.0.0')