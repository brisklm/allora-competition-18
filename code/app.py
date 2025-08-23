import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD
app = Flask(__name__)
load_dotenv()
COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results. Tuned for R2 > 0.1, directional accuracy > 0.6, correlation > 0.25.",
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
try:
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    selected_features = []
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    for key in data:
        if np.isnan(data[key]):
            if NAN_HANDLING == 'mean':
                data[key] = 0
    input_data = [data.get(f, 0) for f in selected_features]
    input_scaled = scaler.transform(np.array([input_data]))
    prediction = model.predict(input_scaled)[0]
    return jsonify({"prediction": prediction})
@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)
@app.route('/tool/<tool_name>', methods=['POST'])
def execute_tool(tool_name):
    from config import optuna
    if tool_name == 'optimize':
        if optuna is None:
            return jsonify({"error": "Optuna not installed"}), 500
        try:
            result = subprocess.run(['python', 'optimize_script.py'], capture_output=True, text=True)
            return jsonify({"result": result.stdout})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif tool_name == 'write_code':
        params = request.json
        title = params['title']
        content = params['content']
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {e}"}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"success": True})
    elif tool_name == 'commit_to_github':
        params = request.json
        message = params['message']
        files = params['files']
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Tool not found"}), 404
if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)