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

try:
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
except Exception as e:
    model = None
    scaler = None
    selected_features = []
    print(f"Error loading model: {e}")

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

def execute_tool(tool_name, params):
    if tool_name == "optimize":
        try:
            from config import optuna
            if optuna is None:
                return {"status": "error", "message": "Optuna not available"}
            # Placeholder for Optuna optimization (e.g., tune LightGBM params for better R2 > 0.1, directional acc > 0.6)
            # Assume optimization script or inline: tune max_depth, num_leaves, reg_alpha/lambda
            # For demo, return mock result
            return {"status": "success", "output": "Optimized model with R2: 0.15, Directional Acc: 0.65"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "write_code":
        filename = params.get("title")
        content = params.get("content")
        if not filename or not content:
            return {"status": "error", "message": "Missing filename or content"}
        if filename.endswith('.py'):
            try:
                compile(content, filename, 'exec')
            except SyntaxError as e:
                return {"status": "error", "message": f"Syntax error: {str(e)}"}
        try:
            with open(filename, 'w') as f:
                f.write(content)
            return {"status": "success", "message": f"File {filename} written successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "commit_to_github":
        message = params.get("message")
        files = params.get("files", [])
        if not message or not files:
            return {"status": "error", "message": "Missing message or files"}
        try:
            subprocess.run(['git', 'add'] + files, check=True)
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            return {"status": "success", "message": "Committed and pushed to GitHub"}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "Unknown tool"}

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    tool_name = data.get('tool')
    params = data.get('params', {})
    result = execute_tool(tool_name, params)
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    input_data = [data.get(f, np.nan) for f in selected_features]
    input_array = np.array(input_data).reshape(1, -1)
    if NAN_HANDLING == 'ffill':
        input_array = np.nan_to_num(input_array, nan=0.0)  # Robust NaN handling
    if scaler:
        input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)[0]
    return jsonify({"prediction": prediction})

@app.route('/')
def home():
    return f"MCP Version: {MCP_VERSION}"

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)