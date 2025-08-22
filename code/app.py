import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, FEATURES, OPTUNA_TRIALS, MODEL_PARAMS, LOW_VARIANCE_THRESHOLD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import optuna
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
MODEL_CACHE = {}

def load_model():
    if os.path.exists(model_file_path):
        MODEL_CACHE['model'] = joblib.load(model_file_path)

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke/<tool_name>', methods=['POST'])
def invoke_tool(tool_name):
    if tool_name == 'optimize':
        result = run_optuna_optimization()
        return jsonify(result)
    elif tool_name == 'write_code':
        data = request.json
        title = data['title']
        content = data['content']
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({'error': str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({'success': True})
    elif tool_name == 'commit_to_github':
        data = request.json
        message = data['message']
        files = data.get('files', [])
        for file in files:
            subprocess.run(['git', 'add', file])
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Unknown tool'}), 404

def run_optuna_optimization():
    df = pd.read_csv('data/price_data.csv')  # Assume path
    df = df.fillna(method=NAN_HANDLING)
    X = df[FEATURES]
    y = df['log_return']
    corrs = X.corrwith(y).abs()
    selected = corrs[corrs > 0.25].index.tolist()
    X = X[selected]
    selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    X = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, model_file_path)
    with open(selected_features_path, 'w') as f:
        json.dump(selected, f)
    return {'best_r2': study.best_value, 'best_params': best_params}

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    features = pd.DataFrame({f: [data.get(f, np.nan)] for f in FEATURES})
    features = features.fillna(method=NAN_HANDLING)
    features = np.nan_to_num(features.values)
    model = MODEL_CACHE['model']
    pred = model.predict(features)[0]
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)