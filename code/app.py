import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import joblib
import subprocess
from config import model_file_path, selected_features_path, NAN_HANDLING, scaler_file_path, LOW_VARIANCE_THRESHOLD, FEATURES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
try:
    import optuna
except Exception:
    optuna = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

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

@app.route('/tool/<tool_name>', methods=['POST'])
def execute_tool(tool_name):
    if tool_name == 'optimize':
        if optuna is None or LGBMRegressor is None:
            return jsonify({'error': 'Required libraries not available'})
        df = pd.read_csv(training_price_data_path)
        df = df.fillna(method=NAN_HANDLING)
        X = df[FEATURES]
        y = df['log_return']  # Assume target column
        selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
        selector.fit(X)
        selected_features = [f for f, i in zip(FEATURES, selector.get_support()) if i]
        joblib.dump(selected_features, selected_features_path)
        X_selected = selector.transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        joblib.dump(scaler, scaler_file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7)
            }
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            directional_acc = np.mean(np.sign(preds) == np.sign(y_test))
            corr = np.corrcoef(preds, y_test)[0, 1]
            return - (r2 + 0.5 * directional_acc + 0.3 * corr)
        study = optuna.create_study()
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        model = LGBMRegressor(**best_params)
        model.fit(X_scaled, y)
        joblib.dump(model, model_file_path)
        results = {'best_params': best_params, 'best_value': -study.best_value}
        with open(best_model_info_path, 'w') as f:
            json.dump(results, f)
        return jsonify(results)
    elif tool_name == 'write_code':
        params = request.json
        title = params['title']
        content = params['content']
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({'error': str(e)})
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({'status': 'written'})
    elif tool_name == 'commit_to_github':
        params = request.json
        message = params['message']
        files = params['files']
        for file in files:
            subprocess.run(['git', 'add', file])
        subprocess.run(['git', 'commit', '-m', message])
        subprocess.run(['git', 'push'])
        return jsonify({'status': 'committed'})
    else:
        return jsonify({'error': 'Unknown tool'})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)