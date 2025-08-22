import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from lightgbm import LGBMRegressor
import joblib
from config import (
    FEATURES, MODEL_PARAMS, ENABLE_ENSEMBLE, NAN_HANDLING, 
    LOW_VARIANCE_THRESHOLD, model_file_path, scaler_file_path,
    selected_features_path, best_model_info_path, training_price_data_path
)

def load_data():
    df = pd.read_csv(training_price_data_path)
    df['log_return'] = np.log(df['close']).diff()
    df['log_return_lag1'] = df['log_return'].shift(1)
    df['log_return_lag2'] = df['log_return'].shift(2)
    df['sign_return'] = np.sign(df['log_return'])
    df['momentum_filter_1'] = (df['log_return'] > 0).rolling(window=5).sum()
    
    # VADER sentiment
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df['vader_sentiment_compound'] = df['news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    except Exception:
        print("VADER sentiment analysis not available. Skipping.")
    
    return df.dropna()

def preprocess_data(df):
    # Handle NaN values
    if NAN_HANDLING == 'fillna_mean':
        df = df.fillna(df.mean())
    elif NAN_HANDLING == 'dropna':
        df = df.dropna()
    
    # Remove low-variance features
    variances = df[FEATURES].var()
    high_variance_features = variances[variances > LOW_VARIANCE_THRESHOLD].index.tolist()
    df = df[high_variance_features + ['log_return']]
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[high_variance_features])
    y = df['log_return'].values
    
    # Save scaler and selected features
    joblib.dump(scaler, scaler_file_path)
    with open(selected_features_path, 'w') as f:
        json.dump(high_variance_features, f)
    
    return X, y, scaler, high_variance_features

def train_model(optimize=False):
    df = load_data()
    X, y, scaler, selected_features = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if optimize:
        try:
            import optuna
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.1)
                }
                
                model = LGBMRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return -r2_score(y_test, y_pred)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            
            best_params = study.best_params
            model = LGBMRegressor(**best_params)
        except Exception:
            print("Optuna not available. Using default parameters.")
            model = LGBMRegressor(**MODEL_PARAMS)
    else:
        model = LGBMRegressor(**MODEL_PARAMS)
    
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_file_path)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    directional_accuracy = accuracy_score(np.sign(y_test), np.sign(y_pred))
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    result = {
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'correlation': correlation,
        'params': model.get_params()
    }
    
    with open(best_model_info_path, 'w') as f:
        json.dump(result, f)
    
    return result

def predict(data):
    # Load model and scaler
    model = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)
    
    # Load selected features
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    
    # Prepare input data
    df = pd.DataFrame(data, index=[0])
    X = df[selected_features]
    
    # Scale input data
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    
    if ENABLE_ENSEMBLE:
        # Simple ensemble: average of multiple predictions
        predictions = [model.predict(X_scaled) for _ in range(5)]
        prediction = np.mean(predictions, axis=0)
    
    return prediction