import os
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import optuna
except Exception:
    optuna = None
data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
best_model_info_path = os.path.join(data_base_path, 'best_model.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
TOKEN = os.getenv('TOKEN', 'BTC')
TIMEFRAME = os.getenv('TIMEFRAME', '8h')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LightGBM')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
NAN_HANDLING = 'fill_median'  # Robust NaN handling
LOW_VARIANCE_THRESHOLD = 0.01  # Low-variance check
OPTUNA_TRIALS = 200  # Increased for better tuning
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 6,  # Adjusted
    'num_leaves': 31,  # Adjusted
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'reg_alpha': 0.1,  # Added regularization
    'reg_lambda': 0.1,  # Added regularization
    'bagging_fraction': 0.8,  # For ensembling/stability
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'random_state': 42
}
# Expanded features for better prediction, including sign/log-return lags and momentum filters
FEATURES = [
    'log_return_lag1', 'log_return_lag2', 'log_return_lag3', 'log_return_lag4', 'log_return_lag5', 'log_return_lag6', 'log_return_lag7',
    'sign_return', 'sign_return_lag1', 'sign_return_lag2', 'sign_return_lag3', 'sign_return_lag4', 'sign_return_lag5', 'sign_return_lag6', 'sign_return_lag7',
    'momentum_3', 'momentum_5', 'momentum_7',  # Added momentum filters for stability and correlation
    'vader_compound' if SentimentIntensityAnalyzer else None  # Add VADER sentiment if available
]
# Filter None from FEATURES
FEATURES = [f for f in FEATURES if f is not None]