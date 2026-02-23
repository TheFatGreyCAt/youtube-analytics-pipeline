"""
Configuration for Viral Prediction Model
"""

# Model Configuration
VIRAL_PERCENTILE = 0.9  # Top 10% considered viral

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Features to use for training
FEATURE_COLUMNS = [
    # Video engagement metrics
    'view_count',
    'like_count', 
    'comment_count',
    'like_rate_pct',
    'comment_rate_pct',
    # Video metadata
    'duration_seconds',
    # Channel metrics
    'subscriber_count',
    'video_count',
    'channel_total_views',
    'channel_total_videos',
    'channel_avg_views',
    'channel_avg_like_rate',
    'channel_avg_comment_rate'
]

# Training Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Data Fetching
MAX_ROWS = 10000

# Model Persistence
MODEL_PATH = 'viral_predictor_model.pkl'
