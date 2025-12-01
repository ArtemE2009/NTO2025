"""
Configuration file (S3: enable TF-IDF + exposure; baseline-like params; BERT/SVD off).
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"


# --- PARAMETERS ---
N_SPLITS = 5
RANDOM_STATE = 42
TARGET = constants.COL_TARGET

# --- TEMPORAL SPLIT CONFIG ---
TEMPORAL_SPLIT_RATIO = 0.8

# --- FLAGS ---
USE_TFIDF = True
USE_BERT = True
USE_EXPOSURE = True


# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 80
MODEL_FILENAME = "lgb_model.txt"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.75


# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- MODEL PARAMETERS (baseline-like + mild regularization) ---
LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 3500,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.3,
    "lambda_l2": 0.2,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],
}
