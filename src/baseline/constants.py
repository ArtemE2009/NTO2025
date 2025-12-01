"""
Project-wide constants (add calibration filename).
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
PROCESSED_DATA_FILENAME = "processed_features.parquet"
IMPLICIT_SVD_MODEL_FILENAME = "implicit_svd.pkl"
CALIBRATION_FILENAME = "calibration.npz"

# --- COLUMN NAMES ---
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"

# Aggregates
F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"

# Exposure / implicit
F_IMPLICIT_SCORE = "implicit_svd_score"
F_USER_TOREAD_COUNT = "user_to_read_count"
F_BOOK_WISHLIST_COUNT = "book_wishlist_count"
F_USER_BOOK_WISHLIST_FLAG = "user_book_in_wishlist"

# Similarity
F_TFIDF_USER_BOOK_SIM = "tfidf_user_book_sim"
F_GENRE_OVERLAP_COUNT = "user_book_genre_overlap_count"
F_GENRE_JACCARD = "user_book_genre_jaccard"

# Metadata columns from raw data
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"

# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 10
