"""Configuration for the Swiggy restaurant recommendation system."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

for _folder in (DATA_DIR, ARTIFACT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "swiggy.csv"
CLEANED_CSV = DATA_DIR / "cleaned_data.csv"
ENCODED_CSV = DATA_DIR / "encoded_data.csv"
ENCODED_NPZ = DATA_DIR / "encoded_data.npz"
ENCODED_COLUMNS_JSON = DATA_DIR / "encoded_columns.json"
ENCODER_PKL = ARTIFACT_DIR / "encoder.pkl"
SCALER_PKL = ARTIFACT_DIR / "scaler.pkl"
KMEANS_PKL = ARTIFACT_DIR / "kmeans.pkl"
SQLITE_PATH = DATA_DIR / "guvi_db.sqlite3"

MYSQL_CONFIG = {
    "host": os.getenv("SWIGGY_DB_HOST", "localhost"),
    "port": int(os.getenv("SWIGGY_DB_PORT", "3306")),
    "user": os.getenv("SWIGGY_DB_USER", "root"),
    "password": os.getenv("SWIGGY_DB_PASSWORD", "root"),
    "database": os.getenv("SWIGGY_DB_NAME", "guvi_db"),
}
DB_BACKEND = os.getenv("SWIGGY_DB_BACKEND", "auto").lower()

RANDOM_SEED = 42
N_SYNTHETIC_ROWS = 4000
TOP_K = 10
N_CLUSTERS = 12

# The encoded matrix is ~148k x ~680 and is held in memory as a scipy sparse
# matrix (encoded_data.npz). The dense CSV deliverable is ~200 MB, so writing it
# is opt-out via the environment variable below.
WRITE_ENCODED_CSV = os.getenv("SWIGGY_WRITE_ENCODED_CSV", "1") == "1"
ENCODED_CSV_CHUNK_ROWS = 10_000

# "Locality,City" strings are normalised to their trailing city before one-hot
# encoding, so restaurants in different neighbourhoods of the same city share a
# city dimension. The raw string is preserved in cleaned_data.csv.
CITY_COLUMN_RAW = "city"
CITY_COLUMN_NORMALISED = "base_city"

# Sentinel for "search every city". One constant instead of the string "Any"
# repeated across the UI, the query encoder and the hard filters.
ANY_CITY = "Any"

# rating_count arrives as bucketed text ("1K+ ratings", "Too Few Ratings").
RATING_COUNT_MAP = {
    "too few ratings": None,
    "20+ ratings": 20,
    "50+ ratings": 50,
    "100+ ratings": 100,
    "500+ ratings": 500,
    "1k+ ratings": 1000,
    "5k+ ratings": 5000,
    "10k+ ratings": 10000,
}

# Schema from the requirement document.
RAW_COLUMNS = [
    "id", "name", "city", "rating", "rating_count", "cost", "cuisine",
    "lic_no", "link", "address", "menu",
]
CATEGORICAL_FEATURES = ["city", "cuisine"]
NUMERIC_FEATURES = ["rating", "rating_count", "cost"]

CITIES = [
    "Bangalore", "Chennai", "Hyderabad", "Mumbai", "Delhi", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Coimbatore", "Lucknow",
]

CUISINES = [
    "North Indian", "South Indian", "Chinese", "Biryani", "Desserts",
    "Fast Food", "Beverages", "Pizza", "Burgers", "Rolls", "Bakery",
    "Continental", "Mughlai", "Italian", "Street Food", "Seafood",
    "Healthy Food", "Ice Cream", "Thai", "Andhra",
]
