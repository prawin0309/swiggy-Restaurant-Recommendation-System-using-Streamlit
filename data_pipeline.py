"""Data pipeline for the Swiggy restaurant recommendation system.

Stages (mirroring the requirement document)
-------------------------------------------
1. **Data understanding and cleaning** - drop duplicates, impute or drop
   missing values, coerce ``rating`` / ``rating_count`` / ``cost`` to numeric,
   and write ``data/cleaned_data.csv``.
2. **Data preprocessing** - one-hot encode ``city``, multi-hot encode the
   comma-separated ``cuisine`` field, scale the numeric columns, save the
   fitted transformers to ``artifacts/encoder.pkl`` and
   ``artifacts/scaler.pkl``, and write ``data/encoded_data.csv``.
   The row index of ``cleaned_data.csv`` and ``encoded_data.csv`` match
   one-for-one, so recommendation indices map straight back.
3. **Persistence** - load the cleaned frame into MySQL through
   ``mysql-connector-python`` (cursor-based, no SQLAlchemy), with an automatic
   SQLite fallback.

Run standalone::

    python data_pipeline.py
"""

from __future__ import annotations

import pickle
import random
import sqlite3
import sys

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

import config

try:  # pragma: no cover - import guard only
    import mysql.connector
    from mysql.connector import Error as MySQLError

    MYSQL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MYSQL_AVAILABLE = False

    class MySQLError(Exception):
        """Placeholder so except-clauses stay valid without the driver."""


def save_artifact(obj, path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    print(f"[save] {path.name}")


def load_artifact(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# Unpickling the encoder bundle and the KMeans model on every query is pure
# waste - they only change when the pipeline is re-run. Cached per path and
# invalidated on mtime, so a retrained artefact is still picked up.
_ARTIFACT_CACHE: dict = {}


def load_artifact_cached(path):
    """``load_artifact`` with an mtime-invalidated in-process cache."""
    key = str(path)
    mtime = path.stat().st_mtime_ns
    cached = _ARTIFACT_CACHE.get(key)
    if cached is None or cached[0] != mtime:
        _ARTIFACT_CACHE[key] = (mtime, load_artifact(path))
    return _ARTIFACT_CACHE[key][1]


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------
class Database:
    """Cursor-based SQL wrapper over MySQL, falling back to SQLite."""

    def __init__(self) -> None:
        self.backend = "sqlite"
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        backend = config.DB_BACKEND
        if backend in ("auto", "mysql") and MYSQL_AVAILABLE:
            try:
                self.conn = self._connect_mysql()
                self.backend = "mysql"
                print(f"[db] connected to MySQL {config.MYSQL_CONFIG['host']}:"
                      f"{config.MYSQL_CONFIG['port']}/"
                      f"{config.MYSQL_CONFIG['database']}")
                return
            except MySQLError as exc:
                if backend == "mysql":
                    raise
                print(f"[db] MySQL unavailable ({exc}); falling back to SQLite.")

        self.conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.backend = "sqlite"
        print(f"[db] connected to SQLite at {config.SQLITE_PATH}")

    @staticmethod
    def _connect_mysql():
        cfg = dict(config.MYSQL_CONFIG)
        database = cfg.pop("database")
        bootstrap = mysql.connector.connect(connection_timeout=5, **cfg)
        cur = bootstrap.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        cur.close()
        bootstrap.close()
        return mysql.connector.connect(connection_timeout=5, database=database, **cfg)

    def _adapt(self, sql: str) -> str:
        return sql.replace("%s", "?") if self.backend == "sqlite" else sql

    def execute(self, sql: str, params: tuple = ()) -> None:
        cur = self.conn.cursor()
        cur.execute(self._adapt(sql), params)
        self.conn.commit()
        cur.close()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        cur = self.conn.cursor()
        cur.executemany(self._adapt(sql), rows)
        self.conn.commit()
        cur.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.backend == "mysql":
            cur = self.conn.cursor(dictionary=True)
            cur.execute(self._adapt(sql), params)
            rows = cur.fetchall()
        else:
            cur = self.conn.cursor()
            cur.execute(self._adapt(sql), params)
            rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

    def create_schema(self) -> None:
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurants (
                row_index    INTEGER PRIMARY KEY,
                id           VARCHAR(30),
                name         VARCHAR(255),
                city         VARCHAR(80),
                rating       DOUBLE,
                rating_count INT,
                cost         DOUBLE,
                cuisine      VARCHAR(255),
                lic_no       VARCHAR(60),
                link         TEXT,
                address      TEXT,
                menu         TEXT
            )
            """
        )
        print("[db] schema ready (restaurants)")

    def load_restaurants(self, frame: pd.DataFrame) -> None:
        self.execute("DELETE FROM restaurants")
        rows = [
            (int(idx), str(r["id"]), r["name"], r["city"], float(r["rating"]),
             int(r["rating_count"]), float(r["cost"]), r["cuisine"],
             r["lic_no"], r["link"], r["address"], r["menu"])
            for idx, r in frame.reset_index(drop=True).iterrows()
        ]
        self.executemany(
            "INSERT INTO restaurants (row_index, id, name, city, rating, "
            "rating_count, cost, cuisine, lic_no, link, address, menu) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
        print(f"[db] loaded {len(rows)} restaurant rows")


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
_NAME_PREFIX = [
    "Sri", "Royal", "Spice", "Urban", "The", "Cafe", "Hotel", "Grand",
    "Tandoor", "Curry", "Coastal", "Golden", "Green", "Star", "Metro",
]
_NAME_CORE = [
    "Kitchen", "Darbar", "Biryani House", "Bistro", "Dhaba", "Tiffins",
    "Grill", "Bakes", "Kulfi", "Junction", "Corner", "Express", "Adda",
    "Canteen", "Table",
]
_MENU_ITEMS = [
    "Butter Naan", "Paneer Tikka", "Chicken Biryani", "Masala Dosa",
    "Idli Sambar", "Veg Fried Rice", "Gulab Jamun", "Filter Coffee",
    "Chole Bhature", "Prawn Curry", "Margherita Pizza", "Chicken Roll",
    "Cold Coffee", "Rasmalai", "Hakka Noodles", "Fish Fry",
]


def generate_synthetic_dataset() -> pd.DataFrame:
    """Build a deterministic dataset with the exact documented column list."""
    rng = random.Random(config.RANDOM_SEED)
    np_rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []

    for i in range(1, config.N_SYNTHETIC_ROWS + 1):
        city = rng.choice(config.CITIES)
        cuisines = rng.sample(config.CUISINES, k=rng.randint(1, 3))
        cost = float(np_rng.choice([100, 150, 200, 250, 300, 400, 500, 600,
                                    800, 1000, 1200, 1500],
                                   p=[.08, .11, .13, .12, .11, .10, .09, .08,
                                      .07, .05, .04, .02]))
        # Pricier and dessert-led places skew slightly higher rated.
        base = 3.4 + 0.0006 * cost + (0.25 if "Desserts" in cuisines else 0.0)
        rating = float(np.clip(base + np_rng.normal(0, 0.45), 1.0, 5.0))
        rating_count = int(max(5, np_rng.lognormal(4.6, 1.0)))

        name = f"{rng.choice(_NAME_PREFIX)} {rng.choice(_NAME_CORE)}"
        rows.append(
            {
                "id": f"SWG{i:06d}",
                "name": name,
                "city": city,
                "rating": round(rating, 1),
                "rating_count": rating_count,
                "cost": cost,
                "cuisine": ", ".join(cuisines),
                "lic_no": f"{rng.randint(10**13, 10**14 - 1)}",
                "link": f"https://www.swiggy.com/restaurants/"
                        f"{name.lower().replace(' ', '-')}-{city.lower()}-{i}",
                "address": f"{rng.randint(1, 250)}, "
                           f"{rng.choice(['MG Road', 'Anna Nagar', 'Sector 14', 'Park Street', 'Banjara Hills'])}, "
                           f"{city}",
                "menu": ", ".join(rng.sample(_MENU_ITEMS, k=rng.randint(3, 6))),
            }
        )

    frame = pd.DataFrame(rows, columns=config.RAW_COLUMNS)

    # Inject dirt: duplicates plus missing values, so cleaning is meaningful.
    duplicates = frame.sample(frac=0.04, random_state=config.RANDOM_SEED)
    frame = pd.concat([frame, duplicates], ignore_index=True)

    for column, fraction in (("rating", 0.05), ("cost", 0.03),
                             ("rating_count", 0.04), ("cuisine", 0.01)):
        idx = frame.sample(frac=fraction,
                           random_state=config.RANDOM_SEED + len(column)).index
        frame.loc[idx, column] = np.nan

    return frame.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)


def load_raw_dataset() -> pd.DataFrame:
    if config.RAW_CSV.exists():
        print(f"[data] using real dataset: {config.RAW_CSV.name}")
        return pd.read_csv(config.RAW_CSV)
    print(f"[data] {config.RAW_CSV.name} not found in {config.DATA_DIR}; "
          "generating a deterministic synthetic catalogue with the documented "
          "schema so the pipeline still runs end to end")
    frame = generate_synthetic_dataset()
    frame.to_csv(config.RAW_CSV, index=False)
    return frame


# ---------------------------------------------------------------------------
# Stage 1: cleaning
# ---------------------------------------------------------------------------
def parse_rating(series: pd.Series) -> pd.Series:
    """`4.4` -> 4.4, `--` / blank -> NaN."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"--": pd.NA, "": pd.NA, "-": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def parse_rating_count(series: pd.Series) -> pd.Series:
    """`1K+ ratings` -> 1000, `Too Few Ratings` -> NaN."""
    lowered = series.astype("string").str.strip().str.lower()
    mapped = pd.to_numeric(lowered.map(config.RATING_COUNT_MAP), errors="coerce")

    # Anything outside the known buckets: salvage the leading number, honouring
    # a K suffix (e.g. an unseen "2K+ ratings").
    unknown = lowered.notna() & ~lowered.isin(config.RATING_COUNT_MAP)
    if bool(unknown.any()):
        extracted = lowered[unknown].str.extract(
            r"(?P<value>[\d.]+)\s*(?P<unit>[kK]?)", expand=True
        )
        value = pd.to_numeric(extracted["value"], errors="coerce")
        is_thousands = extracted["unit"].fillna("").str.lower().eq("k")
        mapped.loc[unknown] = value * np.where(is_thousands.to_numpy(), 1000, 1)

    return pd.to_numeric(mapped, errors="coerce")


def parse_cost(series: pd.Series) -> pd.Series:
    """`₹ 200` -> 200.0 (currency symbols, spaces and separators stripped)."""
    cleaned = (
        series.astype("string")
        .str.replace(r"[^\d.]", "", regex=True)
        .replace({"": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalise_city(series: pd.Series) -> pd.Series:
    """`Vastrapur,Ahmedabad` -> `Ahmedabad`; plain names pass through."""
    return (
        series.astype("string").str.split(",").str[-1].str.strip().str.title()
    )


def is_any_city(city) -> bool:
    """True when the caller wants every city rather than one named city.

    Single source of truth for the sentinel, shared by ``encode_query`` (which
    emits an all-zero city block) and ``models._apply_hard_filters`` (which
    skips the city constraint).
    """
    return city is None or str(city).strip() in ("", config.ANY_CITY)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce numerics and impute or drop missing values."""
    frame = frame.copy()
    before = len(frame)

    for column in config.RAW_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[config.RAW_COLUMNS]

    frame["rating"] = parse_rating(frame["rating"])
    frame["rating_count"] = parse_rating_count(frame["rating_count"])
    frame["cost"] = parse_cost(frame["cost"])

    duplicates = int(frame.duplicated(subset=["name", "city", "address"]).sum())
    frame = frame.drop_duplicates(subset=["name", "city", "address"], keep="first")

    # Rows with no cuisine or no city carry no recommendation signal.
    frame = frame.dropna(subset=["name", "city", "cuisine"])

    # 59% of the catalogue ships as "--" / "Too Few Ratings". Imputing silently
    # would fabricate credibility for unrated venues, so the imputation is
    # flagged per row and surfaced in the UI.
    frame["rating_imputed"] = frame["rating"].isna()
    frame["is_unrated"] = frame["rating"].isna() | frame["rating_count"].isna()

    missing_rating = int(frame["rating"].isna().sum())
    missing_count = int(frame["rating_count"].isna().sum())
    missing_cost = int(frame["cost"].isna().sum())

    frame["city"] = frame["city"].astype("string").str.strip()
    frame[config.CITY_COLUMN_NORMALISED] = normalise_city(frame["city"])

    # Impute within city first (local price levels differ), then globally.
    frame["cost"] = frame.groupby(config.CITY_COLUMN_NORMALISED)["cost"].transform(
        lambda s: s.fillna(s.median())
    )
    frame["cost"] = frame["cost"].fillna(frame["cost"].median())

    frame["rating"] = frame.groupby(config.CITY_COLUMN_NORMALISED)["rating"].transform(
        lambda s: s.fillna(s.median())
    )
    frame["rating"] = frame["rating"].fillna(frame["rating"].median())

    # "Too Few Ratings" genuinely means very few, not average - floor it at 1
    # rather than pulling it up to the catalogue median.
    frame["rating_count"] = frame["rating_count"].fillna(1).astype("int64")

    for column in ("lic_no", "link", "address", "menu", "id"):
        frame[column] = frame[column].astype("string").fillna("")

    frame["cuisine"] = frame["cuisine"].astype("string").str.strip()

    frame = frame.reset_index(drop=True)
    print(f"[clean] {before} -> {len(frame)} rows | {duplicates} duplicates dropped")
    print(f"[clean] imputed: rating={missing_rating:,} "
          f"({100 * missing_rating / max(1, len(frame)):.1f}%), "
          f"rating_count={missing_count:,}, cost={missing_cost:,} "
          "-- flagged in the `rating_imputed` / `is_unrated` columns")
    print(f"[clean] {frame['city'].nunique()} raw city strings -> "
          f"{frame[config.CITY_COLUMN_NORMALISED].nunique()} normalised cities")
    return frame


def split_cuisines(value: str) -> list[str]:
    """Split the comma-separated cuisine field into normalised tokens."""
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


# ---------------------------------------------------------------------------
# Stage 2: encoding
# ---------------------------------------------------------------------------
def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder that works across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def encode_dataset(cleaned: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    """One-hot city, multi-hot cuisine, scale numerics - as a sparse matrix.

    Row *i* of the returned matrix corresponds to row *i* of ``cleaned``, so a
    recommendation index maps straight back to the readable catalogue.
    """
    city_frame = cleaned[[config.CITY_COLUMN_NORMALISED]]
    city_encoder = _make_encoder().fit(city_frame)
    city_encoded = sparse.csr_matrix(
        city_encoder.transform(city_frame), dtype=np.float32
    )
    city_columns = [
        f"city_{name.split('_', 1)[-1]}"
        for name in city_encoder.get_feature_names_out(
            [config.CITY_COLUMN_NORMALISED]
        )
    ]

    cuisine_lists = cleaned["cuisine"].map(split_cuisines)
    cuisine_encoder = MultiLabelBinarizer(sparse_output=True).fit(cuisine_lists)
    cuisine_encoded = sparse.csr_matrix(
        cuisine_encoder.transform(cuisine_lists), dtype=np.float32
    )
    cuisine_columns = [f"cuisine_{c}" for c in cuisine_encoder.classes_]

    scaler = StandardScaler().fit(cleaned[config.NUMERIC_FEATURES])
    numeric_scaled = sparse.csr_matrix(
        scaler.transform(cleaned[config.NUMERIC_FEATURES]).astype(np.float32)
    )

    encoded = sparse.hstack(
        [numeric_scaled, city_encoded, cuisine_encoded], format="csr"
    )
    columns = list(config.NUMERIC_FEATURES) + city_columns + cuisine_columns

    save_artifact(
        {
            "city_encoder": city_encoder,
            "cuisine_encoder": cuisine_encoder,
            "columns": columns,
            "numeric": config.NUMERIC_FEATURES,
            "city_column": config.CITY_COLUMN_NORMALISED,
        },
        config.ENCODER_PKL,
    )
    save_artifact(scaler, config.SCALER_PKL)

    print(f"[encode] encoded matrix shape = {encoded.shape} "
          f"({len(city_columns)} city columns, "
          f"{len(cuisine_columns)} cuisine columns, "
          f"{encoded.nnz:,} non-zeros, "
          f"{100 * encoded.nnz / (encoded.shape[0] * encoded.shape[1]):.2f}% dense)")
    return encoded, columns


def write_encoded_outputs(encoded: sparse.csr_matrix,
                          columns: list[str]) -> None:
    """Persist the encoded matrix as sparse .npz and as the dense CSV."""
    sparse.save_npz(config.ENCODED_NPZ, encoded)
    with open(config.ENCODED_COLUMNS_JSON, "w", encoding="utf-8") as handle:
        json.dump(columns, handle)
    print(f"[data] sparse encoded matrix -> {config.ENCODED_NPZ.name} "
          f"({config.ENCODED_NPZ.stat().st_size / 1e6:.1f} MB)")

    if not config.WRITE_ENCODED_CSV:
        print("[data] encoded_data.csv skipped "
              "(SWIGGY_WRITE_ENCODED_CSV=0)")
        return

    # Densify in chunks: the full dense frame would need ~800 MB of RAM.
    chunk = config.ENCODED_CSV_CHUNK_ROWS
    with open(config.ENCODED_CSV, "w", encoding="utf-8", newline="") as handle:
        for start in range(0, encoded.shape[0], chunk):
            block = pd.DataFrame(
                encoded[start:start + chunk].toarray(), columns=columns
            )
            block.to_csv(handle, index=False, header=(start == 0),
                         float_format="%.5g")
    print(f"[data] dense encoded dataset -> {config.ENCODED_CSV.name} "
          f"({config.ENCODED_CSV.stat().st_size / 1e6:.1f} MB)")


def load_encoded() -> tuple[sparse.csr_matrix, list[str]]:
    """Load the persisted sparse encoded matrix and its column names."""
    encoded = sparse.load_npz(config.ENCODED_NPZ)
    with open(config.ENCODED_COLUMNS_JSON, "r", encoding="utf-8") as handle:
        columns = json.load(handle)
    return encoded.tocsr(), columns


def encode_query(city: str | None, cuisines: list[str], rating: float,
                 rating_count: int, cost: float) -> sparse.csr_matrix:
    """Encode a user preference vector into the trained feature space.

    ``city`` may be ``config.ANY_CITY`` or ``None`` for a nationwide search, in
    which case the whole city block stays at zero and cosine similarity ranks
    on cuisine, rating, review count and cost alone.
    """
    bundle = load_artifact_cached(config.ENCODER_PKL)
    scaler = load_artifact_cached(config.SCALER_PKL)

    city_column = bundle.get("city_column", config.CITY_COLUMN_NORMALISED)
    if not is_any_city(city):
        city_frame = pd.DataFrame(
            {city_column: [normalise_city(pd.Series([city]))[0]]}
        )
        city_encoded = sparse.csr_matrix(
            bundle["city_encoder"].transform(city_frame), dtype=np.float32
        )
    else:
        # "Any" means no city preference: emit an all-zero city block so no
        # single city dominates the cosine score. The hard filters already
        # skip the city constraint for this value.
        city_width = len(bundle["city_encoder"].get_feature_names_out())
        city_encoded = sparse.csr_matrix((1, city_width), dtype=np.float32)

    known = set(bundle["cuisine_encoder"].classes_)
    cuisine_encoded = sparse.csr_matrix(
        bundle["cuisine_encoder"].transform(
            [[c for c in cuisines if c in known]]
        ),
        dtype=np.float32,
    )

    numeric_scaled = sparse.csr_matrix(
        scaler.transform(
            pd.DataFrame([[rating, rating_count, cost]],
                         columns=bundle["numeric"])
        ).astype(np.float32)
    )

    return sparse.hstack(
        [numeric_scaled, city_encoded, cuisine_encoded], format="csr"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline() -> tuple[pd.DataFrame, sparse.csr_matrix, list[str]]:
    raw = load_raw_dataset()
    cleaned = clean_dataset(raw)
    cleaned.to_csv(config.CLEANED_CSV, index=False)
    print(f"[data] cleaned dataset -> {config.CLEANED_CSV.name}")

    encoded, columns = encode_dataset(cleaned)
    write_encoded_outputs(encoded, columns)

    # Matching row counts is necessary but not sufficient. Assert the actual
    # correspondence the recommender relies on, so a future reordering or
    # partial drop fails loudly here instead of silently returning the wrong
    # restaurant for a given index.
    assert encoded.shape[0] == len(cleaned), (
        f"index alignment broken: encoded has {encoded.shape[0]} rows, "
        f"cleaned has {len(cleaned)}"
    )
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned))), (
        "cleaned_data.csv must carry a contiguous 0..n-1 index for positional "
        "lookup to be valid"
    )
    print(f"[verify] index alignment confirmed for {len(cleaned)} rows "
          "(encoded row i == cleaned row i, contiguous 0..n-1)")

    db = Database()
    try:
        db.create_schema()
        db.load_restaurants(cleaned)
        total = db.fetch_all("SELECT COUNT(*) AS n FROM restaurants")[0]["n"]
        print(f"[verify] restaurants row count = {total}")
    finally:
        db.close()

    return cleaned, encoded, columns


def load_datasets() -> tuple[pd.DataFrame, sparse.csr_matrix, list[str]]:
    """Return (cleaned, encoded, columns), running the pipeline if required."""
    if config.CLEANED_CSV.exists() and config.ENCODED_NPZ.exists():
        cleaned = pd.read_csv(config.CLEANED_CSV, low_memory=False)
        encoded, columns = load_encoded()
        return cleaned, encoded, columns
    return run_pipeline()


def main() -> int:
    print("=" * 70)
    print("Swiggy Restaurant Recommendation System - data pipeline")
    print("=" * 70)
    cleaned, encoded, _ = run_pipeline()
    print("\nCleaned data preview:")
    print(cleaned[["name", "city", "cuisine", "rating", "cost"]].head().to_string(index=False))
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
