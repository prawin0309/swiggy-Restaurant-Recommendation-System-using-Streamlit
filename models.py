"""Recommendation engines for the Swiggy restaurant system.

Two interchangeable methodologies, both operating on ``encoded_data.csv``
and mapping their result indices back to ``cleaned_data.csv``:

* **Cosine similarity** - rank every restaurant against the encoded user
  preference vector.
* **KMeans clustering** - assign the preference vector to a cluster, then rank
  only within that cluster (fast, and produces more thematically coherent
  results on large catalogues).

Run standalone::

    python models.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_pipeline import (
    encode_query,
    is_any_city,
    load_artifact_cached,
    load_datasets,
    normalise_city,
    save_artifact,
    split_cuisines,
)

METHODS = ("Cosine Similarity", "KMeans Clustering")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_kmeans(encoded, n_clusters: int | None = None) -> dict:
    """Fit and persist KMeans over the sparse encoded dataset."""
    n_clusters = n_clusters or min(config.N_CLUSTERS, max(2, encoded.shape[0] // 50))

    # MiniBatch is used above ~50k rows: full Lloyd iterations on a 148k x 680
    # matrix are needlessly slow for a catalogue-segmentation task.
    if encoded.shape[0] > 50_000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=config.RANDOM_SEED,
            n_init=5, batch_size=4096, max_iter=300,
        )
        flavour = "MiniBatchKMeans"
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED,
                        n_init=10)
        flavour = "KMeans"

    labels = kmeans.fit_predict(encoded)

    sample = min(5000, encoded.shape[0])
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(encoded.shape[0], sample, replace=False)
    score = float(silhouette_score(encoded[idx], labels[idx]))

    save_artifact(kmeans, config.KMEANS_PKL)
    print(f"[kmeans] {flavour} k={n_clusters}  silhouette={score:.4f}  "
          f"(sampled {sample:,} of {encoded.shape[0]:,} rows)")
    return {"kmeans": kmeans, "labels": labels, "silhouette": score,
            "n_clusters": n_clusters}


def artifacts_ready() -> bool:
    return (
        config.ENCODER_PKL.exists()
        and config.SCALER_PKL.exists()
        and config.KMEANS_PKL.exists()
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
# Candidates fetched per requested result before de-duplication. Large
# enough that dropping repeated venues still leaves a full page of results.
OVERFETCH_FACTOR = 6


def _apply_hard_filters(cleaned: pd.DataFrame, city: str | None,
                        max_cost: float | None,
                        min_rating: float | None,
                        max_rating: float | None = None) -> np.ndarray:
    """Positional indices surviving the user's non-negotiable constraints."""
    mask = pd.Series(True, index=cleaned.index)
    if not is_any_city(city):
        target = normalise_city(pd.Series([city]))[0]
        mask &= cleaned[config.CITY_COLUMN_NORMALISED].astype("string") == target
    if max_cost is not None:
        mask &= cleaned["cost"] <= max_cost
    if min_rating is not None:
        mask &= cleaned["rating"] >= min_rating
    if max_rating is not None:
        mask &= cleaned["rating"] <= max_rating
    return np.flatnonzero(mask.to_numpy())


def recommend(
    city: str | None,
    cuisines: list[str],
    rating: float,
    cost: float,
    max_rating: float | None = None,
    rating_count: int = 100,
    method: str = "Cosine Similarity",
    top_k: int = config.TOP_K,
    apply_filters: bool = True,
    exclude_unrated: bool = True,
    cleaned: pd.DataFrame | None = None,
    encoded=None,
) -> pd.DataFrame:
    """Return the ``top_k`` recommended restaurants for a preference vector.

    ``city`` may be ``config.ANY_CITY`` or ``None`` for a nationwide search.

    Result positions are mapped back onto the non-encoded ``cleaned_data.csv``
    so the caller always receives human-readable rows.
    """
    if cleaned is None or encoded is None:
        cleaned, encoded, _ = load_datasets()

    query = encode_query(city, cuisines, rating, rating_count, cost)

    relaxed_to: str | None = None

    if apply_filters:
        # Constraint tiers, strictest first. The old implementation went
        # straight to cost * 1.35 / rating - 0.7 and only tightened later,
        # so a "4.0+ stars, up to Rs.400" request could return a 3.3-star
        # venue at Rs.540 even while thousands of exact matches existed.
        # Now the exact tier is used whenever it yields anything at all, and
        # the caller is told when a looser tier had to be used instead.
        # ``max_rating`` caps the band the UI asked for ("3.0+ stars" means
        # 3.0-3.9), so it is never widened - only the floor and the budget are.
        tiers = (
            (None, cost, rating, max_rating),
            ("budget +35% / rating -0.7", cost * 1.35, max(0.0, rating - 0.7),
             max_rating),
            ("city only", None, None, None),
        )
        rated = (
            np.flatnonzero(~cleaned["is_unrated"].to_numpy(dtype=bool))
            if exclude_unrated and "is_unrated" in cleaned.columns
            else None
        )

        candidates = np.empty(0, dtype=int)
        for label, max_cost, min_rating, cap_rating in tiers:
            found = _apply_hard_filters(cleaned, city, max_cost, min_rating,
                                        cap_rating)
            if rated is not None:
                only_rated = np.intersect1d(found, rated)
                if len(only_rated):
                    found = only_rated
            if len(found):
                candidates, relaxed_to = found, label
                break

        if not len(candidates):  # Nothing anywhere: fall back to everything.
            candidates = np.arange(len(cleaned))
            relaxed_to = "the whole catalogue"
    else:
        candidates = np.arange(len(cleaned))

    subset = encoded[candidates]

    if method == "KMeans Clustering" and config.KMEANS_PKL.exists():
        kmeans = load_artifact_cached(config.KMEANS_PKL)
        target_cluster = int(kmeans.predict(query)[0])
        cluster_labels = kmeans.predict(subset)
        in_cluster = np.flatnonzero(cluster_labels == target_cluster)
        if len(in_cluster) >= top_k:
            candidates = candidates[in_cluster]
            subset = subset[in_cluster]

    scores = cosine_similarity(query, subset).ravel()

    # One-hot preference vectors produce large blocks of tied 1.00 scores, and
    # the catalogue contains the same brand at several addresses. Over-fetch,
    # collapse duplicate venues, then break ties on popularity so the top-N is
    # genuinely diverse rather than an arbitrary slice of identical scores.
    pool = min(len(scores), max(top_k * OVERFETCH_FACTOR, top_k))
    order = np.argsort(scores)[::-1][:pool]
    chosen = candidates[order]

    result = cleaned.iloc[chosen].copy()
    result["similarity"] = scores[order].round(4)

    # Sort BEFORE de-duplicating: drop_duplicates(keep="first") keeps whichever
    # row it meets first, so the ordering has to be final by then or the
    # surviving outlet of a chain is arbitrary rather than the best-rated one.
    tie_breakers = ["similarity"]
    ascending = [False]
    for column in ("rating", "rating_count"):
        if column in result.columns:
            tie_breakers.append(column)
            ascending.append(False)
    result = result.sort_values(tie_breakers, ascending=ascending, kind="mergesort")

    # De-duplicate on the *normalised* city. The raw `city` column carries the
    # locality ("Koramangala,Bangalore" vs "BTM,Bangalore"), so keying on it
    # lets one chain occupy half the page under different neighbourhoods -
    # which is the exact outcome this step exists to prevent. Keying on
    # base_city collapses a brand within a city while still allowing the same
    # brand to surface in a different city on a nationwide search.
    city_key = (config.CITY_COLUMN_NORMALISED
                if config.CITY_COLUMN_NORMALISED in result.columns else "city")
    dedupe_keys = [c for c in ("name", city_key) if c in result.columns]
    if dedupe_keys:
        result = result.drop_duplicates(subset=dedupe_keys, keep="first")

    result = result.head(top_k)
    result["match_method"] = method
    result = result.reset_index(drop=True)
    # Surfaced by the UI so a widened search is never passed off as an exact one.
    result.attrs["relaxed_to"] = relaxed_to
    return result


def cluster_assignments(encoded=None) -> np.ndarray:
    """Cluster label for every row of the encoded dataset."""
    if encoded is None:
        _, encoded, _ = load_datasets()
    kmeans = load_artifact_cached(config.KMEANS_PKL)
    return kmeans.predict(encoded)


def cuisine_popularity(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Restaurant count, mean rating and mean cost per cuisine tag."""
    exploded = cleaned.assign(
        cuisine_tag=cleaned["cuisine"].map(split_cuisines)
    ).explode("cuisine_tag")
    return (
        exploded.groupby("cuisine_tag", as_index=False)
        .agg(restaurants=("name", "count"),
             avg_rating=("rating", "mean"),
             avg_cost=("cost", "mean"))
        .sort_values("restaurants", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )


def city_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    column = (
        config.CITY_COLUMN_NORMALISED
        if config.CITY_COLUMN_NORMALISED in cleaned.columns
        else "city"
    )
    return (
        cleaned.groupby(column, as_index=False)
        .agg(restaurants=("name", "count"),
             avg_rating=("rating", "mean"),
             avg_cost=("cost", "mean"),
             total_reviews=("rating_count", "sum"))
        .rename(columns={column: "city"})
        .sort_values("restaurants", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("Swiggy Restaurant Recommendation System - model training")
    print("=" * 70)
    cleaned, encoded, columns = load_datasets()
    print(f"[data] cleaned={cleaned.shape}  encoded={encoded.shape} "
          f"({len(columns)} feature columns, {encoded.nnz:,} non-zeros)")

    train_kmeans(encoded)

    demo = {
        "city": "Bangalore",
        "cuisines": ["Biryani", "North Indian"],
        "rating": 4.2,
        "cost": 400.0,
    }

    print(f"\n[demo] preferences: {demo}")

    for method in METHODS:
        print(f"\n--- {method} ---")
        result = recommend(method=method, top_k=5, cleaned=cleaned,
                           encoded=encoded, **demo)
        print(result[["name", "city", "cuisine", "rating", "cost",
                      "similarity"]].to_string(index=False))

    print("\nTop cuisines by coverage")
    print(cuisine_popularity(cleaned).head(10).to_string(index=False))

    print("\nTop 15 cities by coverage")
    print(city_summary(cleaned).head(15).to_string(index=False))

    print("\nModel training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
