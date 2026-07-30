"""Streamlit application for the Swiggy restaurant recommendation system.

Pages
-----
Find Restaurants · Explore Data · Cuisine & City Insights · Cluster Explorer

Run::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import models
from data_pipeline import load_datasets, split_cuisines

st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
)

PAGES = [
    "Find Restaurants",
    "Explore Data",
    "Cuisine & City Insights",
    "Cluster Explorer",
]


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading restaurant catalogue…")
def get_data():
    """Cleaned catalogue plus the sparse encoded matrix (cached per session)."""
    cleaned, encoded, columns = load_datasets()
    return cleaned, encoded, columns


@st.cache_data(show_spinner=False)
def get_cuisine_options(cleaned: pd.DataFrame) -> list[str]:
    tags = cleaned["cuisine"].map(split_cuisines).explode().dropna()
    counts = tags.value_counts()
    return counts.index.tolist()


@st.cache_data(show_spinner=False)
def city_options(cleaned: pd.DataFrame) -> list[str]:
    """Cities ordered by catalogue coverage, biggest first."""
    column = (
        config.CITY_COLUMN_NORMALISED
        if config.CITY_COLUMN_NORMALISED in cleaned.columns
        else "city"
    )
    return ["Any"] + cleaned[column].value_counts().index.tolist()


def artefacts_warning() -> None:
    st.warning(
        "Model artefacts not found. Run `python data_pipeline.py` then "
        "`python models.py` before using this page."
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_recommend(cleaned: pd.DataFrame, encoded: pd.DataFrame) -> None:
    st.header("🍽️ Find Restaurants")
    st.caption(
        "Your preferences are one-hot encoded into the same feature space as "
        "the catalogue, then ranked by cosine similarity or within a KMeans "
        "cluster. Results are mapped back to the readable dataset."
    )

    cuisine_options = get_cuisine_options(cleaned)

    with st.form("preferences"):
        col_a, col_b, col_c = st.columns(3)

        city = col_a.selectbox("City", city_options(cleaned))
        cuisines = col_a.multiselect(
            "Preferred cuisines", cuisine_options,
            default=cuisine_options[:2] if len(cuisine_options) >= 2 else cuisine_options,
        )
        rating = col_b.slider("Minimum acceptable rating", 1.0, 5.0, 4.0, 0.1)
        cost = col_b.slider("Budget for two (₹)", 100, 2000, 400, 50)
        method = col_c.radio("Recommendation method", models.METHODS)
        top_k = col_c.slider("Results", 3, 25, config.TOP_K)
        apply_filters = col_c.checkbox(
            "Apply hard filters (city / budget / rating)", value=True
        )
        exclude_unrated = col_c.checkbox(
            "Exclude unrated venues", value=True,
            help="59% of the catalogue has no real rating ('--' / "
                 "'Too Few Ratings'). Those rows carry an imputed rating and "
                 "are flagged in the `is_unrated` column.",
        )

        submitted = st.form_submit_button("Recommend restaurants",
                                          type="primary",
                                          use_container_width=True)

    if not submitted:
        return
    if not models.artifacts_ready():
        artefacts_warning()
        return
    if not cuisines:
        st.error("Select at least one cuisine.")
        return

    result = models.recommend(
        city=city,
        cuisines=cuisines,
        rating=rating,
        cost=float(cost),
        method=method,
        top_k=top_k,
        apply_filters=apply_filters,
        exclude_unrated=exclude_unrated,
        cleaned=cleaned,
        encoded=encoded,
    )

    if result.empty:
        st.info("No matches found. Loosen the filters and try again.")
        return

    st.success(f"{len(result)} recommendations via **{method}**")

    for rank, row in enumerate(result.itertuples(index=False), start=1):
        with st.container(border=True):
            head, rating_col, cost_col, sim_col = st.columns([4, 1, 1, 1])
            head.markdown(f"### {rank}. {row.name}")
            head.caption(f"{row.cuisine} · {row.city}")
            rating_col.metric(
                "Rating",
                f"{row.rating:.1f}" + ("*" if getattr(row, "rating_imputed", False) else ""),
            )
            cost_col.metric("Cost for two", f"₹{row.cost:,.0f}")
            sim_col.metric("Match", f"{row.similarity:.2f}")
            with head.expander("Details"):
                st.write(f"**Address:** {row.address}")
                st.write(f"**Menu highlights:** {row.menu}")
                st.write(f"**Reviews:** {row.rating_count:,}")
                if isinstance(row.link, str) and row.link.startswith("http"):
                    st.write(f"**Link:** {row.link}")

    if bool(result.get("rating_imputed", pd.Series(dtype=bool)).any()):
        st.caption("\\* rating imputed from the city median — the source "
                   "catalogue had no rating for this venue.")

    st.plotly_chart(
        px.scatter(result, x="cost", y="rating", size="similarity",
                   color="city", hover_name="name",
                   title="Recommended restaurants — price versus rating"),
        use_container_width=True,
    )

    with st.expander("Raw result table"):
        st.dataframe(
            result[["name", "city", "cuisine", "rating", "rating_count",
                    "cost", "similarity"]],
            use_container_width=True, hide_index=True,
        )


def page_explore(cleaned: pd.DataFrame) -> None:
    st.header("🔎 Explore Data")

    cols = st.columns(4)
    cols[0].metric("Restaurants", f"{len(cleaned):,}")
    cols[1].metric("Cities", cleaned["city"].nunique())
    cols[2].metric("Average rating", f"{cleaned['rating'].mean():.2f}")
    cols[3].metric("Median cost for two", f"₹{cleaned['cost'].median():,.0f}")

    col_a, col_b = st.columns(2)
    city = col_a.selectbox("Filter by city", ["All"] + city_options(cleaned)[1:])
    search = col_b.text_input("Search name, cuisine or address")

    view = cleaned
    if city != "All":
        column = (
            config.CITY_COLUMN_NORMALISED
            if config.CITY_COLUMN_NORMALISED in cleaned.columns
            else "city"
        )
        view = view[view[column] == city]
    if search:
        mask = (
            view["name"].astype(str).str.contains(search, case=False, na=False)
            | view["cuisine"].astype(str).str.contains(search, case=False, na=False)
            | view["address"].astype(str).str.contains(search, case=False, na=False)
        )
        view = view[mask]

    st.caption(f"{len(view):,} of {len(cleaned):,} restaurants "
               "(showing the first 2,000)")
    st.dataframe(
        view[["name", "city", "cuisine", "rating", "rating_count", "cost"]]
        .head(2000),
        use_container_width=True, hide_index=True, height=460,
    )


def page_insights(cleaned: pd.DataFrame) -> None:
    st.header("📊 Cuisine & City Insights")

    cuisines = models.cuisine_popularity(cleaned)
    cities = models.city_summary(cleaned)

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(cuisines.head(15).sort_values("restaurants"),
               x="restaurants", y="cuisine_tag", orientation="h",
               color="avg_rating", color_continuous_scale="Viridis",
               title="Most common cuisines"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(cities.sort_values("avg_cost"), x="avg_cost", y="city",
               orientation="h", color="avg_rating",
               color_continuous_scale="Plasma",
               title="Average cost for two by city"),
        use_container_width=True,
    )

    st.plotly_chart(
        px.scatter(cuisines.head(40), x="avg_cost", y="avg_rating",
                   size="restaurants", hover_name="cuisine_tag",
                   color="cuisine_tag",
                   title="Cuisine positioning — price versus rating (top 40)"),
        use_container_width=True,
    )

    top_cities = cities.head(8)["city"].tolist()
    column = (
        config.CITY_COLUMN_NORMALISED
        if config.CITY_COLUMN_NORMALISED in cleaned.columns
        else "city"
    )
    st.plotly_chart(
        px.histogram(cleaned[cleaned[column].isin(top_cities)], x="rating",
                     nbins=30, color=column,
                     title="Rating distribution — eight largest cities"),
        use_container_width=True,
    )

    st.dataframe(cities, use_container_width=True, hide_index=True)


def page_clusters(cleaned: pd.DataFrame, encoded: pd.DataFrame) -> None:
    st.header("🧩 Cluster Explorer")
    if not config.KMEANS_PKL.exists():
        artefacts_warning()
        return

    labelled = cleaned.copy()
    labelled["cluster"] = models.cluster_assignments(encoded)

    profile = (
        labelled.groupby("cluster", as_index=False)
        .agg(restaurants=("name", "count"),
             avg_rating=("rating", "mean"),
             avg_cost=("cost", "mean"),
             top_city=("city", lambda s: s.mode().iat[0]),
             top_cuisine=("cuisine", lambda s: s.mode().iat[0]))
        .round(2)
    )

    plot_sample = labelled.sample(
        min(8000, len(labelled)), random_state=config.RANDOM_SEED
    )
    st.plotly_chart(
        px.scatter(plot_sample, x="cost", y="rating",
                   color=plot_sample["cluster"].astype(str),
                   hover_name="name", opacity=0.5,
                   labels={"color": "cluster"},
                   title="KMeans clusters in price/rating space "
                         f"({len(plot_sample):,}-row sample)"),
        use_container_width=True,
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)

    chosen = st.selectbox("Inspect a cluster", sorted(labelled["cluster"].unique()))
    st.dataframe(
        labelled[labelled["cluster"] == chosen][
            ["name", "city", "cuisine", "rating", "cost"]
        ].head(1000),
        use_container_width=True, hide_index=True, height=400,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("🍽️ Swiggy Recommender")
    choice = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    st.sidebar.caption(
        "One-hot encoding + cosine similarity / KMeans clustering. "
        "Indices map back to cleaned_data.csv."
    )

    cleaned, encoded, _ = get_data()

    if choice == "Find Restaurants":
        page_recommend(cleaned, encoded)
    elif choice == "Explore Data":
        page_explore(cleaned)
    elif choice == "Cuisine & City Insights":
        page_insights(cleaned)
    elif choice == "Cluster Explorer":
        page_clusters(cleaned, encoded)


if __name__ == "__main__":
    main()
