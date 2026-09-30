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
from scipy import sparse

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
    return [config.ANY_CITY] + cleaned[column].value_counts().index.tolist()


def artefacts_warning() -> None:
    st.warning(
        "Model artefacts not found. Run `python data_pipeline.py` then "
        "`python models.py` before using this page."
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
PAGE_SIZE = 10
MAX_RESULTS = 100

# Plain-language labels for the two ranking strategies. The underlying values
# stay exactly as models.METHODS defines them, so nothing downstream changes.
METHOD_LABELS = {
    "Cosine Similarity":
        "Similar restaurants — closest match to your taste profile "
        "(Cosine Similarity)",
    "KMeans Clustering":
        "More like these — picks from the same restaurant group "
        "(KMeans Clustering)",
}

# Short form for the results banner, where the full sentence is too long.
METHOD_SHORT = {
    "Cosine Similarity": "Similar restaurants (Cosine Similarity)",
    "KMeans Clustering": "More like these (KMeans Clustering)",
}

# Rating bands, not open-ended floors: "3.0+ stars" is 3.0-3.9, so the three
# bands partition the catalogue instead of nesting inside one another.
RATING_BANDS: dict[str, tuple[float | None, float | None]] = {
    "Any rating": (None, None),
    "2.0+ stars": (2.0, 2.9),
    "3.0+ stars": (3.0, 3.9),
    "4.0+ stars": (4.0, 5.0),
}
COST_CHOICES = [150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

SORT_RULES = {
    "Best match": (["similarity", "rating", "rating_count"], [False, False, False]),
    "Rating: high to low": (["rating", "rating_count"], [False, False]),
    "Rating: low to high": (["rating"], [True]),
    "Cost: low to high": (["cost", "rating"], [True, False]),
    "Cost: high to low": (["cost", "rating"], [False, False]),
    "Most reviewed": (["rating_count", "rating"], [False, False]),
    "Name: A to Z": (["name"], [True]),
}


def _cost_label(value: int) -> str:
    return f"Up to ₹{value:,}"


def _sort_results(frame: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Re-order a result pool without recomputing the recommendation."""
    if frame.empty:
        return frame
    columns, ascending = SORT_RULES.get(choice, SORT_RULES["Best match"])
    keep = [(c, a) for c, a in zip(columns, ascending) if c in frame.columns]
    if not keep:
        return frame
    return frame.sort_values(
        [c for c, _ in keep], ascending=[a for _, a in keep], kind="mergesort"
    ).reset_index(drop=True)


def _turn_page(delta: int, pages: int) -> None:
    """Page callback: runs before the rerun, so the nav renders in sync."""
    current = st.session_state.get("rec_page", 0)
    st.session_state["rec_page"] = max(0, min(pages - 1, current + delta))


def _render_card(rank: int, row) -> None:
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


def page_recommend(cleaned: pd.DataFrame, encoded: sparse.csr_matrix) -> None:
    st.header("🍽️ Find Restaurants")
    st.caption(
        "Your preferences are one-hot encoded into the same feature space as "
        "the catalogue, then ranked by cosine similarity or within a KMeans "
        "cluster. Results are mapped back to the readable dataset."
    )

    cuisine_options = get_cuisine_options(cleaned)

    with st.form("preferences"):
        col_a, col_b, col_c = st.columns(3)

        city = col_a.selectbox(
            "City", city_options(cleaned),
            help=f"'{config.ANY_CITY}' searches the whole catalogue: the city "
                 "block of the query vector is zeroed and the city filter is "
                 "skipped, so ranking falls back to cuisine, rating and cost.",
        )
        cuisines = col_a.multiselect(
            "Preferred cuisines", cuisine_options,
            default=cuisine_options[:2] if len(cuisine_options) >= 2 else cuisine_options,
        )
        rating_band = col_b.selectbox(
            "Minimum acceptable rating", list(RATING_BANDS),
            index=list(RATING_BANDS).index("4.0+ stars"),
            help="Bands, not floors: 2.0+ is 2.0-2.9, 3.0+ is 3.0-3.9, "
                 "4.0+ is 4.0-5.0.",
        )
        rating_floor, rating_cap = RATING_BANDS[rating_band]
        cost = col_b.selectbox(
            "Budget for two (₹)", COST_CHOICES,
            index=COST_CHOICES.index(400), format_func=_cost_label,
        )
        sort_choice = col_b.selectbox(
            "Sort results by", list(SORT_RULES),
            help="Applied to the whole result set, not just the current page.",
        )
        method = col_c.radio(
            "Recommendation method", models.METHODS,
            format_func=lambda m: METHOD_LABELS.get(m, m),
        )
        apply_filters = col_c.checkbox(
            "Apply hard filters (city / budget / rating)", value=True
        )
        exclude_unrated = col_c.checkbox(
            "Exclude unrated venues", value=True,
            help="59% of the catalogue has no real rating ('--' / "
                 "'Too Few Ratings'). Those rows carry an imputed rating and "
                 "are flagged in the `is_unrated` column.",
        )
        col_c.caption(f"{PAGE_SIZE} results per page — page through them below.")

        submitted = st.form_submit_button("Recommend restaurants",
                                          type="primary",
                                          width="stretch")

    if submitted:
        if not models.artifacts_ready():
            artefacts_warning()
            return
        if not cuisines:
            st.error("Select at least one cuisine.")
            return

        signature = (city, tuple(cuisines), rating_band, float(cost), method,
                     bool(apply_filters), bool(exclude_unrated))

        # Only recompute when the query itself changed; a sort-only change
        # re-orders the pool already in session state.
        if st.session_state.get("rec_signature") != signature:
            with st.spinner("Ranking restaurants…"):
                pool = models.recommend(
                    city=city,
                    cuisines=cuisines,
                    # "Any rating" has no floor to encode, and a 1.0 there
                    # would make the query vector look for BAD restaurants.
                    # The catalogue median is the neutral stand-in.
                    rating=float(rating_floor if rating_floor is not None
                                 else cleaned["rating"].median()),
                    max_rating=rating_cap,
                    cost=float(cost),
                    method=method,
                    top_k=MAX_RESULTS,
                    apply_filters=apply_filters,
                    exclude_unrated=exclude_unrated,
                    cleaned=cleaned,
                    encoded=encoded,
                )
            st.session_state["rec_signature"] = signature
            st.session_state["rec_pool"] = pool
            st.session_state["rec_relaxed"] = pool.attrs.get("relaxed_to")
            st.session_state["rec_city"] = city
            st.session_state["rec_method"] = method

        st.session_state["rec_sort"] = sort_choice
        st.session_state["rec_page"] = 0

    pool = st.session_state.get("rec_pool")
    if pool is None:
        return
    if pool.empty:
        st.info("No matches found. Loosen the filters and try again.")
        return

    result = _sort_results(pool, st.session_state.get("rec_sort", "Best match"))
    total = len(result)
    pages = max(1, -(-total // PAGE_SIZE))
    page = min(st.session_state.get("rec_page", 0), pages - 1)

    searched_city = st.session_state.get("rec_city", city)
    scope = ("across every city" if config.ANY_CITY == searched_city
             else f"in **{searched_city}**")
    used = st.session_state.get("rec_method", method)
    st.success(f"{total} recommendations {scope} via "
               f"**{METHOD_SHORT.get(used, used)}**")

    relaxed = st.session_state.get("rec_relaxed")
    if relaxed:
        st.warning(
            f"No restaurant met every constraint exactly, so the search was "
            f"widened to **{relaxed}** — some results sit above your budget or "
            "below your minimum rating."
        )

    st.session_state["rec_page"] = page

    prev_col, info_col, next_col = st.columns([1, 3, 1])
    prev_col.button(
        "◀ Previous", key="rec_prev", width="stretch", disabled=page == 0,
        on_click=_turn_page, args=(-1, pages),
    )
    next_col.button(
        "Next ▶", key="rec_next", width="stretch", disabled=page >= pages - 1,
        on_click=_turn_page, args=(1, pages),
    )

    first = page * PAGE_SIZE
    window = result.iloc[first:first + PAGE_SIZE]
    info_col.markdown(
        f"<div style='text-align:center;padding-top:0.4rem'>Page "
        f"<b>{page + 1}</b> of <b>{pages}</b> — showing "
        f"<b>{first + 1}–{first + len(window)}</b> of <b>{total}</b>"
        f" · sorted by <b>{st.session_state.get('rec_sort', 'Best match')}</b></div>",
        unsafe_allow_html=True,
    )

    for offset, row in enumerate(window.itertuples(index=False), start=first + 1):
        _render_card(offset, row)

    if bool(result.get("rating_imputed", pd.Series(dtype=bool)).any()):
        st.caption("\\* rating imputed from the city median — the source "
                   "catalogue had no rating for this venue.")

    st.plotly_chart(
        px.scatter(window, x="cost", y="rating", size="similarity",
                   color="city", hover_name="name",
                   title=f"Recommended restaurants (page {page + 1}) — "
                         "price versus rating"),
        width="stretch",
    )

    with st.expander(f"Raw result table (all {total} matches)"):
        st.dataframe(
            result[["name", "city", "cuisine", "rating", "rating_count",
                    "cost", "similarity"]],
            width="stretch", hide_index=True,
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
        width="stretch", hide_index=True, height=460,
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
        width="stretch",
    )
    right.plotly_chart(
        px.bar(cities.sort_values("avg_cost"), x="avg_cost", y="city",
               orientation="h", color="avg_rating",
               color_continuous_scale="Plasma",
               title="Average cost for two by city"),
        width="stretch",
    )

    st.plotly_chart(
        px.scatter(cuisines.head(40), x="avg_cost", y="avg_rating",
                   size="restaurants", hover_name="cuisine_tag",
                   color="cuisine_tag",
                   title="Cuisine positioning — price versus rating (top 40)"),
        width="stretch",
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
        width="stretch",
    )

    st.dataframe(cities, width="stretch", hide_index=True)


def page_clusters(cleaned: pd.DataFrame, encoded: sparse.csr_matrix) -> None:
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
        width="stretch",
    )
    st.dataframe(profile, width="stretch", hide_index=True)

    chosen = st.selectbox("Inspect a cluster", sorted(labelled["cluster"].unique()))
    st.dataframe(
        labelled[labelled["cluster"] == chosen][
            ["name", "city", "cuisine", "rating", "cost"]
        ].head(1000),
        width="stretch", hide_index=True, height=400,
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
