"""Render static figures for the README and the project report.

Cleaning outcomes, rating/cost distributions, cuisine and city coverage, and
cluster sizes. Run after the pipeline and models:

    python data_pipeline.py
    python models.py
    python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2E5E8A", "#C1666B", "#4E9F6E", "#D4A24C", "#7C6A9B", "#5B8C93"]
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[fig] {name}")


def main() -> None:
    frame = pd.read_csv(config.CLEANED_CSV, low_memory=False)

    # 1. Rating distribution, split by whether the value was imputed.
    #    58.6% of ratings were missing, so this is the most important
    #    caveat in the whole dataset and it belongs in the README.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    real = frame.loc[~frame["rating_imputed"].astype(bool), "rating"]
    imputed = frame.loc[frame["rating_imputed"].astype(bool), "rating"]
    ax.hist([real, imputed], bins=30, stacked=True,
            color=[PALETTE[0], PALETTE[3]],
            label=[f"observed (n={len(real):,})",
                   f"imputed (n={len(imputed):,})"])
    ax.set_title("Rating distribution - observed vs imputed")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Restaurants")
    ax.legend(fontsize=8)
    save(fig, "01_rating_distribution.png")

    # 2. Cost distribution
    cost = pd.to_numeric(frame["cost"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist(cost.clip(upper=cost.quantile(0.99)), bins=50, color=PALETTE[1])
    ax.set_title("Cost for two (99th percentile clipped)")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Restaurants")
    save(fig, "02_cost_distribution.png")

    # 3. Top cuisines
    cuisines = (frame["cuisine"].fillna("").str.split(",")
                .explode().str.strip())
    cuisines = cuisines[cuisines != ""]
    top = cuisines.value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(top.index, top.values, color=PALETTE[2])
    ax.set_title("Top 15 cuisines by restaurant count")
    ax.set_xlabel("Restaurants")
    save(fig, "03_top_cuisines.png")

    # 4. Top cities
    city_column = "base_city" if "base_city" in frame.columns else "city"
    cities = frame[city_column].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(cities.index, cities.values, color=PALETTE[0])
    ax.set_title("Top 15 cities by coverage")
    ax.set_xlabel("Restaurants")
    save(fig, "04_top_cities.png")

    # 5. Average rating vs cost for the largest cities
    big = frame[frame[city_column].isin(cities.index)]
    summary = (big.groupby(city_column)
               .agg(avg_rating=("rating", "mean"),
                    avg_cost=("cost", "mean"),
                    restaurants=("id", "count"))
               .reset_index())
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(summary["avg_cost"], summary["avg_rating"],
               s=summary["restaurants"] / 40, alpha=0.6, color=PALETTE[4])
    for _, row in summary.iterrows():
        ax.annotate(row[city_column], (row["avg_cost"], row["avg_rating"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Average cost for two")
    ax.set_ylabel("Average rating")
    ax.set_title("Rating vs cost by city (bubble = restaurant count)")
    save(fig, "05_rating_vs_cost_by_city.png")

    # 6. Cluster sizes
    try:
        import models as project_models

        assignments = project_models.cluster_assignments()
        labels = (assignments["cluster"] if hasattr(assignments, "columns")
                  else pd.Series(assignments))
        sizes = labels.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(sizes.index.astype(str), sizes.values, color=PALETTE[5])
        ax.set_title("MiniBatchKMeans cluster sizes (k=12)")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Restaurants")
        save(fig, "06_cluster_sizes.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[fig] skipped cluster sizes: {exc}")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
